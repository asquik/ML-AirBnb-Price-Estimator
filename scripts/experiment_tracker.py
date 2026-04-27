"""
ExperimentTracker — the single shared utility for all model training scripts.

Every training script must use this class. It handles:
  - Creating the run folder under outputs/runs/
  - Writing config.json at the START of the run (so partial runs leave a trace)
  - Epoch-level logging for DL models (history.json)
  - Saving best PyTorch model weights (best_model.pth)
  - Saving predictions.npz with all splits in raw dollars
  - Saving optional extra artifacts (feature_importance.json, etc.)
  - Appending exactly one row to outputs/master_runs_log.csv on finish()

See Model Training Specification.md for the full contract this class implements.
"""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RUNS_DIR = Path("outputs/runs")
MASTER_LOG = Path("outputs/master_runs_log.csv")

# Canonical column order for master_runs_log.csv — must never change.
# New runs appended after this list was established will have NaN in missing cols.
CSV_COLUMNS = [
    "run_id",
    "model_type",
    "modalities",
    "dataset_variant",
    "image_size",
    "fusion_head",
    "lora_applied_to",
    "lora_rank",
    "trainable_parameters",
    "batch_size",
    "train_rmse_raw",
    "val_rmse_raw",
    "test_rmse_raw",
    "test_mae_raw",
    "test_r2",
    "training_time_minutes",
    "avg_time_per_epoch_sec",
    "peak_vram_gb",
    "peak_ram_gb",
    "best_hyperparams",
    "run_name",
    "artifact_folder",
    "notes",
]

VALID_VARIANTS = {"normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"}
VALID_MODEL_TYPES = {
    "DecisionTree", "RandomForest", "GradientBoosting", "LightGBM",
    "Ridge", "PolynomialRidge",
    "TabularMLP",
    "TextMLP", "ImageMLP", "FusionMLP",
    "TextLoRA", "ImageLoRA", "FusionLoRA",
}
VALID_MODALITIES = {"tabular", "tab+text", "tab+image", "tab+text+image"}


class ExperimentTracker:
    """
    Usage pattern (every training script follows this exactly):

        tracker = ExperimentTracker(
            model_type="LightGBM",
            modalities="tabular",
            variant="cleaned_raw",
            run_name="leaves128_lr001",         # from --run-name CLI arg
            config={"num_leaves": 128, ...},    # model hyperparameters
            # DL-only optional args:
            fusion_head=None,
            image_size=None,
            lora_applied_to=None,
            lora_rank=None,
            batch_size=None,
            dataloader_workers=None,
            device_used=None,
        )

        # DL only — call once per epoch inside your training loop:
        tracker.log_epoch(train_loss=0.34, val_loss=0.41, val_rmse_raw=82.3)

        # DL only — call when val loss improves:
        tracker.save_best_model(model)

        # Call once at the very end, after all evaluation is complete:
        tracker.finish(
            train_metrics={"rmse": 91.2, "mae": 58.1, "r2": 0.41},
            val_metrics={"rmse": 88.4, "mae": 55.2, "r2": 0.43},
            test_metrics={"rmse": 87.1, "mae": 54.0, "r2": 0.44},
            predictions={
                "train_y_true": arr, "train_y_pred": arr,
                "val_y_true": arr,   "val_y_pred": arr,
                "test_y_true": arr,  "test_y_pred": arr,
            },
            trainable_parameters=5000,
            training_time_minutes=12.4,
            best_hyperparams={"num_leaves": 128, "learning_rate": 0.01},
            # optional:
            peak_vram_gb=None,
            extra_artifacts={"feature_importance.json": {"room_type": 0.14, ...}},
        )

    All metrics passed to finish() must already be in raw Canadian dollars.
    """

    def __init__(
        self,
        model_type: str,
        modalities: str,
        variant: str,
        run_name: str = "",
        config: dict[str, Any] | None = None,
        fusion_head: str | None = None,
        image_size: int | None = None,
        lora_applied_to: str | None = None,
        lora_rank: int | None = None,
        batch_size: int | None = None,
        dataloader_workers: int | None = None,
        device_used: str | None = None,
        is_smoke_test: bool = False,
    ) -> None:
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type '{model_type}' not in known types: {VALID_MODEL_TYPES}")
        if modalities not in VALID_MODALITIES:
            raise ValueError(f"modalities '{modalities}' not in: {VALID_MODALITIES}")
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant '{variant}' not in: {VALID_VARIANTS}")

        self.model_type = model_type
        self.modalities = modalities
        self.variant = variant
        self.run_name = run_name
        self.fusion_head = fusion_head
        self.image_size = image_size
        self.lora_applied_to = lora_applied_to
        self.lora_rank = lora_rank
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.device_used = device_used or ("cuda:0" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.is_smoke_test = is_smoke_test

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        prefix = "SMOKE_run" if is_smoke_test else "run"
        self.run_id = f"{prefix}_{ts}"
        folder_name = f"{self.run_id}_{model_type}"
        if run_name:
            folder_name = f"{folder_name}_{run_name}"

        self.run_dir = RUNS_DIR / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        self._epoch_times: list[float] = []
        self._epoch_start: float | None = None

        self._history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse_raw": [],
        }

        # Reset VRAM counter so peak reflects only this run
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Write config.json immediately so partial runs leave a trace
        full_config = {
            "run_id": self.run_id,
            "model_type": model_type,
            "modalities": modalities,
            "dataset_variant": variant,
            "image_size": image_size,
            "fusion_head": fusion_head,
            "lora_applied_to": lora_applied_to,
            "lora_rank": lora_rank,
            "seed": 42,
            "target_column": "price_bc" if variant.endswith("_bc") else "price",
            "target_transform": "box_cox" if variant.endswith("_bc") else "none",
            "box_cox_lambda": None,  # populated by training script via tracker.set_box_cox_lambda()
            "run_name": run_name,
            "host_machine": socket.gethostname(),
            "device_used": self.device_used,
            "dataloader_workers": dataloader_workers,
            "started_at": datetime.now().isoformat(),
            **(config or {}),
        }
        self._write_json("config.json", full_config)

    def set_box_cox_lambda(self, lam: float) -> None:
        """
        Must be called for *_bc variants immediately after loading price_transformer.
        Reads back config.json, updates box_cox_lambda, rewrites it.
        """
        config_path = self.run_dir / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["box_cox_lambda"] = lam
        self._write_json("config.json", cfg)

    # ------------------------------------------------------------------
    # Epoch-level API (DL models only)
    # ------------------------------------------------------------------

    def start_epoch(self) -> None:
        """Call at the beginning of each epoch to track epoch wall time."""
        self._epoch_start = time.time()

    def log_epoch(
        self,
        train_loss: float,
        val_loss: float,
        val_rmse_raw: float,
    ) -> None:
        """
        Call once per epoch, after validation. val_rmse_raw must be in raw dollars
        (inverse-transformed if *_bc variant) so learning curves are directly readable.
        """
        self._history["train_loss"].append(float(train_loss))
        self._history["val_loss"].append(float(val_loss))
        self._history["val_rmse_raw"].append(float(val_rmse_raw))

        if self._epoch_start is not None:
            self._epoch_times.append(time.time() - self._epoch_start)
            self._epoch_start = None

        self._write_json("history.json", self._history)

    def save_best_model(self, model: Any) -> None:
        """
        Save PyTorch model weights whenever val loss improves.
        For LoRA models, saves only the adapter weights (model.state_dict() on a
        PEFT model already contains only adapter parameters).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        torch.save(model.state_dict(), self.run_dir / "best_model.pth")

    # ------------------------------------------------------------------
    # Finish — call once, at the very end
    # ------------------------------------------------------------------

    def finish(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        predictions: dict[str, np.ndarray],
        trainable_parameters: int,
        training_time_minutes: float,
        best_hyperparams: dict[str, Any],
        peak_vram_gb: float | None = None,
        extra_artifacts: dict[str, Any] | None = None,
    ) -> None:
        """
        Saves predictions.npz, optional extra artifacts, and appends one row
        to master_runs_log.csv. All metric values must be in raw Canadian dollars.

        predictions dict must contain keys:
            train_y_true, train_y_pred, val_y_true, val_y_pred,
            test_y_true, test_y_pred
        All arrays in raw dollars.
        """
        # Save predictions
        np.savez(
            self.run_dir / "predictions.npz",
            train_y_true=predictions["train_y_true"].astype(np.float32),
            train_y_pred=predictions["train_y_pred"].astype(np.float32),
            val_y_true=predictions["val_y_true"].astype(np.float32),
            val_y_pred=predictions["val_y_pred"].astype(np.float32),
            test_y_true=predictions["test_y_true"].astype(np.float32),
            test_y_pred=predictions["test_y_pred"].astype(np.float32),
        )

        # Save extra artifacts (e.g. feature_importance.json)
        for filename, content in (extra_artifacts or {}).items():
            self._write_json(filename, content)

        # Save final history.json if it has any epochs
        if self._history["train_loss"]:
            self._write_json("history.json", self._history)

        # Measure peak RAM
        peak_ram_gb: float | None = None
        if HAS_PSUTIL:
            rss_bytes = psutil.Process().memory_info().rss
            peak_ram_gb = round(rss_bytes / (1024 ** 3), 3)

        # Measure peak VRAM if not supplied by caller
        if peak_vram_gb is None and HAS_TORCH and torch.cuda.is_available():
            peak_vram_gb = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3
            )

        avg_epoch_sec: float | None = (
            round(sum(self._epoch_times) / len(self._epoch_times), 2)
            if self._epoch_times else None
        )

        row = {
            "run_id": self.run_id,
            "model_type": self.model_type,
            "modalities": self.modalities,
            "dataset_variant": self.variant,
            "image_size": self.image_size,
            "fusion_head": self.fusion_head,
            "lora_applied_to": self.lora_applied_to,
            "lora_rank": self.lora_rank,
            "trainable_parameters": trainable_parameters,
            "batch_size": self.batch_size,
            "train_rmse_raw": round(train_metrics["rmse"], 4),
            "val_rmse_raw": round(val_metrics["rmse"], 4),
            "test_rmse_raw": round(test_metrics["rmse"], 4),
            "test_mae_raw": round(test_metrics["mae"], 4),
            "test_r2": round(test_metrics["r2"], 6),
            "training_time_minutes": round(training_time_minutes, 3),
            "avg_time_per_epoch_sec": avg_epoch_sec,
            "peak_vram_gb": peak_vram_gb,
            "peak_ram_gb": peak_ram_gb,
            "best_hyperparams": json.dumps(best_hyperparams),
            "run_name": self.run_name,
            "artifact_folder": str(self.run_dir),
            "notes": "",
        }

        if self.is_smoke_test:
            print(f"\n✅ SMOKE TEST complete — artifacts at {self.run_dir}")
            print(f"   (smoke test results are NOT written to master_runs_log.csv)")
            print(f"   test RMSE ${row['test_rmse_raw']:.2f} | MAE ${row['test_mae_raw']:.2f} | R² {row['test_r2']:.4f}")
        else:
            self._append_to_master_log(row)
            print(f"\n✅ Run complete — artifacts at {self.run_dir}")
            print(f"   test RMSE ${row['test_rmse_raw']:.2f} | MAE ${row['test_mae_raw']:.2f} | R² {row['test_r2']:.4f}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_json(self, filename: str, data: Any) -> None:
        with open(self.run_dir / filename, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    @staticmethod
    def _append_to_master_log(row: dict[str, Any]) -> None:
        MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)

        new_row = pd.DataFrame([row])

        if MASTER_LOG.exists():
            existing = pd.read_csv(MASTER_LOG)
            # Add any columns present in new_row but missing from existing
            for col in new_row.columns:
                if col not in existing.columns:
                    existing[col] = None
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        # Enforce canonical column order; unknown extra columns go at the end
        ordered = [c for c in CSV_COLUMNS if c in combined.columns]
        extras = [c for c in combined.columns if c not in CSV_COLUMNS]
        combined = combined[ordered + extras]

        combined.to_csv(MASTER_LOG, index=False)


def _json_default(obj: Any) -> Any:
    """Allow numpy scalars and arrays to serialize to JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
