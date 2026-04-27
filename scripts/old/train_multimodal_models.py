"""
Train multimodal Airbnb price models from precomputed embeddings.

Models trained in this script:
1) Text-only regressor (DistilBERT embeddings)
2) Image+Text fusion regressor (CLIP vision + DistilBERT text embeddings)

Both models:
- Train on Box-Cox transformed price (price_bc)
- Evaluate in raw dollars via inverse transform
- Save metrics to outputs/model_runs.csv
- Save checkpoints + learning curves via ExperimentTracker
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

try:
    from utils import ExperimentTracker
except ImportError:
    from scripts.utils import ExperimentTracker

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EMBED_DIR = Path("data/embeddings")
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL_TAG = "distilbert-base-multilingual-cased"
VISION_MODEL_TAG = "clip_vision"
TABULAR_COLS = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "minimum_nights",
    "season_ordinal",
    "beds",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "availability_365",
    "number_of_reviews",
]


@dataclass
class SplitData:
    x_text: np.ndarray
    x_vision: np.ndarray | None
    x_tabular: np.ndarray | None
    y_bc: np.ndarray
    y_raw: np.ndarray
    ids: np.ndarray


def _load_tabular_target(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{split}_tabular.parquet", columns=["id", "price", "price_bc"])
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def _load_tabular_features(split: str) -> pd.DataFrame:
    cols = ["id", *TABULAR_COLS]
    df = pd.read_parquet(DATA_DIR / f"{split}_tabular.parquet", columns=cols)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    for c in TABULAR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[TABULAR_COLS] = df[TABULAR_COLS].fillna(0.0).astype(np.float32)
    return df


def _load_text_embeddings(split: str) -> tuple[np.ndarray, pd.DataFrame]:
    x_text = np.load(EMBED_DIR / f"{split}_{TEXT_MODEL_TAG}.npy")
    text_ids = pd.read_csv(EMBED_DIR / f"{split}_{TEXT_MODEL_TAG}_ids.csv")
    text_ids = text_ids.rename(columns={"id": "id"}).reset_index(drop=True)
    text_ids["text_idx"] = np.arange(len(text_ids), dtype=np.int64)
    return x_text.astype(np.float32), text_ids[["id", "text_idx"]]


def _load_vision_embeddings(split: str) -> tuple[np.ndarray, pd.DataFrame]:
    x_vis = np.load(EMBED_DIR / f"{split}_{VISION_MODEL_TAG}.npy")
    primary_ids_path = EMBED_DIR / f"{split}_{VISION_MODEL_TAG}_ids.csv"
    fallback_ids_path = EMBED_DIR / f"{split}_clip_ids.csv"
    if primary_ids_path.exists():
        vis_ids = pd.read_csv(primary_ids_path)
    elif fallback_ids_path.exists():
        vis_ids = pd.read_csv(fallback_ids_path)
    else:
        raise FileNotFoundError(
            f"Could not find vision ids file. Tried: {primary_ids_path} and {fallback_ids_path}"
        )
    vis_ids = vis_ids.rename(columns={"id": "id"}).reset_index(drop=True)

    # Vision embeddings are often one row per image, not one row per listing.
    # Aggregate to one listing-level vector by averaging all images for the same listing id.
    group = vis_ids.groupby("id", sort=False).indices
    unique_ids = []
    aggregated = np.zeros((len(group), x_vis.shape[1]), dtype=np.float32)

    for out_idx, (listing_id, row_indices) in enumerate(group.items()):
        aggregated[out_idx] = x_vis[row_indices].mean(axis=0, dtype=np.float32)
        unique_ids.append(listing_id)

    vis_listing_ids = pd.DataFrame({
        "id": np.array(unique_ids),
        "vision_idx": np.arange(len(unique_ids), dtype=np.int64),
    })

    return aggregated, vis_listing_ids[["id", "vision_idx"]]


def prepare_split(split: str, need_vision: bool, need_tabular: bool = False) -> SplitData:
    targets = _load_tabular_target(split)
    x_text, text_ids = _load_text_embeddings(split)

    merged = targets.merge(text_ids, on="id", how="inner")

    x_vision = None
    if need_vision:
        x_vis_all, vis_ids = _load_vision_embeddings(split)
        merged = merged.merge(vis_ids, on="id", how="inner")
        x_vision = x_vis_all[merged["vision_idx"].to_numpy()]

    x_tabular = None
    if need_tabular:
        tab_df = _load_tabular_features(split)
        merged = merged.merge(tab_df, on="id", how="inner")
        x_tabular = merged[TABULAR_COLS].to_numpy(dtype=np.float32)

    x_text_aligned = x_text[merged["text_idx"].to_numpy()]
    y_bc = merged["price_bc"].to_numpy(dtype=np.float32)
    y_raw = merged["price"].to_numpy(dtype=np.float32)
    ids = merged["id"].to_numpy()

    return SplitData(
        x_text=x_text_aligned,
        x_vision=x_vision,
        x_tabular=x_tabular,
        y_bc=y_bc,
        y_raw=y_raw,
        ids=ids,
    )


class EmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()

        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tracker: ExperimentTracker,
    epochs: int,
    lr: float,
    patience: int,
) -> tuple[nn.Module, float]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).unsqueeze(1)
                pred = model(xb)
                val_running += criterion(pred, yb).item() * len(xb)

        val_loss = val_running / len(val_loader.dataset)
        scheduler.step(val_loss)

        tracker.log_epoch(epoch, train_loss, val_loss, lr=optimizer.param_groups[0]["lr"])

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        tracker.save_checkpoint(model, optimizer, epoch, val_loss, is_best=is_best)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    tracker.plot_curves()
    return model, best_val


def evaluate(model: nn.Module, loader: DataLoader, y_raw: np.ndarray, price_transformer) -> tuple[float, float, float]:
    model.eval()
    preds_bc = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb).squeeze(1).cpu().numpy()
            preds_bc.append(pred)

    y_pred_bc = np.concatenate(preds_bc)
    y_pred_raw = price_transformer.inverse_transform(y_pred_bc.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_raw, y_pred_raw)))
    mae = float(mean_absolute_error(y_raw, y_pred_raw))
    r2 = float(r2_score(y_raw, y_pred_raw))
    return rmse, mae, r2


def run_experiment(
    model_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y_bc: np.ndarray,
    val_y_bc: np.ndarray,
    test_y_bc: np.ndarray,
    val_y_raw: np.ndarray,
    test_y_raw: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    price_transformer,
) -> dict:
    train_ds = EmbeddingDataset(train_x, train_y_bc)
    val_ds = EmbeddingDataset(val_x, val_y_bc)
    test_ds = EmbeddingDataset(test_x, test_y_bc)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    tracker = ExperimentTracker(experiment_name=model_name, base_dir=str(MODELS_DIR))

    model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    val_rmse, val_mae, val_r2 = evaluate(model, val_loader, val_y_raw, price_transformer)
    test_rmse, test_mae, test_r2 = evaluate(model, test_loader, test_y_raw, price_transformer)

    print("-" * 90)
    print(f"{model_name} | Val RMSE=${val_rmse:.2f} MAE=${val_mae:.2f} R2={val_r2:.4f}")
    print(f"{model_name} | Test RMSE=${test_rmse:.2f} MAE=${test_mae:.2f} R2={test_r2:.4f}")
    print("-" * 90)

    return {
        "model_name": model_name,
        "val_rmse_raw": val_rmse,
        "val_mae_raw": val_mae,
        "val_r2_raw": val_r2,
        "test_rmse_raw": test_rmse,
        "test_mae_raw": test_mae,
        "test_r2_raw": test_r2,
        "best_params": f"hidden={hidden_dims},dropout={dropout},lr={lr},epochs={epochs},batch={batch_size},best_val_loss={best_val_loss:.6f}",
    }


def main() -> None:
    print("=" * 90)
    print("LOADING TARGETS + EMBEDDINGS")
    print("=" * 90)

    price_transformer = joblib.load(DATA_DIR / "price_transformer.joblib")

    train_text = prepare_split("train", need_vision=False)
    val_text = prepare_split("val", need_vision=False)
    test_text = prepare_split("test", need_vision=False)

    train_fusion = prepare_split("train", need_vision=True)
    val_fusion = prepare_split("val", need_vision=True)
    test_fusion = prepare_split("test", need_vision=True)

    train_full = prepare_split("train", need_vision=True, need_tabular=True)
    val_full = prepare_split("val", need_vision=True, need_tabular=True)
    test_full = prepare_split("test", need_vision=True, need_tabular=True)

    print(
        f"Text-only rows: train={len(train_text.ids)} val={len(val_text.ids)} test={len(test_text.ids)}"
    )
    print(
        f"Fusion rows:    train={len(train_fusion.ids)} val={len(val_fusion.ids)} test={len(test_fusion.ids)}"
    )
    print(
        f"Full rows:      train={len(train_full.ids)} val={len(val_full.ids)} test={len(test_full.ids)}"
    )

    results = []

    results.append(
        run_experiment(
            model_name="Text-Only MLP (DistilBERT)",
            train_x=train_text.x_text,
            val_x=val_text.x_text,
            test_x=test_text.x_text,
            train_y_bc=train_text.y_bc,
            val_y_bc=val_text.y_bc,
            test_y_bc=test_text.y_bc,
            val_y_raw=val_text.y_raw,
            test_y_raw=test_text.y_raw,
            input_dim=train_text.x_text.shape[1],
            hidden_dims=[512, 256, 64],
            dropout=0.25,
            lr=1e-3,
            epochs=80,
            batch_size=256,
            patience=12,
            price_transformer=price_transformer,
        )
    )

    train_fusion_x = np.concatenate([train_fusion.x_text, train_fusion.x_vision], axis=1)
    val_fusion_x = np.concatenate([val_fusion.x_text, val_fusion.x_vision], axis=1)
    test_fusion_x = np.concatenate([test_fusion.x_text, test_fusion.x_vision], axis=1)

    results.append(
        run_experiment(
            model_name="Image+Text Fusion MLP (CLIP+DistilBERT)",
            train_x=train_fusion_x,
            val_x=val_fusion_x,
            test_x=test_fusion_x,
            train_y_bc=train_fusion.y_bc,
            val_y_bc=val_fusion.y_bc,
            test_y_bc=test_fusion.y_bc,
            val_y_raw=val_fusion.y_raw,
            test_y_raw=test_fusion.y_raw,
            input_dim=train_fusion_x.shape[1],
            hidden_dims=[768, 256, 64],
            dropout=0.30,
            lr=8e-4,
            epochs=90,
            batch_size=256,
            patience=14,
            price_transformer=price_transformer,
        )
    )

    train_full_x = np.concatenate([train_full.x_tabular, train_full.x_text, train_full.x_vision], axis=1)
    val_full_x = np.concatenate([val_full.x_tabular, val_full.x_text, val_full.x_vision], axis=1)
    test_full_x = np.concatenate([test_full.x_tabular, test_full.x_text, test_full.x_vision], axis=1)

    results.append(
        run_experiment(
            model_name="Tabular+Image+Text Fusion MLP",
            train_x=train_full_x,
            val_x=val_full_x,
            test_x=test_full_x,
            train_y_bc=train_full.y_bc,
            val_y_bc=val_full.y_bc,
            test_y_bc=test_full.y_bc,
            val_y_raw=val_full.y_raw,
            test_y_raw=test_full.y_raw,
            input_dim=train_full_x.shape[1],
            hidden_dims=[1024, 256, 64],
            dropout=0.30,
            lr=7e-4,
            epochs=100,
            batch_size=256,
            patience=14,
            price_transformer=price_transformer,
        )
    )

    summary_df = pd.DataFrame(results)
    summary_df["timestamp"] = datetime.now().isoformat()
    summary_df["train_size"] = len(train_text.ids)
    summary_df["val_size"] = len(val_text.ids)
    summary_df["test_size"] = len(test_text.ids)

    csv_path = OUTPUTS_DIR / "model_runs.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("MULTIMODAL TRAINING SUMMARY")
    print("=" * 90)
    print(pd.DataFrame(results)[["model_name", "test_rmse_raw", "test_mae_raw", "test_r2_raw"]].to_string(index=False))
    print(f"\nSaved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
