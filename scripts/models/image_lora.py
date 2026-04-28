"""
ImageLoRA — CLIP vision encoder + tabular with LoRA fine-tuning.
Registry run #33: variant=normal_bc, head=deep_256, image=336px, rank=16

Architecture:
  - Image encoder: CLIP ViT-L/14 @336  (LoRA on q_proj, v_proj)
  - Tabular branch: Linear(16 → 64) + ReLU
  - Fusion head: configurable MLP → scalar price

Target: price_bc (MSE loss). Inverse-transformed for all metrics.
Sample weights applied per-sample before loss mean.
All data preloaded into RAM per Rule 8.

Usage:
    python scripts/models/image_lora.py --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 --run-name "ablation"
    python scripts/models/image_lora.py --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 --smoke-test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModel

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracker import ExperimentTracker

DATA_DIR   = Path("data")
IMAGE_BASE = Path("images")

TABULAR_COLS = [
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable",
    "accommodates", "bathrooms", "bedrooms", "beds", "host_total_listings_count",
    "latitude", "longitude", "minimum_nights", "availability_365",
    "number_of_reviews", "season_ordinal", "has_valid_image",
]

CLIP_224_ID  = "openai/clip-vit-base-patch32"
CLIP_336_ID  = "openai/clip-vit-large-patch14-336"
FUSION_HEADS = {
    "shallow_64": [64],
    "medium_128": [128],
    "deep_256":   [256, 128],
    "deep_512":   [512, 256],
}
MAX_EPOCHS        = 20
EARLY_STOP_PAT    = 5
LORA_ALPHA_MULT   = 2
LORA_DROPOUT      = 0.05
TABULAR_EMBED_DIM = 64
DROPOUT_FUSION    = 0.15
LOG_EVERY         = 50

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageLoRADataset(Dataset):
    def __init__(
        self,
        tab_df: pd.DataFrame,
        image_dir: Path,
        image_size: int,
        target_col: str,
    ) -> None:
        n = len(tab_df)
        listing_ids = tab_df["listing_id"].to_numpy()

        self.tabular        = tab_df[TABULAR_COLS].to_numpy(dtype=np.float32)
        self.targets        = tab_df[target_col].to_numpy(dtype=np.float32)
        self.sample_weights = tab_df["sample_weight"].to_numpy(dtype=np.float32)

        print(f"  Loading {n} images from {image_dir}...", flush=True)
        imgs = torch.empty(n, 3, image_size, image_size, dtype=torch.float16)

        img_index: dict[str, Path] = {}
        if image_dir.exists():
            for p in image_dir.glob("*.jpg"):
                img_index[p.stem] = p

        placeholder = self._make_placeholder(image_size)
        for i, lid in enumerate(listing_ids):
            p = img_index.get(str(lid))
            if p is not None:
                try:
                    img = Image.open(p).convert("RGB")
                    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
                    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                    imgs[i] = arr.half()
                    continue
                except Exception:
                    pass
            imgs[i] = placeholder.half()

        self.images = imgs

    @staticmethod
    def _make_placeholder(image_size: int) -> torch.Tensor:
        arr = IMAGENET_MEAN.expand(3, image_size, image_size).clone()
        return (arr - IMAGENET_MEAN) / IMAGENET_STD

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return {
            "image":         self.images[idx].float(),
            "tabular":       torch.from_numpy(self.tabular[idx]),
            "target":        torch.tensor(self.targets[idx]),
            "sample_weight": torch.tensor(self.sample_weights[idx]),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_fusion_head(input_dim: int, hidden_sizes: list[int]) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(DROPOUT_FUSION)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class ImageLoRAModel(nn.Module):
    def __init__(self, image_model_id: str, tabular_dim: int, fusion_head_sizes: list[int]) -> None:
        super().__init__()
        img_dtype = torch.float16 if image_model_id == CLIP_336_ID else None
        self.image_encoder = CLIPVisionModel.from_pretrained(
            image_model_id, use_safetensors=True,
            **({"dtype": img_dtype} if img_dtype is not None else {}),
        )
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        image_dim = self.image_encoder.config.hidden_size  # 768 for both variants
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, TABULAR_EMBED_DIM), nn.ReLU()
        )
        self.fusion_head = _build_fusion_head(image_dim + TABULAR_EMBED_DIM, fusion_head_sizes)

    def forward(self, image, tabular):
        img_out   = self.image_encoder(pixel_values=image)
        img_embed = img_out.pooler_output
        tab_embed = self.tabular_branch(tabular)
        return self.fusion_head(torch.cat([img_embed, tab_embed], dim=1))


def apply_lora(model: ImageLoRAModel, lora_rank: int) -> int:
    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * LORA_ALPHA_MULT,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # No task_type — prevents PEFT from injecting NLP kwargs into vision forward
        target_modules=["q_proj", "v_proj"],
    )
    model.image_encoder = get_peft_model(model.image_encoder, cfg)
    for p in model.tabular_branch.parameters():
        p.requires_grad = True
    for p in model.fusion_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied — trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return trainable


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def weighted_mse(preds, targets, weights):
    return ((preds.squeeze(1) - targets) ** 2 * weights).mean()


def run_epoch(model, loader, optimizer, device, accum_steps, training: bool) -> float:
    model.train() if training else model.eval()
    total_loss = 0.0
    if training:
        optimizer.zero_grad()

    outer_ctx = nullcontext() if training else torch.no_grad()
    amp_ctx   = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with outer_ctx:
        for step, batch in enumerate(loader):
            imgs = batch["image"].to(device, non_blocking=True)
            tab  = batch["tabular"].to(device, non_blocking=True)
            tgt  = batch["target"].to(device, non_blocking=True)
            sw   = batch["sample_weight"].to(device, non_blocking=True)

            with amp_ctx:
                preds = model(imgs, tab)
            loss  = weighted_mse(preds.float(), tgt, sw)
            if training:
                loss = loss / accum_steps
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * accum_steps
                if (step + 1) % LOG_EVERY == 0:
                    print(f"    step {step+1}/{len(loader)}  loss={loss.item()*accum_steps:.4f}", flush=True)
            else:
                total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict_raw(model, loader, device, price_transformer, target_col):
    model.eval()
    all_preds, all_true = [], []
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        tab  = batch["tabular"].to(device, non_blocking=True)
        with amp_ctx:
            preds = model(imgs, tab).float().squeeze(1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(batch["target"].numpy())

    preds_np = np.concatenate(all_preds)
    true_np  = np.concatenate(all_true)
    if target_col == "price_bc" and price_transformer is not None:
        preds_np = price_transformer.inverse_transform(preds_np.reshape(-1, 1)).ravel()
        true_np  = price_transformer.inverse_transform(true_np.reshape(-1, 1)).ravel()
    return true_np, preds_np


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ImageLoRA")
    parser.add_argument("--variant",     required=True, choices=["normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"])
    parser.add_argument("--image-size",  type=int, default=336, choices=[224, 336], dest="image_size")
    parser.add_argument("--lora-rank",   type=int, default=16, dest="lora_rank")
    parser.add_argument("--fusion-head", type=str, default="deep_256", dest="fusion_head", choices=list(FUSION_HEADS.keys()))
    parser.add_argument("--batch-size",  type=int, default=16, dest="batch_size")
    parser.add_argument("--accum-steps", type=int, default=2,  dest="accum_steps")
    parser.add_argument("--lr-adapters", type=float, default=2e-5, dest="lr_adapters")
    parser.add_argument("--lr-head",     type=float, default=5e-4, dest="lr_head")
    parser.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS, dest="max_epochs")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--run-name",    type=str, default="", dest="run_name")
    parser.add_argument("--smoke-test",  action="store_true", dest="smoke_test")
    args = parser.parse_args()

    variant    = args.variant
    target_col = "price_bc" if variant.endswith("_bc") else "price"
    suffix     = "_cleaned" if variant.startswith("cleaned") else ""
    clip_id    = CLIP_336_ID if args.image_size == 336 else CLIP_224_ID
    image_dir  = IMAGE_BASE / f"processed_{args.image_size}"
    head_sizes = FUSION_HEADS[args.fusion_head]
    eff_batch  = args.batch_size * args.accum_steps
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*70}")
    print(f"ImageLoRA | variant={variant} | image={args.image_size}px | rank={args.lora_rank} | head={args.fusion_head}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"{'='*70}\n")

    print("Loading parquets...")
    train_tab = pd.read_parquet(DATA_DIR / f"train{suffix}_tabular.parquet")
    val_tab   = pd.read_parquet(DATA_DIR / f"val{suffix}_tabular.parquet")
    test_tab  = pd.read_parquet(DATA_DIR / f"test{suffix}_tabular.parquet")

    price_transformer = None
    if target_col == "price_bc":
        price_transformer = joblib.load(DATA_DIR / f"price_transformer{suffix}.joblib")

    if args.smoke_test:
        print("  [SMOKE TEST] truncating all splits to 100 rows")
        train_tab = train_tab.iloc[:100].copy()
        val_tab   = val_tab.iloc[:100].copy()
        test_tab  = test_tab.iloc[:100].copy()

    print(f"  train={len(train_tab)}  val={len(val_tab)}  test={len(test_tab)}")

    tracker = ExperimentTracker(
        model_type="ImageLoRA", modalities="tab+image", variant=variant,
        run_name=args.run_name, is_smoke_test=args.smoke_test,
        fusion_head=args.fusion_head, image_size=args.image_size,
        lora_applied_to="image", lora_rank=args.lora_rank,
        batch_size=args.batch_size, dataloader_workers=args.workers, device_used=str(device),
        config={
            "lr_adapters": args.lr_adapters, "lr_head": args.lr_head,
            "accum_steps": args.accum_steps, "eff_batch": eff_batch,
            "max_epochs": args.max_epochs, "early_stop_patience": EARLY_STOP_PAT,
            "lora_alpha": args.lora_rank * LORA_ALPHA_MULT, "lora_dropout": LORA_DROPOUT,
            "clip_model": clip_id, "tabular_embed_dim": TABULAR_EMBED_DIM,
            "fusion_head_sizes": head_sizes, "target_column": target_col,
        },
    )
    if price_transformer is not None:
        tracker.set_box_cox_lambda(float(price_transformer.lambdas_[0]))

    print("\nBuilding datasets (preloading into RAM)...")
    train_ds = ImageLoRADataset(train_tab, image_dir, args.image_size, target_col)
    val_ds   = ImageLoRADataset(val_tab,   image_dir, args.image_size, target_col)
    test_ds  = ImageLoRADataset(test_tab,  image_dir, args.image_size, target_col)

    loader_kw = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print("\nLoading encoder and applying LoRA...")
    model            = ImageLoRAModel(clip_id, len(TABULAR_COLS), head_sizes).to(device)
    trainable_params = apply_lora(model, args.lora_rank)

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    adapter_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    head_params    = list(model.tabular_branch.parameters()) + \
                     list(model.fusion_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": args.lr_adapters, "weight_decay": 1e-4},
        {"params": head_params,    "lr": args.lr_head,     "weight_decay": 1e-3},
    ])
    # float32 training — no GradScaler needed

    epochs    = 1 if args.smoke_test else args.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_rmse  = float("inf")
    patience_count = 0
    best_state     = None
    t0             = time.time()

    print(f"\nTraining for up to {epochs} epoch(s)  |  eff_batch={eff_batch}  |  lr_adapters={args.lr_adapters}  |  lr_head={args.lr_head}")

    for epoch in range(epochs):
        tracker.start_epoch()
        _epoch_t0 = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---", flush=True)

        train_loss = run_epoch(model, train_loader, optimizer, device, args.accum_steps, training=True)
        val_loss   = run_epoch(model, val_loader,   None,      device, args.accum_steps, training=False)

        val_true_raw, val_pred_raw = predict_raw(model, val_loader, device, price_transformer, target_col)
        val_rmse_raw = float(np.sqrt(mean_squared_error(val_true_raw, val_pred_raw)))

        scheduler.step()
        current_lr = scheduler.get_last_lr()
        epoch_mins = (time.time() - _epoch_t0) / 60

        tracker.log_epoch(train_loss=train_loss, val_loss=val_loss, val_rmse_raw=val_rmse_raw)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_RMSE=${val_rmse_raw:.2f}  "
              f"epoch_time={epoch_mins:.1f}min  lr={current_lr[0]:.2e}", flush=True)

        torch.save(model.state_dict(), tracker.run_dir / "checkpoint_latest.pth")

        if val_rmse_raw < best_val_rmse:
            best_val_rmse  = val_rmse_raw
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tracker.save_best_model(model)
            print(f"  ✅ New best val RMSE: ${best_val_rmse:.2f}", flush=True)
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOP_PAT})", flush=True)
            if patience_count >= EARLY_STOP_PAT and not args.smoke_test:
                print("  Early stopping.")
                break

    training_time = (time.time() - t0) / 60
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print("\nFinal evaluation...")
    train_true, train_pred = predict_raw(model, train_loader, device, price_transformer, target_col)
    val_true,   val_pred   = predict_raw(model, val_loader,   device, price_transformer, target_col)
    test_true,  test_pred  = predict_raw(model, test_loader,  device, price_transformer, target_col)

    train_metrics = compute_metrics(train_true, train_pred)
    val_metrics   = compute_metrics(val_true,   val_pred)
    test_metrics  = compute_metrics(test_true,  test_pred)

    print(f"  Train — RMSE ${train_metrics['rmse']:.2f}  MAE ${train_metrics['mae']:.2f}  R² {train_metrics['r2']:.4f}")
    print(f"  Val   — RMSE ${val_metrics['rmse']:.2f}  MAE ${val_metrics['mae']:.2f}  R² {val_metrics['r2']:.4f}")
    print(f"  Test  — RMSE ${test_metrics['rmse']:.2f}  MAE ${test_metrics['mae']:.2f}  R² {test_metrics['r2']:.4f}")

    peak_vram = None
    if device.type == "cuda":
        peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        print(f"  Peak VRAM: {peak_vram:.2f} GB")

    tracker.finish(
        train_metrics=train_metrics, val_metrics=val_metrics, test_metrics=test_metrics,
        predictions={
            "train_y_true": train_true, "train_y_pred": train_pred,
            "val_y_true":   val_true,   "val_y_pred":   val_pred,
            "test_y_true":  test_true,  "test_y_pred":  test_pred,
        },
        trainable_parameters=trainable_params,
        training_time_minutes=training_time,
        best_hyperparams={
            "lora_rank": args.lora_rank, "lr_adapters": args.lr_adapters,
            "lr_head": args.lr_head, "batch_size": args.batch_size,
            "accum_steps": args.accum_steps, "fusion_head": args.fusion_head,
            "image_size": args.image_size,
        },
        peak_vram_gb=peak_vram,
    )


if __name__ == "__main__":
    main()
