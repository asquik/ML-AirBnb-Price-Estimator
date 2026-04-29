"""
FusionLoRA — Late-fusion multimodal model with LoRA fine-tuning.
Registry runs: 31 (224px, rank 16), 32 (336px, rank 16), 35 (shallow head, 336px),
               36 (336px, rank 8)

Architecture:
  - Text encoder: distilbert-base-multilingual-cased  (LoRA on q_lin, v_lin)
  - Image encoder: CLIP ViT-B/32 @224 or ViT-L/14 @336  (LoRA on q_proj, v_proj)
  - Tabular branch: Linear(16 → 64) + ReLU
  - Fusion head: configurable depth/width MLP → scalar price

Target: price_bc (Box-Cox) with MSE loss. Inverse-transformed for all metrics.
Sample weights applied per-sample before loss mean.

Per Rule 8: ALL data (tokens, images, tabular) preloaded into RAM in Dataset.__init__.
LoRA note: because encoder weights are being updated, raw data must be loaded —
           pre-computed .npy embeddings are frozen-backbone only and are NOT used here.

Usage:
    python scripts/models/fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 --run-name "priority1"
    python scripts/models/fusion_lora.py --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 --run-name "priority2"
    python scripts/models/fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 --smoke-test
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
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPVisionModel

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracker import ExperimentTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
IMAGE_BASE = Path("images")

TABULAR_COLS = [
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable",
    "accommodates", "bathrooms", "bedrooms", "beds", "host_total_listings_count",
    "latitude", "longitude", "minimum_nights", "availability_365",
    "number_of_reviews", "season_ordinal", "has_valid_image",
]

TEXT_MODEL_ID  = "distilbert-base-multilingual-cased"
CLIP_224_ID    = "openai/clip-vit-base-patch32"
CLIP_336_ID    = "openai/clip-vit-large-patch14-336"

# ImageNet normalisation constants (used to produce 0-tensor for placeholder images)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Fusion head configs: name → list of hidden layer sizes
FUSION_HEADS = {
    "shallow_64": [64],
    "medium_128": [128],
    "deep_256":   [256, 128],
    "deep_512":   [512, 256],
}

MAX_EPOCHS       = 20
EARLY_STOP_PAT   = 5
MAX_TEXT_LEN     = 128
LORA_ALPHA_MULT  = 2   # lora_alpha = rank * LORA_ALPHA_MULT
LORA_DROPOUT     = 0.05
TABULAR_EMBED_DIM = 64
DROPOUT_FUSION   = 0.15
LOG_EVERY        = 50


# ---------------------------------------------------------------------------
# Dataset — lazy image loading with DataLoader prefetch via num_workers
# ---------------------------------------------------------------------------

class FusionLoRADataset(Dataset):
    """
    Text and tabular are tokenized/loaded into RAM once at init.
    Images are loaded from disk on demand in __getitem__ — DataLoader workers
    prefetch the next batch in parallel while the GPU processes the current one.
    """

    def __init__(
        self,
        tab_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        tokenizer,
        image_dir: Path,
        image_size: int,
        target_col: str,
    ) -> None:
        assert len(tab_df) == len(raw_df), "tabular and raw parquet row counts must match"

        n = len(tab_df)
        self.image_size = image_size

        # --- Tabular (already scaled/encoded upstream) ---
        self.tabular = tab_df[TABULAR_COLS].to_numpy(dtype=np.float32)   # (N, 16)
        self.targets = tab_df[target_col].to_numpy(dtype=np.float32)     # (N,)
        self.sample_weights = tab_df["sample_weight"].to_numpy(dtype=np.float32)

        # --- Text: tokenize full_text from raw parquet ---
        print(f"  Tokenizing {n} texts (max_length={MAX_TEXT_LEN})...", flush=True)
        texts = raw_df["full_text"].fillna("").tolist()
        encoded = tokenizer(
            texts,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        self.input_ids      = encoded["input_ids"].astype(np.int64)       # (N, seq)
        self.attention_mask = encoded["attention_mask"].astype(np.int64)  # (N, seq)

        # --- Images: build path index only, load lazily in __getitem__ ---
        print(f"  Indexing {n} images from {image_dir}...", flush=True)
        listing_ids = tab_df["listing_id"].to_numpy()
        img_index: dict[str, Path] = {}
        if image_dir.exists():
            for p in image_dir.glob("*.jpg"):
                img_index[p.stem] = p
        self.image_paths = [img_index.get(str(lid)) for lid in listing_ids]
        self._placeholder: torch.Tensor | None = None

    def _get_placeholder(self) -> torch.Tensor:
        if self._placeholder is None:
            arr = IMAGENET_MEAN.expand(3, self.image_size, self.image_size).clone()
            self._placeholder = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return self._placeholder

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img_tensor = self._get_placeholder()
        if p is not None:
            try:
                img = Image.open(p).convert("RGB")
                arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
                arr = arr.permute(2, 0, 1)
                img_tensor = (arr - IMAGENET_MEAN) / IMAGENET_STD
            except Exception:
                pass
        return {
            "input_ids":      torch.from_numpy(self.input_ids[idx]),
            "attention_mask": torch.from_numpy(self.attention_mask[idx]),
            "image":          img_tensor,
            "tabular":        torch.from_numpy(self.tabular[idx]),
            "target":         torch.tensor(self.targets[idx]),
            "sample_weight":  torch.tensor(self.sample_weights[idx]),
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


class FusionLoRAModel(nn.Module):
    def __init__(
        self,
        image_model_id: str,
        tabular_dim: int,
        fusion_head_sizes: list[int],
        text_proj_dim: int = 0,
        image_proj_dim: int = 0,
    ) -> None:
        super().__init__()

        self.text_encoder  = AutoModel.from_pretrained(TEXT_MODEL_ID, use_safetensors=True)
        # ViT-L/14@336 is ~1.76 GB in float32 — too large for 6 GB VRAM alongside activations.
        # Load it in float16 to halve the footprint; LoRA adapters are cast back to float32 after apply_lora().
        img_dtype = torch.float16 if image_model_id == CLIP_336_ID else None
        self.image_encoder = CLIPVisionModel.from_pretrained(
            image_model_id, use_safetensors=True,
            **({"dtype": img_dtype} if img_dtype is not None else {}),
        )

        # Freeze all encoder params by default; LoRA will selectively unfreeze adapters
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        text_dim  = self.text_encoder.config.hidden_size          # 768 for DistilBERT
        image_dim = self.image_encoder.config.hidden_size         # 768 for both CLIP variants

        # Optional per-modality projection heads (text_proj_dim=0 means no projection)
        if text_proj_dim > 0:
            self.text_proj = nn.Sequential(nn.Linear(text_dim, text_proj_dim), nn.ReLU())
            text_fuse_dim  = text_proj_dim
        else:
            self.text_proj = None
            text_fuse_dim  = text_dim

        if image_proj_dim > 0:
            self.image_proj = nn.Sequential(nn.Linear(image_dim, image_proj_dim), nn.ReLU())
            image_fuse_dim  = image_proj_dim
        else:
            self.image_proj = None
            image_fuse_dim  = image_dim

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, TABULAR_EMBED_DIM),
            nn.ReLU(),
        )

        fusion_input_dim = text_fuse_dim + image_fuse_dim + TABULAR_EMBED_DIM
        self.fusion_head = _build_fusion_head(fusion_input_dim, fusion_head_sizes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        text_out   = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_out.last_hidden_state[:, 0, :]          # [CLS]
        if self.text_proj is not None:
            text_embed = self.text_proj(text_embed)

        img_out    = self.image_encoder(pixel_values=image)
        img_embed  = img_out.pooler_output                        # (B, image_dim)
        if self.image_proj is not None:
            img_embed = self.image_proj(img_embed)

        tab_embed  = self.tabular_branch(tabular)

        fused = torch.cat([text_embed, img_embed, tab_embed], dim=1)
        return self.fusion_head(fused)                            # (B, 1)


def apply_lora(model: FusionLoRAModel, lora_rank: int) -> int:
    """
    Attach LoRA adapters to both encoders. Returns trainable parameter count.

    DistilBERT target modules: q_lin, v_lin  (from old trainer — verified working)
    CLIP vision target modules: q_proj, v_proj  (task_type omitted to avoid NLP kwargs injection)
    """
    alpha = lora_rank * LORA_ALPHA_MULT

    text_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_lin", "v_lin"],
    )
    image_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    model.text_encoder  = get_peft_model(model.text_encoder,  text_cfg)
    model.image_encoder = get_peft_model(model.image_encoder, image_cfg)

    # Ensure non-encoder branches stay trainable
    for p in model.tabular_branch.parameters():
        p.requires_grad = True
    for p in model.fusion_head.parameters():
        p.requires_grad = True
    if model.text_proj is not None:
        for p in model.text_proj.parameters():
            p.requires_grad = True
    if model.image_proj is not None:
        for p in model.image_proj.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied — trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return trainable


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def weighted_mse(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = ((preds.squeeze(1) - targets) ** 2 * weights).mean()
    return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    accum_steps: int,
) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0

    if training:
        optimizer.zero_grad()

    outer_ctx = nullcontext() if training else torch.no_grad()
    amp_ctx   = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    t_epoch = time.time()
    with outer_ctx:
        for step, batch in enumerate(loader):
            t_step = time.time()
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            imgs = batch["image"].to(device, non_blocking=True)
            tab  = batch["tabular"].to(device, non_blocking=True)
            tgt  = batch["target"].to(device, non_blocking=True)
            sw   = batch["sample_weight"].to(device, non_blocking=True)

            with amp_ctx:
                preds = model(ids, mask, imgs, tab)
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
                    elapsed = time.time() - t_epoch
                    secs_per_step = (time.time() - t_step)
                    eta_min = (len(loader) - step - 1) * secs_per_step / 60
                    print(f"    step {step+1}/{len(loader)}  loss={loss.item()*accum_steps:.4f}  "
                          f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min", flush=True)
            else:
                total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict_raw(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    price_transformer,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true_raw, y_pred_raw) in Canadian dollars."""
    model.eval()
    all_preds, all_true = [], []

    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for batch in loader:
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        imgs = batch["image"].to(device, non_blocking=True)
        tab  = batch["tabular"].to(device, non_blocking=True)

        with amp_ctx:
            preds = model(ids, mask, imgs, tab).float().squeeze(1).cpu().numpy()

        all_preds.append(preds)
        all_true.append(batch["target"].numpy())

    preds_np = np.concatenate(all_preds)
    true_np  = np.concatenate(all_true)

    if target_col == "price_bc" and price_transformer is not None:
        preds_np = price_transformer.inverse_transform(preds_np.reshape(-1, 1)).ravel()
        true_np  = price_transformer.inverse_transform(true_np.reshape(-1, 1)).ravel()

    return true_np, preds_np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FusionLoRA — late-fusion multimodal with LoRA fine-tuning")
    parser.add_argument("--variant",     required=True,
                        choices=["normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"])
    parser.add_argument("--image-size",  type=int, default=224, choices=[224, 336], dest="image_size")
    parser.add_argument("--lora-rank",   type=int, default=16, dest="lora_rank")
    parser.add_argument("--fusion-head", type=str, default="deep_256", dest="fusion_head",
                        choices=list(FUSION_HEADS.keys()))
    parser.add_argument("--batch-size",  type=int, default=16, dest="batch_size")
    parser.add_argument("--accum-steps", type=int, default=2,  dest="accum_steps")
    parser.add_argument("--lr-adapters", type=float, default=5e-5, dest="lr_adapters",
                        help="LR for LoRA adapter parameters and tabular branch")
    parser.add_argument("--lr-head",     type=float, default=1e-4, dest="lr_head",
                        help="LR for fusion head (can be higher — trained from scratch)")
    parser.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS, dest="max_epochs")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--text-proj-dim",  type=int, default=0, dest="text_proj_dim",
                        help="Project text CLS embedding to this dim before fusion (0=disabled)")
    parser.add_argument("--image-proj-dim", type=int, default=0, dest="image_proj_dim",
                        help="Project image pooler output to this dim before fusion (0=disabled)")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*70}")
    print(f"FusionLoRA | variant={variant} | image={args.image_size}px | rank={args.lora_rank} | head={args.fusion_head}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading parquets...")
    train_tab = pd.read_parquet(DATA_DIR / f"train{suffix}_tabular.parquet")
    val_tab   = pd.read_parquet(DATA_DIR / f"val{suffix}_tabular.parquet")
    test_tab  = pd.read_parquet(DATA_DIR / f"test{suffix}_tabular.parquet")
    train_raw = pd.read_parquet(DATA_DIR / f"train{suffix}.parquet")
    val_raw   = pd.read_parquet(DATA_DIR / f"val{suffix}.parquet")
    test_raw  = pd.read_parquet(DATA_DIR / f"test{suffix}.parquet")

    price_transformer = None
    if target_col == "price_bc":
        price_transformer = joblib.load(DATA_DIR / f"price_transformer{suffix}.joblib")

    if args.smoke_test:
        print("  [SMOKE TEST] truncating all splits to 100 rows")
        train_tab = train_tab.iloc[:100].copy(); train_raw = train_raw.iloc[:100].copy()
        val_tab   = val_tab.iloc[:100].copy();   val_raw   = val_raw.iloc[:100].copy()
        test_tab  = test_tab.iloc[:100].copy();  test_raw  = test_raw.iloc[:100].copy()

    print(f"  train={len(train_tab)}  val={len(val_tab)}  test={len(test_tab)}")

    # -------------------------------------------------------------------------
    # Tracker (writes config.json immediately)
    # -------------------------------------------------------------------------
    tracker = ExperimentTracker(
        model_type="FusionLoRA",
        modalities="tab+text+image",
        variant=variant,
        run_name=args.run_name,
        is_smoke_test=args.smoke_test,
        fusion_head=args.fusion_head,
        image_size=args.image_size,
        lora_applied_to="text+image",
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        dataloader_workers=args.workers,
        device_used=str(device),
        config={
            "lr_adapters":  args.lr_adapters,
            "lr_head":      args.lr_head,
            "accum_steps":  args.accum_steps,
            "eff_batch":    eff_batch,
            "max_epochs":   args.max_epochs,
            "early_stop_patience": EARLY_STOP_PAT,
            "max_text_len": MAX_TEXT_LEN,
            "lora_alpha":   args.lora_rank * LORA_ALPHA_MULT,
            "lora_dropout": LORA_DROPOUT,
            "clip_model":   clip_id,
            "text_model":   TEXT_MODEL_ID,
            "tabular_embed_dim": TABULAR_EMBED_DIM,
            "text_proj_dim":    args.text_proj_dim,
            "image_proj_dim":   args.image_proj_dim,
            "fusion_head_sizes": head_sizes,
            "target_column": target_col,
        },
    )
    if price_transformer is not None:
        tracker.set_box_cox_lambda(float(price_transformer.lambdas_[0]))

    # -------------------------------------------------------------------------
    # Build datasets (RAM preload happens inside __init__)
    # -------------------------------------------------------------------------
    print("\nBuilding tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)

    print("Building datasets (preloading into RAM)...")
    t_data = time.time()
    train_ds = FusionLoRADataset(train_tab, train_raw, tokenizer, image_dir, args.image_size, target_col)
    val_ds   = FusionLoRADataset(val_tab,   val_raw,   tokenizer, image_dir, args.image_size, target_col)
    test_ds  = FusionLoRADataset(test_tab,  test_raw,  tokenizer, image_dir, args.image_size, target_col)
    print(f"  Data preload complete in {(time.time()-t_data)/60:.1f} min")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # -------------------------------------------------------------------------
    # Build model + apply LoRA
    # -------------------------------------------------------------------------
    print("\nLoading encoders and applying LoRA...")
    model = FusionLoRAModel(
        image_model_id=clip_id,
        tabular_dim=len(TABULAR_COLS),
        fusion_head_sizes=head_sizes,
        text_proj_dim=args.text_proj_dim,
        image_proj_dim=args.image_proj_dim,
    ).to(device)

    trainable_params = apply_lora(model, args.lora_rank)

    # LoRA adapters must be float32 for stable gradient updates even when the
    # frozen backbone is float16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # Two-group optimizer: lower LR for LoRA adapters, higher for all fresh layers
    adapter_params = [p for p in model.text_encoder.parameters()  if p.requires_grad] + \
                     [p for p in model.image_encoder.parameters()  if p.requires_grad]
    head_params    = list(model.tabular_branch.parameters()) + \
                     list(model.fusion_head.parameters()) + \
                     (list(model.text_proj.parameters())  if model.text_proj  is not None else []) + \
                     (list(model.image_proj.parameters()) if model.image_proj is not None else [])

    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": args.lr_adapters, "weight_decay": 1e-4},
        {"params": head_params,    "lr": args.lr_head,     "weight_decay": 1e-3},
    ])
    # float32 training — no GradScaler needed

    epochs    = 1 if args.smoke_test else args.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # -------------------------------------------------------------------------
    # Training loop — early stopping on val_rmse_raw (more stable than val_loss)
    # -------------------------------------------------------------------------
    best_val_rmse  = float("inf")
    patience_count = 0
    best_state     = None
    t0             = time.time()

    print(f"\nTraining for up to {epochs} epoch(s)  |  eff_batch={eff_batch}  |  lr_adapters={args.lr_adapters}  |  lr_head={args.lr_head}")

    for epoch in range(epochs):
        tracker.start_epoch()
        _epoch_t0 = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---", flush=True)

        train_loss = run_epoch(model, train_loader, optimizer, device, args.accum_steps)
        val_loss   = run_epoch(model, val_loader,   None,      device, args.accum_steps)

        val_true_raw, val_pred_raw = predict_raw(model, val_loader, device, price_transformer, target_col)
        val_rmse_raw = float(np.sqrt(mean_squared_error(val_true_raw, val_pred_raw)))

        scheduler.step()
        current_lr = scheduler.get_last_lr()

        epoch_mins = (time.time() - _epoch_t0) / 60
        tracker.log_epoch(train_loss=train_loss, val_loss=val_loss, val_rmse_raw=val_rmse_raw)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_RMSE=${val_rmse_raw:.2f}  "
              f"epoch_time={epoch_mins:.1f}min  lr={current_lr[0]:.2e}", flush=True)

        # Rolling single checkpoint — overwrite previous to keep disk use flat
        ckpt_path = tracker.run_dir / "checkpoint_latest.pth"
        torch.save(model.state_dict(), ckpt_path)

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

    # Restore best weights for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # -------------------------------------------------------------------------
    # Final evaluation — all splits, raw dollars
    # -------------------------------------------------------------------------
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
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions={
            "train_y_true": train_true, "train_y_pred": train_pred,
            "val_y_true":   val_true,   "val_y_pred":   val_pred,
            "test_y_true":  test_true,  "test_y_pred":  test_pred,
        },
        trainable_parameters=trainable_params,
        training_time_minutes=training_time,
        best_hyperparams={
            "lora_rank":   args.lora_rank,
            "lr_adapters": args.lr_adapters,
            "lr_head":     args.lr_head,
            "batch_size":  args.batch_size,
            "accum_steps": args.accum_steps,
            "fusion_head": args.fusion_head,
        },
        peak_vram_gb=peak_vram,
    )


if __name__ == "__main__":
    main()
