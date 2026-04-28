"""
Training curve visualizations — reads history.json from each DL run folder.
Output: outputs/figures/curves_*.png
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

RUNS = Path("outputs/runs")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

def load_history(run_dir: Path):
    h = run_dir / "history.json"
    if not h.exists():
        return None
    with open(h) as f:
        return json.load(f)

def load_config(run_dir: Path):
    c = run_dir / "config.json"
    if not c.exists():
        return {}
    with open(c) as f:
        return json.load(f)

# Collect all DL runs (those with a history.json)
runs = []
for d in sorted(RUNS.iterdir()):
    if d.name.startswith("SMOKE"):
        continue
    h = load_history(d)
    if h is None or len(h.get("val_rmse_raw", [])) == 0:
        continue
    cfg = load_config(d)
    runs.append({"dir": d, "history": h, "config": cfg,
                 "model_type": cfg.get("model_type", d.name),
                 "variant": cfg.get("dataset_variant", ""),
                 "label": f"{cfg.get('model_type', '')} ({cfg.get('dataset_variant', '')})"})

if not runs:
    print("No DL runs with history.json found.")
    exit(0)

# ── 1. Val RMSE curves for all DL models (one subplot each) ─────────────────

# Group by model_type for cleaner plots
from collections import defaultdict
by_type = defaultdict(list)
for r in runs:
    by_type[r["model_type"]].append(r)

for mtype, group in by_type.items():
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in group:
        epochs = range(1, len(r["history"]["val_rmse_raw"]) + 1)
        ax.plot(epochs, r["history"]["val_rmse_raw"],
                label=r["variant"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE (CAD $)")
    ax.set_title(f"{mtype} — Validation RMSE over Training")
    ax.legend(fontsize=8)
    plt.tight_layout()
    safe = mtype.lower().replace("+", "_")
    plt.savefig(OUT / f"curves_val_rmse_{safe}.png", dpi=150)
    plt.close()
    print(f"Saved curves_val_rmse_{safe}.png")

# ── 2. Train vs val loss curves for LoRA models ─────────────────────────────

lora_runs = [r for r in runs if "LoRA" in r["model_type"]]
if lora_runs:
    fig, axes = plt.subplots(1, len(lora_runs), figsize=(5 * len(lora_runs), 4), squeeze=False)
    for ax, r in zip(axes[0], lora_runs):
        epochs = range(1, len(r["history"]["train_loss"]) + 1)
        ax.plot(epochs, r["history"]["train_loss"], label="Train loss", linewidth=1.5)
        ax.plot(epochs, r["history"]["val_loss"],   label="Val loss",   linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(r["label"], fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("LoRA Models — Train vs Val Loss", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "curves_lora_train_val_loss.png", dpi=150)
    plt.close()
    print("Saved curves_lora_train_val_loss.png")

# ── 3. LoRA val RMSE: compare image_lora 224 vs 336 if both exist ───────────

img_lora = [r for r in runs if r["model_type"] == "ImageLoRA"]
if len(img_lora) >= 2:
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in img_lora:
        img_size = r["config"].get("image_size", "?")
        label = f"ImageLoRA {img_size}px ({r['variant']})"
        epochs = range(1, len(r["history"]["val_rmse_raw"]) + 1)
        ax.plot(epochs, r["history"]["val_rmse_raw"], label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE (CAD $)")
    ax.set_title("ImageLoRA — 224px vs 336px")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "curves_image_lora_resolution.png", dpi=150)
    plt.close()
    print("Saved curves_image_lora_resolution.png")

# ── 4. Combined LoRA vs best frozen MLP val RMSE ────────────────────────────

# Best frozen MLP per modality is FusionMLP normal_bc
# Plot alongside FusionLoRA if history exists
fusion_lora = [r for r in runs if r["model_type"] == "FusionLoRA"]
fusion_mlp  = [r for r in runs if r["model_type"] == "FusionMLP"
               and r["variant"] == "normal_bc"]

if fusion_lora and fusion_mlp:
    fig, ax = plt.subplots(figsize=(9, 4))
    best_mlp = min(fusion_mlp, key=lambda r: min(r["history"]["val_rmse_raw"]))
    # FusionMLP has constant val_rmse across epochs (frozen encoder + shallow head converges fast)
    # plot as horizontal dashed line for clarity
    mlp_best_val = min(best_mlp["history"]["val_rmse_raw"])
    ax.axhline(mlp_best_val, color="gray", linestyle="--", linewidth=1.2,
               label=f"FusionMLP best val RMSE ${mlp_best_val:.0f}")
    for r in fusion_lora:
        epochs = range(1, len(r["history"]["val_rmse_raw"]) + 1)
        ax.plot(epochs, r["history"]["val_rmse_raw"],
                label=f"FusionLoRA ({r['variant']})", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val RMSE (CAD $)")
    ax.set_title("FusionLoRA vs FusionMLP (frozen) — Val RMSE")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "curves_lora_vs_frozen_mlp.png", dpi=150)
    plt.close()
    print("Saved curves_lora_vs_frozen_mlp.png")

print("\nDone. All figures in outputs/figures/")
