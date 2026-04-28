"""
Error analysis visualizations — reads predictions.npz from run folders.
Output: outputs/figures/error_*.png

Picks the best run per model family (lowest test RMSE, normal_bc).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG  = Path("outputs/master_runs_log.csv")
RUNS = Path("outputs/runs")
OUT  = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG)
df = df[~df["run_id"].str.startswith("SMOKE")]
df = df[df["test_rmse_raw"] < 5000]
df = df[df["dataset_variant"] == "normal_bc"]

# Best run per model_type
best = (df.sort_values("test_rmse_raw")
          .groupby("model_type")
          .first()
          .reset_index())

def load_preds(artifact_folder: str):
    # artifact_folder may be a relative path like outputs/runs/...
    p = Path(artifact_folder)
    if not p.is_absolute():
        p = Path.cwd() / p
    npz = p / "predictions.npz"
    if not npz.exists():
        # Also try the NVMe path
        name = p.name
        npz = Path("/mnt/nvme_data/linux_sys/aribnb_outputs/runs") / name / "predictions.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    return data["test_y_true"], data["test_y_pred"]

# ── 1. Predicted vs Actual scatter — best 4 models ──────────────────────────

# Pick top 4 by test_rmse_raw (ascending)
top4 = best.nsmallest(4, "test_rmse_raw")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

plotted = 0
for _, row in top4.iterrows():
    preds = load_preds(row["artifact_folder"])
    if preds is None:
        continue
    y_true, y_pred = preds
    ax = axes[plotted]
    lim = (0, max(y_true.max(), y_pred.max()) * 1.05)
    ax.scatter(y_true, y_pred, alpha=0.15, s=5, color="#4C72B0")
    ax.plot(lim, lim, "r--", linewidth=1, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Price (CAD $)")
    ax.set_ylabel("Predicted Price (CAD $)")
    ax.set_title(f"{row['model_type']}\nRMSE=${row['test_rmse_raw']:.0f} R²={row['test_r2']:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7)
    plotted += 1
    if plotted == 4:
        break

for i in range(plotted, 4):
    axes[i].set_visible(False)

plt.suptitle("Predicted vs Actual — Top 4 Models (normal_bc)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "error_pred_vs_actual_top4.png", dpi=150)
plt.close()
print("Saved error_pred_vs_actual_top4.png")

# ── 2. Residual distribution — best model ───────────────────────────────────

best_row = best.nsmallest(1, "test_rmse_raw").iloc[0]
preds = load_preds(best_row["artifact_folder"])
if preds is not None:
    y_true, y_pred = preds
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=80, color="#4C72B0", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
    ax.axvline(residuals.mean(), color="orange", linewidth=1.2, linestyle="--",
               label=f"Mean error ${residuals.mean():.0f}")
    ax.set_xlabel("Residual (Predicted − Actual, CAD $)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {best_row['model_type']} (normal_bc)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "error_residual_distribution.png", dpi=150)
    plt.close()
    print("Saved error_residual_distribution.png")

# ── 3. Error by price bucket — best model ───────────────────────────────────

if preds is not None:
    y_true, y_pred = preds
    bins = np.percentile(y_true, [0, 20, 40, 60, 80, 100])
    labels = ["Budget\n(0–20%)", "Low-Mid\n(20–40%)", "Mid\n(40–60%)",
              "High\n(60–80%)", "Luxury\n(80–100%)"]
    bucket_mae = []
    bucket_rmse = []
    for i in range(len(labels)):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() == 0:
            bucket_mae.append(0); bucket_rmse.append(0)
            continue
        bucket_mae.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        bucket_rmse.append(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, bucket_mae,  w, label="MAE",  color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, bucket_rmse, w, label="RMSE", color="#DD8452", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Error (CAD $)")
    ax.set_title(f"Error by Price Quintile — {best_row['model_type']} (normal_bc)")
    ax.legend()
    # Add price range annotations
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        ax.text(x[i], -12, f"${lo:.0f}–${hi:.0f}", ha="center", fontsize=6.5,
                color="gray", clip_on=False)
    plt.tight_layout()
    plt.savefig(OUT / "error_by_price_bucket.png", dpi=150)
    plt.close()
    print("Saved error_by_price_bucket.png")

# ── 4. Price distribution — actual vs predicted (best model) ────────────────

if preds is not None:
    y_true, y_pred = preds
    cap = np.percentile(y_true, 99)  # cap at 99th percentile for readability
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y_true[y_true <= cap],  bins=80, alpha=0.6, label="Actual",    color="#4C72B0", density=True)
    ax.hist(y_pred[y_pred <= cap],  bins=80, alpha=0.6, label="Predicted", color="#DD8452", density=True)
    ax.set_xlabel("Price (CAD $)")
    ax.set_ylabel("Density")
    ax.set_title(f"Price Distribution: Actual vs Predicted — {best_row['model_type']}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "error_price_distribution_actual_vs_pred.png", dpi=150)
    plt.close()
    print("Saved error_price_distribution_actual_vs_pred.png")

print("\nDone. All figures in outputs/figures/")
