"""
Model comparison visualizations — reads master_runs_log.csv.
Output: outputs/figures/comparison_*.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG = Path("outputs/master_runs_log.csv")
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG)

# Drop smoke tests and obviously broken runs (cleaned_bc TabularMLP explosion)
df = df[~df["run_id"].str.startswith("SMOKE")]
df = df[df["test_rmse_raw"] < 5000]

# Keep only the best run per (model_type, dataset_variant, fusion_head, lora_rank)
# "best" = lowest test_rmse_raw
best = (
    df.sort_values("test_rmse_raw")
    .groupby(["model_type", "dataset_variant"], dropna=False)
    .first()
    .reset_index()
)

# ── 1. Bar chart: test RMSE by model, coloured by modality ──────────────────

MODALITY_COLORS = {
    "tabular":          "#4C72B0",
    "tab+text":         "#DD8452",
    "tab+image":        "#55A868",
    "tab+text+image":   "#C44E52",
}

# Focus on normal_bc (cleanest apples-to-apples)
sub = best[best["dataset_variant"] == "normal_bc"].sort_values("test_rmse_raw")

fig, ax = plt.subplots(figsize=(12, 5))
colors = [MODALITY_COLORS.get(m, "#888") for m in sub["modalities"]]
bars = ax.bar(sub["model_type"], sub["test_rmse_raw"], color=colors, edgecolor="white")
ax.set_ylabel("Test RMSE (CAD $)")
ax.set_title("Test RMSE by Model — normal_bc dataset")
ax.set_xlabel("")
plt.xticks(rotation=35, ha="right")
for bar, val in zip(bars, sub["test_rmse_raw"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"${val:.0f}", ha="center", va="bottom", fontsize=8)
patches = [mpatches.Patch(color=c, label=l) for l, c in MODALITY_COLORS.items()]
ax.legend(handles=patches, title="Modality", loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "comparison_rmse_normal_bc.png", dpi=150)
plt.close()
print("Saved comparison_rmse_normal_bc.png")

# ── 2. Bar chart: test R² by model ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
colors = [MODALITY_COLORS.get(m, "#888") for m in sub["modalities"]]
bars = ax.bar(sub["model_type"], sub["test_r2"], color=colors, edgecolor="white")
ax.set_ylabel("Test R²")
ax.set_title("Test R² by Model — normal_bc dataset")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(rotation=35, ha="right")
for bar, val in zip(bars, sub["test_r2"]):
    ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
patches = [mpatches.Patch(color=c, label=l) for l, c in MODALITY_COLORS.items()]
ax.legend(handles=patches, title="Modality", loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "comparison_r2_normal_bc.png", dpi=150)
plt.close()
print("Saved comparison_r2_normal_bc.png")

# ── 3. Dataset variant ablation (RandomForest as representative) ─────────────

rf = df[df["model_type"] == "RandomForest"].sort_values("test_rmse_raw")
if len(rf) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rf["dataset_variant"], rf["test_rmse_raw"], color="#4C72B0", edgecolor="white")
    ax.set_ylabel("Test RMSE (CAD $)")
    ax.set_title("RandomForest — Dataset Variant Ablation")
    for i, (_, row) in enumerate(rf.iterrows()):
        ax.text(i, row["test_rmse_raw"] + 0.5, f"${row['test_rmse_raw']:.0f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "ablation_dataset_variant_rf.png", dpi=150)
    plt.close()
    print("Saved ablation_dataset_variant_rf.png")

# ── 4. Modality contribution: Tabular → +Text → +Image → +All ───────────────

modality_order = [
    ("TabularMLP",  "tabular"),
    ("TextMLP",     "tab+text"),
    ("ImageMLP",    "tab+image"),
    ("FusionMLP",   "tab+text+image"),
]
rows = []
for mtype, mod in modality_order:
    match = best[(best["model_type"] == mtype) & (best["dataset_variant"] == "normal_bc")]
    if len(match):
        rows.append({"label": f"{mtype}\n({mod})",
                     "rmse": match.iloc[0]["test_rmse_raw"],
                     "modality": mod})
if rows:
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MODALITY_COLORS[r["modality"]] for _, r in rdf.iterrows()]
    ax.bar(rdf["label"], rdf["rmse"], color=colors, edgecolor="white")
    ax.set_ylabel("Test RMSE (CAD $)")
    ax.set_title("Modality Contribution (MLP family, normal_bc)")
    for i, row in rdf.iterrows():
        ax.text(i, row["rmse"] + 0.5, f"${row['rmse']:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "ablation_modality_contribution.png", dpi=150)
    plt.close()
    print("Saved ablation_modality_contribution.png")

# ── 5. Fusion head depth ablation (shallow_64 vs deep_256) ──────────────────

fh = df[df["dataset_variant"] == "normal_bc"].dropna(subset=["fusion_head"])
fh = fh[fh["model_type"].isin(["TabularMLP", "TextMLP", "ImageMLP", "FusionMLP"])]
if len(fh) > 0:
    pivot = fh.groupby(["model_type", "fusion_head"])["test_rmse_raw"].min().unstack("fusion_head")
    pivot = pivot.dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_ylabel("Test RMSE (CAD $)")
    ax.set_title("Fusion Head Depth Ablation — normal_bc")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "ablation_fusion_head_depth.png", dpi=150)
    plt.close()
    print("Saved ablation_fusion_head_depth.png")

# ── 6. Params vs RMSE scatter ───────────────────────────────────────────────

pv = best[best["dataset_variant"] == "normal_bc"].dropna(subset=["trainable_parameters"])
pv = pv[pv["trainable_parameters"] > 0]
if len(pv) > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(pv["trainable_parameters"], pv["test_rmse_raw"],
                    c=[list(MODALITY_COLORS.keys()).index(m) if m in MODALITY_COLORS else 0
                       for m in pv["modalities"]],
                    cmap="tab10", s=80, edgecolors="black", linewidths=0.5)
    for _, row in pv.iterrows():
        ax.annotate(row["model_type"], (row["trainable_parameters"], row["test_rmse_raw"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)")
    ax.set_ylabel("Test RMSE (CAD $)")
    ax.set_title("Parameter Efficiency — normal_bc")
    plt.tight_layout()
    plt.savefig(OUT / "scatter_params_vs_rmse.png", dpi=150)
    plt.close()
    print("Saved scatter_params_vs_rmse.png")

print("\nDone. All figures in outputs/figures/")
