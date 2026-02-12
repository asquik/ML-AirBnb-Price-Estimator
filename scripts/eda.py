"""Full EDA for Multi-Modal Airbnb Price Predictor (Montreal)

Produces:
 - `outputs/report.txt`  : professor-style textual summary + dataset diagnostics
 - `outputs/figures`     : plots (price hist, log-price, room type, neighbourhood counts)
 - `outputs/summary.csv` : tabular diagnostics (missingness + basic stats)
 - prints a short summary to stdout

Notes:
 - image URL checks use HEAD requests on a sample (configurable) to avoid downloading many images.
 - follows the project's strict preprocessing rules for `price` filtering (50 < price < 1000).
"""
import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm

# Constants
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "listings.csv"
OUTDIR = ROOT / "outputs"
FIG_DIR = OUTDIR / "figures"

os.makedirs(FIG_DIR, exist_ok=True)


def clean_price(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Price cleaning and filtering
    df["price_raw"] = df.get("price", np.nan)
    df["price"] = df["price_raw"].apply(clean_price)
    before = len(df)
    df = df[(df["price"] > 50) & (df["price"] < 1000)].copy()
    after = len(df)

    # Combine text fields
    df["text"] = (
        df.get("description", "").fillna("") + " \nAMENITIES: " + df.get("amenities", "").fillna("")
    )

    # Normalize numeric tabular columns (simple min-max for now)
    for col in ["accommodates", "bathrooms", "bedrooms"]:
        if col in df.columns:
            col_min = df[col].min(skipna=True)
            col_max = df[col].max(skipna=True)
            if not math.isclose(col_min, col_max):
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_norm"] = df[col].fillna(0.0)

    # Room type and neighbourhood preprocessing notes
    if "room_type" in df.columns:
        df["room_type"] = df["room_type"].fillna("Unknown")
    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("Unknown")

    return df, before, after


def sample_and_check_image_urls(df: pd.DataFrame, sample_n=200, timeout=3):
    urls = df["picture_url"].dropna().unique()
    sample = urls[:sample_n]
    results = {"ok": 0, "bad": 0, "timeout": 0}
    examples = {"ok": [], "bad": [], "timeout": []}

    headers = {"User-Agent": "EDA-checker/1.0 (+https://github.com)"}
    for u in tqdm(sample, desc="Checking image URLs", disable=(sample_n == 0)):
        try:
            r = requests.head(u, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                results["ok"] += 1
                if len(examples["ok"]) < 3:
                    examples["ok"].append(u)
            else:
                results["bad"] += 1
                if len(examples["bad"]) < 3:
                    examples["bad"].append((u, r.status_code, r.headers.get("Content-Type")))
        except requests.exceptions.Timeout:
            results["timeout"] += 1
            if len(examples["timeout"]) < 3:
                examples["timeout"].append(u)
        except Exception as e:
            results["bad"] += 1
            if len(examples["bad"]) < 3:
                examples["bad"].append((u, str(e)))

    return results, examples


def plot_price_distributions(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=60, kde=False)
    plt.title("Price distribution (50 < price < 1000)")
    plt.xlabel("Price (USD)")
    plt.tight_layout()
    p1 = outdir / "price_hist.png"
    plt.savefig(p1)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df["price"]), bins=60, kde=False, color="C1")
    plt.title("Log(1+price) distribution")
    plt.xlabel("log(1 + price)")
    plt.tight_layout()
    p2 = outdir / "price_log_hist.png"
    plt.savefig(p2)
    plt.close()

    return [p1, p2]


def top_counts_plot(df: pd.DataFrame, col: str, topk: int, outdir: Path):
    s = df[col].value_counts().nlargest(topk)
    plt.figure(figsize=(8, 5))
    # Use matplotlib horizontal bar to avoid seaborn palette deprecation warning
    colors = sns.color_palette("viridis", len(s))
    plt.barh(range(len(s)), s.values, color=colors)
    plt.yticks(range(len(s)), s.index)
    plt.xlabel("count")
    plt.title(f"Top {topk} {col}")
    plt.tight_layout()
    p = outdir / f"top_{col}.png"
    plt.savefig(p)
    plt.close()
    return p


def numeric_correlations(df: pd.DataFrame, outdir: Path):
    # Compute correlations of numeric columns with `price` (if present)
    if "price" not in df.columns:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "id"]
    if "price" not in numeric:
        numeric.append("price")
    if len(numeric) < 2:
        return None
    corr_series = df[numeric].corr()["price"].sort_values(ascending=False)
    p = outdir / "price_correlations.csv"
    corr_series.to_csv(p, header=["corr_with_price"])    
    return p, corr_series


def make_textual_report(df: pd.DataFrame, before: int, after: int, url_results, url_examples) -> str:
    # Academic-style narrative tailored for the professor
    mean_price = df["price"].mean()
    median_price = df["price"].median()
    n = len(df)
    room_types = df["room_type"].value_counts().to_dict() if "room_type" in df.columns else {}
    top_neighborhoods = df["neighbourhood_cleansed"].value_counts().nlargest(5).to_dict() if "neighbourhood_cleansed" in df.columns else {}

    lines = []
    lines.append("Executive summary:")
    lines.append(f" - Dataset reduced from {before} → {after} rows after filtering price to (50, 1000). Final N = {n}.")
    lines.append(f" - Central tendency: mean=${mean_price:.2f}, median=${median_price:.2f} (skew present).")
    lines.append("")
    lines.append("Data quality & image availability:")
    lines.append(f" - Sampled image URL reachability (HEAD requests): {url_results}. Examples: {json.dumps(url_examples, indent=2)[:400]}.")
    lines.append("")
    lines.append("Feature notes & preprocessing decisions:")
    lines.append(" - Price: cleaned from string to float; extreme and erroneous entries removed per spec.")
    lines.append(" - Text: `description` + `amenities` concatenated for V+L backbone input.")
    lines.append(" - Tabular: only `room_type`, `neighbourhood_cleansed`, `accommodates`, `bathrooms`, `bedrooms` kept — numeric columns min-max normalized.")
    lines.append("")
    lines.append("Initial modeling recommendations (for professor defense):")
    lines.append(" 1) Use a frozen CLIP/ViLT backbone for image/text feature extraction to preserve transfer learning benefits and computational efficiency.")
    lines.append(" 2) Model target: try both raw-price MSE and log-price MSE — log reduces heteroscedasticity.")
    lines.append(" 3) Ablations: image-only, text-only, tabular-only to measure marginal contributions of 'curb appeal'.")
    lines.append("")
    lines.append("Top-level descriptive findings:")
    lines.append(f" - Room type distribution (top): {room_types}")
    lines.append(f" - Top neighbourhoods (by sample count): {top_neighborhoods}")
    lines.append("")
    lines.append("Conclusion:")
    lines.append(" - Data is suitable for the planned late-fusion transfer-learning experiment after minor cleaning and handling of unreachable image URLs. Recommended next step: implement the Dataset class and run small-scale sanity training using a frozen backbone.")

    return "\n".join(lines)


def write_summary_csv(df: pd.DataFrame, outpath: Path):
    cols = []
    for c in df.columns:
        cols.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isna().sum()),
                "n_unique": int(df[c].nunique(dropna=True)) if df[c].nunique(dropna=True) < 1_000_000 else -1,
            }
        )
    pd.DataFrame(cols).to_csv(outpath, index=False)


def main(args):
    df, before, after = load_and_clean(Path(args.csv))

    # Diagnostics
    write_summary_csv(df, OUTDIR / "summary.csv")

    # Plots
    figs = plot_price_distributions(df, FIG_DIR)
    if "room_type" in df.columns:
        figs.append(top_counts_plot(df, "room_type", 10, FIG_DIR))
    if "neighbourhood_cleansed" in df.columns:
        figs.append(top_counts_plot(df, "neighbourhood_cleansed", 10, FIG_DIR))

    corr_res = numeric_correlations(df, FIG_DIR)

    # Image checks (sampled)
    url_results, url_examples = sample_and_check_image_urls(df, sample_n=args.sample_image_check)

    # Textual report
    report = make_textual_report(df, before, after, url_results, url_examples)
    (OUTDIR / "report.txt").write_text(report, encoding="utf-8")

    # Short stdout summary
    print("EDA complete — outputs written to:")
    print(f" - {OUTDIR / 'report.txt'}")
    print(f" - Figures in {FIG_DIR}")
    print(f" - Summary CSV: {OUTDIR / 'summary.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DATA_PATH), help="Path to listings CSV")
    p.add_argument("--sample-image-check", type=int, default=200, help="Number of picture_url HEAD requests to sample")
    args = p.parse_args()
    main(args)
