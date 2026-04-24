"""
Feature-Price Diagnostic Script
================================
Simple scatter plots: X-axis = feature, Y-axis = price.
No fancy transformations or overlays. Just raw data points.

Usage:
    python3 scripts/feature_price_diagnostic.py
    
Outputs:
    outputs/diagnostics/ — PNG images for each feature vs price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Vertical orientation: narrow width, tall height to show price spread
DPI = 100
WIDTH = 6   # inches (narrow, limited unique X values)
HEIGHT = 30  # inches (very tall to show price distribution)
FIG_SIZE = (WIDTH, HEIGHT)

def load_data():
    """Load train parquet (raw, not scaled)."""
    train_path = Path(__file__).parent.parent / "data" / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    df = pd.read_parquet(train_path)
    
    # APPLY PRICE FILTER: Remove prices > $5000
    df_filtered = df[df['price'] <= 5000].copy()
    print(f"Loaded {len(df):,} records, filtered to {len(df_filtered):,} (removed {len(df)-len(df_filtered)} outliers > $5000)")
    
    return df_filtered


def _safe_corr(x: pd.Series, y: pd.Series):
    """Return (pearson_r, spearman_rho) with NaNs for degenerate inputs."""
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return np.nan, np.nan
    x_vals = xy.iloc[:, 0].to_numpy()
    y_vals = xy.iloc[:, 1].to_numpy()
    if np.nanstd(x_vals) == 0 or np.nanstd(y_vals) == 0:
        return np.nan, np.nan

    pearson_r = stats.pearsonr(x_vals, y_vals)[0]
    spearman_rho = stats.spearmanr(x_vals, y_vals)[0]
    return float(pearson_r), float(spearman_rho)


def create_feature_diagnostics(df: pd.DataFrame):
    """
    Simple scatter plots: one plot per feature.
    X-axis = feature, Y-axis = price.
    Every single point plotted with transparency.
    """
    output_dir = Path(__file__).parent.parent / "outputs" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target = 'price'
    
    numeric_features = [
        'accommodates',
        'bathrooms',
        'bedrooms',
        'beds',
        'minimum_nights',
        'season_ordinal',
        'host_total_listings_count',
        'latitude',
        'longitude',
        'availability_365',
        'number_of_reviews',
    ]
    
    categorical_features = [
        'room_type',
        'neighbourhood_cleansed',
        'property_type',
        'instant_bookable',
    ]
    
    print("=" * 80)
    print("FEATURE-PRICE SCATTER PLOTS")
    print("=" * 80)
    print(f"Dataset: {len(df)} records")
    print(f"Target: {target} (mean=${df[target].mean():.2f}, std=${df[target].std():.2f})")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # ===== NUMERIC FEATURES =====
    print("\n📊 NUMERIC FEATURES (Scatter Plots):")
    print("-" * 80)
    
    for feature in numeric_features:
        print(f"\n  📍 {feature.upper()}")
        
        # Compute correlation (robust to constant/degenerate inputs)
        corr, rank_corr = _safe_corr(df[feature], df[target])
        
        print(f"     Range: {df[feature].min():.2f} to {df[feature].max():.2f}")
        if np.isnan(corr) or np.isnan(rank_corr):
            print("     Pearson r: N/A | Spearman ρ: N/A (degenerate / insufficient variation)")
        else:
            print(f"     Pearson r: {corr:.4f} | Spearman ρ: {rank_corr:.4f}")
        
        # Create simple scatter plot
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        
        # Plot all points with transparency
        ax.scatter(df[feature], df[target], alpha=0.2, s=20, edgecolors='none', color='blue')
        
        # Add trend line (linear fit) when feasible
        valid_idx = df[[feature, target]].notna().all(axis=1)
        if valid_idx.sum() >= 3 and df.loc[valid_idx, feature].nunique() >= 2:
            z = np.polyfit(df.loc[valid_idx, feature], df.loc[valid_idx, target], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
            label = f"Linear fit" if np.isnan(corr) else f"Linear fit (r={corr:.3f})"
            ax.plot(x_trend, p(x_trend), "r-", linewidth=3, label=label)
        
        ax.set_xlabel(feature, fontsize=24, fontweight='bold')
        ax.set_ylabel(target, fontsize=24, fontweight='bold')
        if np.isnan(corr) or np.isnan(rank_corr):
            title = f"{feature.upper()} vs {target.upper()}\nPearson r=N/A | Spearman ρ=N/A"
        else:
            title = f"{feature.upper()} vs {target.upper()}\nPearson r={corr:.4f} | Spearman ρ={rank_corr:.4f}"
        ax.set_title(title, fontsize=28, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18)
        ax.tick_params(labelsize=16)
        
        plt.tight_layout()
        
        output_path = output_dir / f"01_numeric_{feature}_scatter.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"     ✅ Saved: {output_path.name}")
        plt.close()
    
    # ===== CATEGORICAL FEATURES =====
    print("\n🏷️  CATEGORICAL FEATURES (Jittered Scatter):")
    print("-" * 80)
    
    for feature in categorical_features:
        print(f"\n  📍 {feature.upper()}")
        
        n_unique = df[feature].nunique()
        print(f"     Unique values: {n_unique}")
        
        # Compute Cramér's V
        price_quartiles = pd.qcut(df[target], q=4, duplicates='drop')
        contingency = pd.crosstab(df[feature], price_quartiles)
        chi2 = stats.chi2_contingency(contingency)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        print(f"     Cramér's V: {cramers_v:.4f}")
        
        # Create jittered scatter plot
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        
        # Convert categories to numeric codes for x-axis
        categories = sorted(df[feature].unique())
        category_to_code = {cat: i for i, cat in enumerate(categories)}
        x_numeric = df[feature].map(category_to_code)
        
        # Add small random jitter to x for visibility
        x_jitter = x_numeric + np.random.normal(0, 0.05, len(df))
        
        ax.scatter(x_jitter, df[target], alpha=0.15, s=15, edgecolors='none', color='green')
        
        # Add mean price line for each category
        category_means = []
        for cat in categories:
            mean_price = df[df[feature] == cat][target].mean()
            category_means.append(mean_price)
        
        ax.plot(range(len(categories)), category_means, 'r-o', linewidth=3, markersize=10, 
                label='Category Mean', zorder=5)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([str(c)[:15] for c in categories], rotation=45, ha='right', fontsize=14)
        ax.set_ylabel(target, fontsize=24, fontweight='bold')
        ax.set_xlabel(f"{feature} (Jittered)", fontsize=24, fontweight='bold')
        ax.set_title(f"{feature.upper()} vs {target.upper()}\nCramér's V={cramers_v:.4f} ({n_unique} categories)", 
                     fontsize=28, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=18)
        ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        
        output_path = output_dir / f"02_categorical_{feature}_scatter.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"     ✅ Saved: {output_path.name}")
        plt.close()
    
    print("\n" + "=" * 80)
    print(f"✅ All scatter plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    df = load_data()
    create_feature_diagnostics(df)
