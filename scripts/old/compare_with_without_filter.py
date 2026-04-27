"""
Comparative Analysis: Data with vs without $5000 Price Filter
==============================================================

Tests whether removing extreme outliers (prices > $5000) improves feature signal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_processor import AirbnbDataProcessor

def analyze_dataset(df, label):
    """Analyze feature-price correlations for a dataset."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*80}")
    print(f"Dataset size: {len(df):,} records")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Price mean: ${df['price'].mean():.2f} | std: ${df['price'].std():.2f}")
    print(f"Price variance (std²): {df['price'].var():.0f}")
    
    numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
    
    print(f"\n📊 NUMERIC FEATURE CORRELATIONS:")
    print("-" * 80)
    
    correlations = {}
    for feature in numeric_features:
        valid_idx = df[[feature, 'price']].notna().all(axis=1)
        corr, pvalue = stats.pearsonr(df.loc[valid_idx, feature], df.loc[valid_idx, 'price'])
        rank_corr, rank_pvalue = stats.spearmanr(df.loc[valid_idx, feature], df.loc[valid_idx, 'price'])
        r_squared = corr ** 2
        
        correlations[feature] = corr
        print(f"  {feature:20s}: Pearson r={corr:7.4f} (R²={r_squared:.4f}) | Spearman ρ={rank_corr:7.4f}")
    
    # Categorical features
    print(f"\n🏷️  CATEGORICAL FEATURE ASSOCIATIONS:")
    print("-" * 80)
    
    for feature in ['room_type', 'neighbourhood_cleansed']:
        price_quartiles = pd.qcut(df['price'], q=4, duplicates='drop')
        contingency = pd.crosstab(df[feature], price_quartiles)
        chi2 = stats.chi2_contingency(contingency)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        print(f"  {feature:20s}: Cramér's V={cramers_v:.4f}")
    
    # Average correlation strength
    avg_corr = np.mean(np.abs(list(correlations.values())))
    print(f"\n📈 AVERAGE ABSOLUTE CORRELATION: {avg_corr:.4f}")
    
    return correlations


# Run analysis
if __name__ == "__main__":
    processor = AirbnbDataProcessor(
        data_dir=str(Path(__file__).parent.parent),
        output_dir=str(Path(__file__).parent.parent / "data" / "diagnostic_temp")
    )
    
    # Load data without filter
    print("\n" + "="*80)
    print("LOADING DATA WITHOUT PRICE FILTER")
    print("="*80)
    df_no_filter = processor.process(max_price=None)
    correlations_no_filter = analyze_dataset(df_no_filter, "NO FILTER (All prices)")
    
    # Load data with $5000 filter
    print("\n" + "="*80)
    print("LOADING DATA WITH $5000 PRICE FILTER")
    print("="*80)
    df_with_filter = processor.process(max_price=5000)
    correlations_with_filter = analyze_dataset(df_with_filter, "WITH $5000 FILTER")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON: Impact of $5000 Price Filter")
    print("="*80)
    
    removed_count = len(df_no_filter) - len(df_with_filter)
    removed_pct = 100 * removed_count / len(df_no_filter)
    
    print(f"\nRecords removed: {removed_count:,} ({removed_pct:.1f}%)")
    print(f"Remaining: {len(df_with_filter):,} records")
    
    print(f"\nPrice statistics before/after filter:")
    print(f"  Without filter: mean=${df_no_filter['price'].mean():.2f}, std=${df_no_filter['price'].std():.2f}, max=${df_no_filter['price'].max():.2f}")
    print(f"  With filter:    mean=${df_with_filter['price'].mean():.2f}, std=${df_with_filter['price'].std():.2f}, max=${df_with_filter['price'].max():.2f}")
    
    print(f"\nCorrelation improvements (Pearson r):")
    print("-" * 80)
    for feature in correlations_no_filter.keys():
        corr_before = correlations_no_filter[feature]
        corr_after = correlations_with_filter[feature]
        delta = corr_after - corr_before
        improvement_pct = 100 * delta / abs(corr_before) if corr_before != 0 else 0
        
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {feature:20s}: {corr_before:7.4f} → {corr_after:7.4f}  {arrow} ({delta:+.4f}, {improvement_pct:+.1f}%)")
    
    avg_before = np.mean(np.abs(list(correlations_no_filter.values())))
    avg_after = np.mean(np.abs(list(correlations_with_filter.values())))
    print(f"\n  Average correlation strength: {avg_before:.4f} → {avg_after:.4f} ({100*(avg_after-avg_before)/avg_before:+.1f}%)")
    
    print("\n" + "="*80)
