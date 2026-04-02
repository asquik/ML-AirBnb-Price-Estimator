import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np

# Get absolute path to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from scripts/
DATA_DIR = PROJECT_ROOT
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ============================================================================
# REUSABLE FUNCTIONS
# ============================================================================

def load_data():
    """Load all three snapshots and return as dict."""
    march = pd.read_csv(os.path.join(DATA_DIR, 'listings-03-25.csv'))
    june = pd.read_csv(os.path.join(DATA_DIR, 'listings-06-25.csv'))
    sept = pd.read_csv(os.path.join(DATA_DIR, 'listings-09-25.csv'))
    return {'march': march, 'june': june, 'sept': sept}


def filter_room_types(data_dict):
    """Keep only 'Entire home/apt' and 'Private room'. Drop hotel and shared rooms."""
    valid_types = ['Entire home/apt', 'Private room']
    filtered = {}
    
    for month, df in data_dict.items():
        original_count = len(df)
        filtered[month] = df[df['room_type'].isin(valid_types)].copy()
        filtered_count = len(filtered[month])
        dropped = original_count - filtered_count
        pct_dropped = 100 * dropped / original_count
        print(f"{month.capitalize()}: {filtered_count} / {original_count} (dropped {dropped} hotel/shared rooms, {pct_dropped:.2f}%)")
    
    return filtered


def clean_price(df):
    """Convert price column from string to float."""
    df = df.copy()
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    return df


def extract_and_merge(data_dict):
    """
    Extract common listings and merge prices across months.
    data_dict: {'march': df, 'june': df, 'sept': df}
    Returns: merged DataFrame with prices from all three months for common listings.
    """
    march = data_dict['march']
    june = data_dict['june']
    sept = data_dict['sept']
    
    # Find overlap between months
    march_ids = set(march['id'])
    june_ids = set(june['id'])
    sept_ids = set(sept['id'])
    
    common_ids = march_ids & june_ids & sept_ids
    
    # Extract price for these listings
    march_common = march[march['id'].isin(common_ids)][['id', 'price']].copy()
    june_common = june[june['id'].isin(common_ids)][['id', 'price']].copy()
    sept_common = sept[sept['id'].isin(common_ids)][['id', 'price']].copy()
    
    # Clean price column
    march_common = clean_price(march_common)
    june_common = clean_price(june_common)
    sept_common = clean_price(sept_common)
    
    # Merge prices by id
    merged = march_common.merge(june_common, on='id', suffixes=('_march', '_june'))
    merged = merged.merge(sept_common, on='id')
    merged.rename(columns={'price': 'price_sept'}, inplace=True)
    
    return merged


def drop_nan_rows(merged):
    """Drop rows with NaN price values."""
    original_count = len(merged)
    merged_clean = merged.dropna(subset=['price_march', 'price_june', 'price_sept'])
    dropped_count = original_count - len(merged_clean)
    return merged_clean, dropped_count


def compute_volatility_metrics(merged):
    """Compute absolute and relative price changes."""
    merged = merged.copy()
    
    # Absolute changes
    merged['abs_change_march_june'] = (merged['price_june'] - merged['price_march']).abs()
    merged['abs_change_june_sept'] = (merged['price_sept'] - merged['price_june']).abs()
    merged['abs_change_march_sept'] = (merged['price_sept'] - merged['price_march']).abs()
    
    # Relative changes
    merged['rel_change_march_june'] = merged['abs_change_march_june'] / merged['price_march']
    merged['rel_change_june_sept'] = merged['abs_change_june_sept'] / merged['price_june']
    merged['rel_change_march_sept'] = merged['abs_change_march_sept'] / merged['price_march']
    
    return merged


def print_overlap_summary(data_dict):
    """Print overlap statistics for all months."""
    march = data_dict['march']
    june = data_dict['june']
    sept = data_dict['sept']
    
    march_ids = set(march['id'])
    june_ids = set(june['id'])
    sept_ids = set(sept['id'])
    
    print(f"March shape: {march.shape}")
    print(f"June shape: {june.shape}")
    print(f"September shape: {sept.shape}")
    print(f"March unique IDs: {march['id'].nunique()}")
    print(f"June unique IDs: {june['id'].nunique()}")
    print(f"September unique IDs: {sept['id'].nunique()}")
    print(f"March ∩ June: {len(march_ids & june_ids)}")
    print(f"June ∩ September: {len(june_ids & sept_ids)}")
    print(f"March ∩ September: {len(march_ids & sept_ids)}")
    print(f"March ∩ June ∩ September: {len(march_ids & june_ids & sept_ids)}")
    print(f"Total unique listings across all months: {len(march_ids | june_ids | sept_ids)}")


def print_volatility_summary(merged):
    """Print summary statistics for price changes."""
    print("\n=== Price Volatility for Listings Present in All Months ===")
    print("Mean absolute change (March→June):", merged['abs_change_march_june'].mean())
    print("Mean absolute change (June→Sept):", merged['abs_change_june_sept'].mean())
    print("Mean absolute change (March→Sept):", merged['abs_change_march_sept'].mean())
    print("Mean relative change (March→June):", merged['rel_change_march_june'].mean())
    print("Mean relative change (June→Sept):", merged['rel_change_june_sept'].mean())
    print("Mean relative change (March→Sept):", merged['rel_change_march_sept'].mean())


def plot_temporal_changes(merged, output_suffix=''):
    """Split data into ~700 listing chunks and plot temporal price changes."""
    chunk_size = 700
    num_chunks = (len(merged) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(merged))
        plt.figure(figsize=(10, 6))
        plt.plot(merged['price_march'].iloc[start:end].values, label='March', alpha=0.7)
        plt.plot(merged['price_june'].iloc[start:end].values, label='June', alpha=0.7)
        plt.plot(merged['price_sept'].iloc[start:end].values, label='September', alpha=0.7)
        plt.title(f'Price changes (listings {start}–{end}){output_suffix}')
        plt.xlabel('Listing index')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../outputs/figures/price_changes_temporal_{i+1}{output_suffix}.png')
        plt.close()


def plot_price_change_distributions(merged, output_suffix=''):
    """Plot histograms for absolute and relative price changes."""
    abs_change = merged['abs_change_march_june'].dropna()
    rel_change = merged['rel_change_march_june'].dropna()
    
    plt.figure(figsize=(8, 5))
    plt.hist(abs_change, bins=50, color='orange', alpha=0.7)
    plt.title(f'Histogram of Absolute Price Change (March→June){output_suffix}')
    plt.xlabel('Absolute Price Change ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/abs_price_change_hist{output_suffix}.png')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.hist(rel_change, bins=50, color='blue', alpha=0.7)
    plt.title(f'Histogram of Relative Price Change (March→June){output_suffix}')
    plt.xlabel('Relative Price Change')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'../outputs/figures/rel_price_change_hist{output_suffix}.png')
    plt.close()


def print_percentiles(merged, output_suffix=''):
    """Print percentile statistics for price changes."""
    abs_change = merged['abs_change_march_june'].dropna()
    rel_change = merged['rel_change_march_june'].dropna()
    
    percentiles = [5, 25, 50, 75, 95]
    print(f"\n=== Absolute Price Change Percentiles (March→June){output_suffix} ===")
    for p in percentiles:
        print(f"{p}%: {np.percentile(abs_change, p):.2f}")
    
    print(f"\n=== Relative Price Change Percentiles (March→June){output_suffix} ===")
    for p in percentiles:
        print(f"{p}%: {np.percentile(rel_change, p):.2f}")


def export_outliers(merged, output_path, output_suffix=''):
    """Export listings with >100% relative price change to txt file."""
    outlier_mask = merged['rel_change_march_june'] > 1.0
    outliers = merged[outlier_mask][['id', 'price_march', 'price_june', 'price_sept', 'rel_change_march_june']]
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUTS_DIR, output_path)
    
    with open(output_file, 'w') as f:
        f.write('id\tprice_march\tprice_june\tprice_sept\trel_change_march_june\n')
        for row in outliers.itertuples(index=False):
            f.write(f'{row.id}\t{row.price_march:.2f}\t{row.price_june:.2f}\t{row.price_sept:.2f}\t{row.rel_change_march_june:.2f}\n')
    
    print(f"Outlier listings written to {output_file} ({len(outliers)} outliers)")


def filter_short_term_rentals(data_dict, min_nights_threshold=31):
    """Filter out long-term rentals (minimum_nights >= threshold)."""
    filtered = {}
    for month, df in data_dict.items():
        original_count = len(df)
        filtered[month] = df[df['minimum_nights'] < min_nights_threshold].copy()
        filtered_count = len(filtered[month])
        dropped = original_count - filtered_count
        print(f"{month.capitalize()}: {filtered_count} / {original_count} (dropped {dropped} long-term rentals)")
    return filtered


def analyze_nan_per_snapshot(data_dict):
    """Report NaN counts for key columns in each month's snapshot."""
    key_columns = ['price', 'minimum_nights', 'accommodates', 'bedrooms', 'bathrooms', 'room_type', 'description']
    
    print(f"\n=== NaN DATA QUALITY (by month) ===\n")
    total_records = 0
    
    for month, df in data_dict.items():
        print(f"{month.upper()}: {len(df)} records")
        print("-" * 60)
        for col in key_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                pct = 100 * nan_count / len(df)
                print(f"  {col:20s}: {nan_count:5d} NaN ({pct:5.2f}%)")
        print()
        total_records += len(df)
    
    print(f"TOTAL RECORDS ACROSS ALL MONTHS: {total_records}")
    return total_records


def analyze_listing_overlap_structure(data_dict):
    """Analyze how many listings appear in 1, 2, or 3 months."""
    march_ids = set(data_dict['march']['id'])
    june_ids = set(data_dict['june']['id'])
    sept_ids = set(data_dict['sept']['id'])
    
    # Calculate all 7 mutually exclusive groups
    all_three = march_ids & june_ids & sept_ids
    march_june_not_sept = (march_ids & june_ids) - sept_ids
    march_sept_not_june = (march_ids & sept_ids) - june_ids
    june_sept_not_march = (june_ids & sept_ids) - march_ids
    march_only = march_ids - june_ids - sept_ids
    june_only = june_ids - march_ids - sept_ids
    sept_only = sept_ids - march_ids - june_ids
    
    # Get record counts (not just unique IDs)
    march_records = len(data_dict['march'])
    june_records = len(data_dict['june'])
    sept_records = len(data_dict['sept'])
    total_records = march_records + june_records + sept_records
    
    print("\n=== LISTING OVERLAP STRUCTURE ===\n")
    print("Unique listings in each group:\n")
    print(f"All 3 months (March ∩ June ∩ Sept):     {len(all_three):6d} listings")
    print(f"March & June only (not Sept):           {len(march_june_not_sept):6d} listings")
    print(f"March & Sept only (not June):           {len(march_sept_not_june):6d} listings")
    print(f"June & Sept only (not March):           {len(june_sept_not_march):6d} listings")
    print(f"March only:                             {len(march_only):6d} listings")
    print(f"June only:                              {len(june_only):6d} listings")
    print(f"Sept only:                              {len(sept_only):6d} listings")
    print("-" * 55)
    total_unique = len(march_ids | june_ids | sept_ids)
    print(f"Total unique listings:                  {total_unique:6d}")
    print()
    
    # Now break down by "temporal coverage"
    print("Summary by temporal coverage:\n")
    
    appears_3_times = len(all_three)
    appears_2_times = len(march_june_not_sept) + len(march_sept_not_june) + len(june_sept_not_march)
    appears_1_time = len(march_only) + len(june_only) + len(sept_only)
    
    # Record count (sum of records in each category)
    records_3_times = appears_3_times * 3
    records_2_times = appears_2_times * 2
    records_1_time = appears_1_time * 1
    
    print(f"Listings appearing 3 times:             {appears_3_times:6d} unique → {records_3_times:6d} records")
    print(f"Listings appearing 2 times:             {appears_2_times:6d} unique → {records_2_times:6d} records")
    print(f"Listings appearing 1 time:              {appears_1_time:6d} unique → {records_1_time:6d} records")
    print("-" * 55)
    print(f"Total unique listings:                  {total_unique:6d}")
    print(f"Total records (summed across months):   {records_3_times + records_2_times + records_1_time:6d}")
    print(f"Actual records in dataset:              {total_records:6d}")
    print()
    
    # Percentage breakdown
    print("Percentage breakdown:\n")
    print(f"Records from 3-month listings:          {100*records_3_times/total_records:.1f}%")
    print(f"Records from 2-month listings:          {100*records_2_times/total_records:.1f}%")
    print(f"Records from 1-month listings:          {100*records_1_time/total_records:.1f}%")
    print()
    
    return {
        'all_three': len(all_three),
        'appears_2_times': appears_2_times,
        'appears_1_time': appears_1_time,
        'total_unique': total_unique,
        'total_records': total_records,
        'records_3_times': records_3_times,
        'records_2_times': records_2_times,
        'records_1_time': records_1_time,
    }


def merge_minimum_nights(merged, data_dict):
    """Add minimum_nights from original data (using March as reference)."""
    march = data_dict['march']
    min_nights_march = march[['id', 'minimum_nights']].copy()
    merged = merged.merge(min_nights_march, on='id', how='left')
    return merged


def analyze_outlier_characteristics(merged):
    """Analyze minimum_nights distribution for outliers vs normal listings."""
    outlier_mask = merged['rel_change_march_june'] > 1.0
    normal_mask = ~outlier_mask
    
    outliers = merged[outlier_mask]
    normal = merged[normal_mask]
    
    print(f"\n=== OUTLIER CHARACTERISTICS (rel_change > 100%) ===")
    print(f"Total outliers: {len(outliers)}")
    print(f"\nOutliers with minimum_nights >= 30: {len(outliers[outliers['minimum_nights'] >= 30])} ({100*len(outliers[outliers['minimum_nights'] >= 30])/len(outliers):.1f}%)")
    print(f"Outliers with minimum_nights < 30: {len(outliers[outliers['minimum_nights'] < 30])} ({100*len(outliers[outliers['minimum_nights'] < 30])/len(outliers):.1f}%)")
    
    print(f"\nOutlier minimum_nights stats:")
    print(f"  Mean: {outliers['minimum_nights'].mean():.1f}")
    print(f"  Median: {outliers['minimum_nights'].median():.1f}")
    print(f"  Max: {outliers['minimum_nights'].max():.1f}")
    
    print(f"\nNormal listings minimum_nights stats:")
    print(f"  Mean: {normal['minimum_nights'].mean():.1f}")
    print(f"  Median: {normal['minimum_nights'].median():.1f}")
    print(f"  Max: {normal['minimum_nights'].max():.1f}")
    
    print(f"\n=== PRICE RANGE ANALYSIS ===")
    print(f"Outliers with price_june > $500: {len(outliers[outliers['price_june'] > 500])} ({100*len(outliers[outliers['price_june'] > 500])/len(outliers):.1f}%)")
    print(f"Outliers with price_june > $1000: {len(outliers[outliers['price_june'] > 1000])} ({100*len(outliers[outliers['price_june'] > 1000])/len(outliers):.1f}%)")
    
    # Cross-tabulation: min_nights >= 30 AND high price
    high_price_long_term = len(outliers[(outliers['minimum_nights'] >= 30) & (outliers['price_june'] > 500)])
    print(f"Outliers with min_nights >= 30 AND price_june > $500: {high_price_long_term} ({100*high_price_long_term/len(outliers):.1f}%)")


def generate_summary_csv(data_dict, overlap_stats, output_path='../outputs/'):
    """Generate a comprehensive summary CSV for EDA report."""
    summaries = []
    
    try:
        # Overall statistics per month (after room_type filtering)
        for month, df in data_dict.items():
            # Clean price first (convert from string to float)
            df_copy = df.copy()
            df_copy = clean_price(df_copy)
            
            df_clean = df_copy.dropna(subset=['price'])
            
            summaries.append({
                'Metric': f'{month.capitalize()} - Total Records',
                'Value': len(df_copy)
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Records with Price Data',
                'Value': len(df_clean)
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - NaN Prices',
                'Value': len(df_copy) - len(df_clean)
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - NaN Price %',
                'Value': f"{100*(len(df_copy) - len(df_clean))/len(df_copy):.2f}%"
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Mean Price',
                'Value': f"${df_clean['price'].mean():.2f}"
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Median Price',
                'Value': f"${df_clean['price'].median():.2f}"
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Std Dev Price',
                'Value': f"${df_clean['price'].std():.2f}"
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Min Price',
                'Value': f"${df_clean['price'].min():.2f}"
            })
            summaries.append({
                'Metric': f'{month.capitalize()} - Max Price',
                'Value': f"${df_clean['price'].max():.2f}"
            })
            summaries.append({
                'Metric': '',
                'Value': ''
            })
        
        # Temporal coverage
        summaries.append({
            'Metric': 'Listings Appearing 3 Times',
            'Value': f"{overlap_stats['all_three']} ({100*overlap_stats['records_3_times']/overlap_stats['total_records']:.1f}% of records)"
        })
        summaries.append({
            'Metric': 'Listings Appearing 2 Times',
            'Value': f"{overlap_stats['appears_2_times']} ({100*overlap_stats['records_2_times']/overlap_stats['total_records']:.1f}% of records)"
        })
        summaries.append({
            'Metric': 'Listings Appearing 1 Time',
            'Value': f"{overlap_stats['appears_1_time']} ({100*overlap_stats['records_1_time']/overlap_stats['total_records']:.1f}% of records)"
        })
        summaries.append({
            'Metric': 'Total Unique Listings',
            'Value': overlap_stats['total_unique']
        })
        summaries.append({
            'Metric': 'Total Records',
            'Value': overlap_stats['total_records']
        })
        
        # Export to CSV
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(output_path + 'summary.csv', index=False)
        print(f"\nSummary CSV saved to {output_path}summary.csv\n")
        
    except Exception as e:
        print(f"\nError generating summary CSV: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

print("=" * 70)
print("PHASE 0: DATA FILTERING")
print("=" * 70)

# Load data
data_original = load_data()

# Filter room types (keep only Entire home/apt and Private room)
print("\nFiltering room types (keeping only Entire home/apt and Private room):\n")
data_original = filter_room_types(data_original)

print("\n" + "=" * 70)
print("PHASE 1: NaN DATA QUALITY CHECK")
print("=" * 70)

# Check NaN in each snapshot
total_records = analyze_nan_per_snapshot(data_original)

print("=" * 70)
print("PHASE 2: TEMPORAL COVERAGE ANALYSIS")
print("=" * 70)

# Analyze listing overlap structure
overlap_stats = analyze_listing_overlap_structure(data_original)

print("=" * 70)
print("PHASE 3: ANALYSIS WITH ORIGINAL DATA")
print("=" * 70)

# Continue with original analysis
print_overlap_summary(data_original)

# Extract and merge
merged_original = extract_and_merge(data_original)

# Drop NaN rows
merged_original_clean, dropped_count = drop_nan_rows(merged_original)
print(f"\nDropped {dropped_count} rows with NaN prices")
print(f"Listings analyzed: {len(merged_original_clean)} / {len(merged_original)}")

# Compute volatility metrics
merged_original_clean = compute_volatility_metrics(merged_original_clean)

# Print summaries
print_volatility_summary(merged_original_clean)

# Plot temporal changes (commented out - use standalone plotting script for report figures)
# plot_temporal_changes(merged_original_clean, output_suffix='_original')

# Plot distributions (commented out - use standalone plotting script for report figures)
# plot_price_change_distributions(merged_original_clean, output_suffix='_original')

# Print percentiles
print_percentiles(merged_original_clean, output_suffix=' (Original)')

# Export outliers
export_outliers(merged_original_clean, 'outlier_price_changes_original.txt')

# Merge minimum_nights and analyze outliers
merged_original_clean = merge_minimum_nights(merged_original_clean, data_original)
analyze_outlier_characteristics(merged_original_clean)
# estimate_price_per_night(merged_original_clean)  # Commented out for playground analysis


print("\n" + "=" * 70)
print("PHASE 4: ANALYSIS WITH FILTERED DATA (min_nights < 31)")
print("=" * 70)

# Load data again
data_filtered_raw = load_data()

# Filter out long-term rentals
data_filtered = filter_short_term_rentals(data_filtered_raw, min_nights_threshold=31)

print()
print_overlap_summary(data_filtered)

# Extract and merge
merged_filtered = extract_and_merge(data_filtered)

# Drop NaN rows
merged_filtered_clean, dropped_count = drop_nan_rows(merged_filtered)
print(f"\nDropped {dropped_count} rows with NaN prices")
print(f"Listings analyzed: {len(merged_filtered_clean)} / {len(merged_filtered)}")

# Compute volatility metrics
merged_filtered_clean = compute_volatility_metrics(merged_filtered_clean)

# Print summaries
print_volatility_summary(merged_filtered_clean)

# Plot temporal changes (commented out - use standalone plotting script for report figures)
# plot_temporal_changes(merged_filtered_clean, output_suffix='_filtered')

# Plot distributions (commented out - use standalone plotting script for report figures)
# plot_price_change_distributions(merged_filtered_clean, output_suffix='_filtered')

# Print percentiles
print_percentiles(merged_filtered_clean, output_suffix=' (Filtered)')

# Export outliers
export_outliers(merged_filtered_clean, 'outlier_price_changes_filtered.txt')

# Merge minimum_nights and analyze outliers
merged_filtered_clean = merge_minimum_nights(merged_filtered_clean, data_filtered)
analyze_outlier_characteristics(merged_filtered_clean)
# estimate_price_per_night(merged_filtered_clean)  # Commented out for playground analysis


print("=" * 70)
print("PHASE 5: COMPARISON (ORIGINAL vs FILTERED)")
print("=" * 70)
print(f"Common listings (original): {len(merged_original_clean)}")
print(f"Common listings (filtered): {len(merged_filtered_clean)}")
print(f"Listings removed by filter: {len(merged_original_clean) - len(merged_filtered_clean)}")
print(f"\nMean relative change (original): {merged_original_clean['rel_change_march_june'].mean():.4f}")
print(f"Mean relative change (filtered): {merged_filtered_clean['rel_change_march_june'].mean():.4f}")
print(f"Reduction in mean volatility: {((merged_original_clean['rel_change_march_june'].mean() - merged_filtered_clean['rel_change_march_june'].mean()) / merged_original_clean['rel_change_march_june'].mean() * 100):.2f}%")

# Generate summary CSV
print("\n" + "=" * 70)
print("GENERATING SUMMARY CSV")
print("=" * 70)
generate_summary_csv(data_original, overlap_stats)

