"""
Explore all available columns in the raw CSV files to identify candidates for feature engineering.
Reports: column names, data types, missing %, sample values, cardinality for categoricals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Current features being used by the model
CURRENT_FEATURES = {
    "id",
    "description",
    "amenities",
    "picture_url",
    "room_type",
    "neighbourhood_cleansed",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "minimum_nights",
    "price",
}

def analyze_csv_files():
    """Load and analyze all CSV snapshots to identify feature opportunities."""
    
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root
    
    snapshot_files = [
        "listings-03-25.csv",
        "listings-06-25.csv",
        "listings-09-25.csv"
    ]
    
    print("=" * 100)
    print("CSV FEATURE EXPLORATION & DATA QUALITY ASSESSMENT")
    print("=" * 100)
    
    all_dataframes = []
    
    for fname in snapshot_files:
        fpath = data_dir / fname
        print(f"\n📄 Loading {fname}...")
        df = pd.read_csv(fpath)
        print(f"   Rows: {len(df):,}")
        all_dataframes.append(df)
    
    # Combine all snapshots
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ Combined dataset: {len(combined_df):,} rows")
    
    # List all columns
    all_columns = sorted(combined_df.columns.tolist())
    print(f"\n📊 Total columns available: {len(all_columns)}")
    print(f"    Columns: {all_columns}\n")
    
    # Categorize columns
    used_columns = set(all_columns) & CURRENT_FEATURES
    unused_columns = set(all_columns) - CURRENT_FEATURES
    
    print(f"✅ Currently used ({len(used_columns)}): {sorted(used_columns)}")
    print(f"\n❓ Available but unused ({len(unused_columns)}): {sorted(unused_columns)}")
    
    # Analyze each unused column for potential
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS OF UNUSED COLUMNS")
    print("=" * 100)
    
    candidates = []
    
    for col in sorted(unused_columns):
        data = combined_df[col]
        null_pct = 100 * data.isna().sum() / len(data)
        dtype = data.dtype
        
        # Skip if mostly null
        if null_pct > 80:
            print(f"\n🟡 {col}")
            print(f"   Type: {dtype} | Missing: {null_pct:.1f}% | ⚠️ Too sparse, skipping")
            continue
        
        print(f"\n🟢 {col}")
        print(f"   Type: {dtype} | Missing: {null_pct:.1f}%")
        
        # Get non-null data
        valid_data = data.dropna()
        
        if dtype == 'object':
            # String/categorical column
            cardinality = valid_data.nunique()
            print(f"   Cardinality: {cardinality:,} unique values")
            
            # Show most common values
            top_5 = valid_data.value_counts().head(5)
            print(f"   Top 5 values:")
            for val, cnt in top_5.items():
                trunc = str(val)[:60]
                print(f"      - {trunc}: {cnt:,} ({100*cnt/len(valid_data):.1f}%)")
            
            # Assess as feature candidate
            if cardinality <= 50 and null_pct < 30:
                candidates.append((col, "categorical", cardinality, 100 - null_pct))
                print(f"   ✅ GOOD CANDIDATE: Moderate cardinality, low missingness")
            elif cardinality <= 20 and null_pct < 20:
                candidates.append((col, "categorical", cardinality, 100 - null_pct))
                print(f"   ✅ GOOD CANDIDATE: Low cardinality, very low missingness")
            else:
                print(f"   ⚠️ Moderate candidate: High cardinality or significant missingness")
        
        elif dtype in ('int64', 'int32', 'float64', 'float32'):
            # Numeric column
            print(f"   Stats: min={valid_data.min():.2f}, max={valid_data.max():.2f}, mean={valid_data.mean():.2f}")
            
            # Check for constant/near-constant
            if valid_data.std() == 0:
                print(f"   ⚠️ CONSTANT value, no variance → not useful")
            elif null_pct < 30:
                candidates.append((col, "numeric", valid_data.nunique(), 100 - null_pct))
                print(f"   ✅ GOOD CANDIDATE: Numeric with variance, low missingness")
            else:
                print(f"   ⚠️ Moderate candidate: Significant missingness")
        
        else:
            # Other types (datetime, bool, etc.)
            print(f"   Sample values: {valid_data.head(3).tolist()}")
            if null_pct < 30:
                candidates.append((col, str(dtype), valid_data.nunique(), 100 - null_pct))
                print(f"   ✅ GOOD CANDIDATE: May be useful with transformation")
    
    # Summary of top candidates
    print("\n" + "=" * 100)
    print("TOP CANDIDATES FOR FEATURE ENGINEERING")
    print("=" * 100)
    
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda x: x[3], reverse=True)  # Sort by data completeness
        
        for col, col_type, card, completeness in candidates_sorted[:10]:
            print(f"\n{col}")
            print(f"  Type: {col_type} | Completeness: {completeness:.1f}%", end="")
            if col_type == "categorical":
                print(f" | Cardinality: {card}")
            else:
                print(f" | Unique values: {card}")
    else:
        print("No strong candidates found beyond current features.")
    
    # Save detailed report
    report_path = project_root / "outputs" / "csv_feature_analysis.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("CSV FEATURE EXPLORATION REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total columns: {len(all_columns)}\n")
        f.write(f"Currently used: {len(used_columns)}\n")
        f.write(f"Available for feature engineering: {len(unused_columns)}\n\n")
        f.write("Candidates ranked by completeness:\n")
        if candidates:
            for col, col_type, card, completeness in sorted(candidates, key=lambda x: x[3], reverse=True)[:15]:
                f.write(f"  {col} ({col_type}, {completeness:.1f}% complete)\n")
    
    print(f"\n✅ Detailed report saved to {report_path}")


if __name__ == "__main__":
    analyze_csv_files()
