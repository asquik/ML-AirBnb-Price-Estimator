import pytest
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal

import sys
# Add scripts dir to python path so we can import the processor
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_processor import AirbnbDataProcessor

@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Creates a temporary directory holding the 3 expected CSV files,
    populated with carefully constructed mock data to test edge cases.
    """
    base_columns = {col: "mock_data" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    
    # --- MOCK FILE 1 (March) ---
    # Focus: Testing price parsing and NaN dropping
    march_data = []
    
    # 1. Valid numeric price
    row1 = dict(base_columns, id=1, price="150")
    # 2. String with $ and comma
    row2 = dict(base_columns, id=2, price="$1,250.00")
    # 3. Pure string (should fail/drop)
    row3 = dict(base_columns, id=3, price="Contact host for price")
    # 4. NaN value (should drop)
    row4 = dict(base_columns, id=4, price=None)
    # 5. String + number mix (should fail/drop per our strict rules)
    row5 = dict(base_columns, id=5, price="100 dollars")
    # 6. Extra junk column to verify schema enforcement
    row6 = dict(base_columns, id=6, price="200.50", junk_column="Delete me", host_name="Alice")
    
    march_data.extend([row1, row2, row3, row4, row5, row6])
    pd.DataFrame(march_data).to_csv(tmp_path / "listings-03-25.csv", index=False)
    
    # --- MOCK FILE 2 (June) ---
    # Focus: Ensuring proper temporal mapping (06 vs 03)
    june_data = [
        dict(base_columns, id=1, price="$200.00"),  # Valid
        dict(base_columns, id=2, price="300")       # Valid
    ]
    pd.DataFrame(june_data).to_csv(tmp_path / "listings-06-25.csv", index=False)
    
    # --- MOCK FILE 3 (September) ---
    # Focus: Same as above but for 09
    sept_data = [
        dict(base_columns, id=1, price="175")
    ]
    pd.DataFrame(sept_data).to_csv(tmp_path / "listings-09-25.csv", index=False)
    
    return tmp_path

def test_price_parsing_and_nan_removal(mock_data_dir):
    """
    Ensures valid formats parse correctly to float, and invalid text/NaNs 
    are gracefully dropped from the dataset.
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    df = processor.process()
    
    # Only rows 1, 2, and 6 from March, both from June, and the 1 from Sept should survive.
    # Total expected: 3 (Mar) + 2 (Jun) + 1 (Sep) = 6 valid rows
    assert len(df) == 6, f"Expected 6 valid rows, got {len(df)}"
    
    march_df = df[df['season_ordinal'] == 1]
    prices = march_df.set_index('id')['price'].to_dict()
    
    assert prices[1] == 150.0  # Basic numeric parsed
    assert prices[2] == 1250.0 # $ and comma removed correctly
    assert prices[6] == 200.50 # Float parsed correctly
    
    # Assert invalid IDs were completely dropped
    assert 3 not in prices # "Contact host"
    assert 4 not in prices # NaN
    assert 5 not in prices # "100 dollars"

def test_temporal_mapping(mock_data_dir):
    """
    Ensures rows are correctly mapped to semantic seasons (1=Winter, 2=Spring, 3=Summer).
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    df = processor.process()
    
    # Verify exact row counts per season match what we explicitly kept (valid targets only)
    season_counts = df['season_ordinal'].value_counts().to_dict()
    assert season_counts.get(1) == 3  # Winter (March)
    assert season_counts.get(2) == 2  # Spring (June)
    assert season_counts.get(3) == 1  # Summer (September)

def test_schema_enforcement(mock_data_dir):
    """
    Ensures ONLY explicitly required columns (+ season_ordinal) are returned,
    even if the raw CSVs contain extra junk columns.
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    df = processor.process()
    
    expected_columns = set(AirbnbDataProcessor.REQUIRED_COLUMNS + ['season_ordinal'])
    actual_columns = set(df.columns)
    
    assert actual_columns == expected_columns, "Processor leaked unexpected columns or dropped required ones."
    assert "junk_column" not in df.columns
    assert "host_name" not in df.columns

def test_missing_file_fails_loudly(tmp_path):
    """
    Ensures the system fails immediately and loudly if a snapshot file
    is entirely missing, rather than generating partial data quietly.
    """
    # Create only march and june, leave september missing
    pd.DataFrame(columns=AirbnbDataProcessor.REQUIRED_COLUMNS).to_csv(tmp_path / "listings-03-25.csv", index=False)
    pd.DataFrame(columns=AirbnbDataProcessor.REQUIRED_COLUMNS).to_csv(tmp_path / "listings-06-25.csv", index=False)
    
    processor = AirbnbDataProcessor(data_dir=tmp_path)
    
    with pytest.raises(FileNotFoundError, match="listings-09-25.csv"):
        processor.process()

def test_reproducibility(mock_data_dir):
    """
    Ensures that calling `.process()` twice on the same data yields the exact
    same DataFrame (no weird side effects, state mutations, or random shuffling).
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    
    df1 = processor.process()
    df2 = processor.process()
    
    assert_frame_equal(df1, df2)

def test_split_and_export(mock_data_dir, tmp_path):
    """
    Ensures train/test split is deterministic, correctly proportioned,
    and that Parquet files are exported with expected structure.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir)
    
    train_df, test_df = processor.split_and_export()
    
    # Check proportions: 80/20 split (80% train, 20% test)
    total = len(train_df) + len(test_df)
    assert total == 6, f"Expected 6 total rows, got {total}"
    assert len(train_df) == 5, f"Expected 5 train rows (~80%), got {len(train_df)}"
    assert len(test_df) == 1, f"Expected 1 test row (~20%), got {len(test_df)}"
    
    # Check that parquet files were created
    train_parquet = output_dir / "train.parquet"
    test_parquet = output_dir / "test.parquet"
    assert train_parquet.exists(), f"Train parquet not found at {train_parquet}"
    assert test_parquet.exists(), f"Test parquet not found at {test_parquet}"
    
    # Load parquets and verify structure (reset index since parquet saves with index=False)
    loaded_train = pd.read_parquet(train_parquet).reset_index(drop=True)
    loaded_test = pd.read_parquet(test_parquet).reset_index(drop=True)
    assert_frame_equal(loaded_train, train_df.reset_index(drop=True))
    assert_frame_equal(loaded_test, test_df.reset_index(drop=True))

def test_split_deterministic(mock_data_dir, tmp_path):
    """
    Ensures that the train/test split is deterministic and reproducible
    when using the same random_state (42).
    """
    output_dir1 = tmp_path / "output1"
    output_dir2 = tmp_path / "output2"
    
    processor1 = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir1)
    processor2 = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir2)
    
    train_df1, test_df1 = processor1.split_and_export()
    train_df2, test_df2 = processor2.split_and_export()
    
    # Both splits should be identical (same rows in same order)
    assert_frame_equal(train_df1, train_df2)
    assert_frame_equal(test_df1, test_df2)

def test_split_contains_all_data(mock_data_dir, tmp_path):
    """
    Ensures that train + test covers all original data with no overlap or missing rows.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir)
    
    master_df = processor.process()
    train_df, test_df = processor.split_and_export()
    
    # Combined train + test should have all rows from master
    combined_ids = set(train_df['id'].tolist() + test_df['id'].tolist())
    master_ids = set(master_df['id'].tolist())
    
    assert len(combined_ids) == len(master_ids), f"Combined has {len(combined_ids)} unique IDs, master has {len(master_ids)}"
    assert combined_ids == master_ids, "Train/test IDs don't cover all original data"
    assert len(train_df) + len(test_df) == len(master_df), f"Train ({len(train_df)}) + test ({len(test_df)}) = {len(train_df) + len(test_df)}, but master has {len(master_df)}"
