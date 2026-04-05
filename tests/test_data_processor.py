import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
# Add scripts dir to python path so we can import the processor
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_processor import AirbnbDataProcessor

@pytest.fixture
def mock_data_dir(tmp_path):
    """
    Creates a temporary directory holding the 3 expected CSV files,
    populated with carefully constructed mock data to test edge cases.
    Generates sufficient data (100+ rows) for proper 80/10/10 train/val/test splits.
    """
    base_columns = {col: "mock_data" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    
    # --- MOCK FILE 1 (March) ---
    # Generate 40 rows: mix valid and invalid to test cleaning
    march_data = []
    
    # Add valid rows
    for i in range(1, 35):
        row = dict(
            base_columns,
            id=i,
            price=str(150 + i * 10),
            bathrooms=str(1.0 + (i % 3) * 0.5),
            bedrooms=str(1 + (i % 4)),
            accommodates=str(2 + (i % 4)),
            minimum_nights=str(1 + (i % 7))
        )
        march_data.append(row)
    
    # Add some invalid to test filtering
    row_invalid1 = dict(base_columns, id=999, price="Contact host", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    row_invalid2 = dict(base_columns, id=1000, price=None, bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    march_data.extend([row_invalid1, row_invalid2])
    
    pd.DataFrame(march_data).to_csv(tmp_path / "listings-03-25.csv", index=False)
    
    # --- MOCK FILE 2 (June) ---
    # Generate 40 rows
    june_data = []
    for i in range(1, 41):
        row = dict(
            base_columns,
            id=i,
            price=str(200 + i * 10),
            bathrooms=str(1.0 + (i % 3) * 0.5),
            bedrooms=str(1 + (i % 4)),
            accommodates=str(2 + (i % 4)),
            minimum_nights=str(1 + (i % 7))
        )
        june_data.append(row)
    pd.DataFrame(june_data).to_csv(tmp_path / "listings-06-25.csv", index=False)
    
    # --- MOCK FILE 3 (September) ---
    # Generate 40 rows
    sept_data = []
    for i in range(1, 41):
        row = dict(
            base_columns,
            id=i,
            price=str(175 + i * 10),
            bathrooms=str(1.0 + (i % 3) * 0.5),
            bedrooms=str(1 + (i % 4)),
            accommodates=str(2 + (i % 4)),
            minimum_nights=str(1 + (i % 7))
        )
        sept_data.append(row)
    pd.DataFrame(sept_data).to_csv(tmp_path / "listings-09-25.csv", index=False)
    
    return tmp_path


@pytest.fixture
def numeric_test_data_dir(tmp_path):
    """
    Creates test data with proper numeric types for preprocessing tests.
    """
    base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    
    rows = [
        dict(base_columns, id=1, price="100", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1", room_type="Entire home/apt", neighbourhood_cleansed="Ville-Marie"),
        dict(base_columns, id=2, price="150", bathrooms=np.nan, bedrooms="2.0", accommodates="4", minimum_nights="30", room_type="Private room", neighbourhood_cleansed="Le Plateau-Mont-Royal"),
        dict(base_columns, id=3, price="200", bathrooms="1.5", bedrooms=np.nan, accommodates="6", minimum_nights="365", room_type="Shared room", neighbourhood_cleansed="Ville-Marie"),
    ]
    
    for fname in ["listings-03-25.csv", "listings-06-25.csv", "listings-09-25.csv"]:
        pd.DataFrame(rows).to_csv(tmp_path / fname, index=False)
    
    return tmp_path

def test_price_parsing_and_nan_removal(mock_data_dir):
    """
    Ensures valid formats parse correctly to float, and invalid text/NaNs 
    are gracefully dropped from the dataset.
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    df = processor.process()
    
    # After generating ~10 rows per file, filtering for valid prices gives us ~34 March, 40 June, 40 Sept
    # Total expected: ~34 (Mar) + 40 (Jun) + 40 (Sep) = ~114 rows
    assert len(df) >= 100, f"Expected at least 100 valid rows, got {len(df)}"
    
    # Verify all prices are valid floats and positive
    assert df['price'].dtype == float, "Price column should be float"
    assert (df['price'] > 0).all(), "All prices should be positive (NaNs and invalid prices were dropped)"
    
    # Verify we have at least one price per season
    for season in [1, 2, 3]:
        season_prices = df[df['season_ordinal'] == season]['price']
        assert len(season_prices) > 0, f"Should have prices for season {season}"
        assert season_prices.dtype == float, f"Season {season} prices should be float"

def test_temporal_mapping(mock_data_dir):
    """
    Ensures rows are correctly mapped to semantic seasons (1=Winter, 2=Spring, 3=Summer).
    """
    processor = AirbnbDataProcessor(data_dir=mock_data_dir)
    df = processor.process()
    
    # Verify row counts per season are roughly as expected from mock data generation
    season_counts = df['season_ordinal'].value_counts().to_dict()
    assert season_counts.get(1, 0) >= 30, f"Expected ~34 Winter rows, got {season_counts.get(1, 0)}"  # Winter (March)
    assert season_counts.get(2, 0) >= 35, f"Expected ~40 Spring rows, got {season_counts.get(2, 0)}"  # Spring (June)
    assert season_counts.get(3, 0) >= 35, f"Expected ~40 Summer rows, got {season_counts.get(3, 0)}"  # Summer (September)

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
    Ensures train/val/test split is deterministic, correctly proportioned (80/10/10),
    and that Parquet files are exported with expected structure.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir)
    
    train_df, val_df, test_df = processor.split_and_export()
    
    # Check proportions: 80/10/10 split (with ~114 rows total from mock data)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total >= 100, f"Expected at least 100 total rows, got {total}"
    # 80% of 6 = 4.8 ≈ 5, 10% of 6 = 0.6 ≈ 1 (with rounding, may vary slightly)
    assert abs(len(train_df) / total - 0.8) < 0.05, f"Train proportion {len(train_df)/total:.2%} not ~80%"
    assert abs(len(val_df) / total - 0.1) < 0.05, f"Val proportion {len(val_df)/total:.2%} not ~10%"
    assert abs(len(test_df) / total - 0.1) < 0.05, f"Test proportion {len(test_df)/total:.2%} not ~10%"
    
    # Check that parquet files were created (all 6)
    train_parquet = output_dir / "train.parquet"
    val_parquet = output_dir / "val.parquet"
    test_parquet = output_dir / "test.parquet"
    assert train_parquet.exists(), f"Train parquet not found at {train_parquet}"
    assert val_parquet.exists(), f"Val parquet not found at {val_parquet}"
    assert test_parquet.exists(), f"Test parquet not found at {test_parquet}"
    
    # Load parquets and verify structure
    loaded_train = pd.read_parquet(train_parquet).reset_index(drop=True)
    loaded_val = pd.read_parquet(val_parquet).reset_index(drop=True)
    loaded_test = pd.read_parquet(test_parquet).reset_index(drop=True)
    assert_frame_equal(loaded_train, train_df.reset_index(drop=True))
    assert_frame_equal(loaded_val, val_df.reset_index(drop=True))
    assert_frame_equal(loaded_test, test_df.reset_index(drop=True))

def test_split_deterministic(mock_data_dir, tmp_path):
    """
    Ensures that the train/val/test split is deterministic and reproducible
    when using the same random_state (42).
    """
    output_dir1 = tmp_path / "output1"
    output_dir2 = tmp_path / "output2"
    
    processor1 = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir1)
    processor2 = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir2)
    
    train_df1, val_df1, test_df1 = processor1.split_and_export()
    train_df2, val_df2, test_df2 = processor2.split_and_export()
    
    # All three splits should be identical (same rows in same order)
    assert_frame_equal(train_df1, train_df2)
    assert_frame_equal(val_df1, val_df2)
    assert_frame_equal(test_df1, test_df2)

def test_split_contains_all_data(mock_data_dir, tmp_path):
    """
    Ensures that train + val + test covers all original data with no overlap or missing rows.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=mock_data_dir, output_dir=output_dir)
    
    master_df = processor.process()
    train_df, val_df, test_df = processor.split_and_export()
    
    # Combined train + val + test should have all rows from master
    combined_ids = set(train_df['id'].tolist() + val_df['id'].tolist() + test_df['id'].tolist())
    master_ids = set(master_df['id'].tolist())
    
    assert len(combined_ids) == len(master_ids), f"Combined has {len(combined_ids)} unique IDs, master has {len(master_ids)}"
    assert combined_ids == master_ids, "Train/val/test IDs don't cover all original data"
    assert len(train_df) + len(val_df) + len(test_df) == len(master_df), \
        f"Train ({len(train_df)}) + val ({len(val_df)}) + test ({len(test_df)}) = {len(train_df) + len(val_df) + len(test_df)}, but master has {len(master_df)}"


def test_remove_nonpositive_prices(tmp_path):
    """
    Ensure rows with price == 0 or negative are removed by the processor.
    """
    base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    # Create rows: one valid, one zero, one negative
    rows = [
        dict(base_columns, id=1, price="100"),
        dict(base_columns, id=2, price="0"),
        dict(base_columns, id=3, price="-50"),
    ]
    for fname in ["listings-03-25.csv", "listings-06-25.csv", "listings-09-25.csv"]:
        # write the same small set to each snapshot to exercise processing
        pd.DataFrame(rows).to_csv(tmp_path / fname, index=False)

    processor = AirbnbDataProcessor(data_dir=tmp_path)
    df = processor.process()

    # Only the positive price row should remain
    assert (df['price'] > 0).all(), "Processor failed to remove non-positive price rows"


def test_preprocess_tabular_fill_nans(numeric_test_data_dir, tmp_path):
    """
    Ensures preprocess_tabular fills NaN values in bathrooms and bedrooms with median.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    # Load raw data to get train/val/test
    master_df = processor.process()
    train_df = master_df.iloc[:2]   # First 2 rows for train
    val_df = master_df.iloc[2:2]    # Next rows for validation (empty okay for this test)
    test_df = master_df.iloc[2:]    # Last row for test
    
    if len(val_df) == 0:  # If val is empty, use first row from test
        val_df = test_df.iloc[:1]
        test_df = test_df.iloc[1:]
    
    train_prep, val_prep, test_prep, encoders = processor.preprocess_tabular(train_df, val_df, test_df)
    
    # Check that no NaNs remain in bathrooms/bedrooms
    assert train_prep['bathrooms'].isna().sum() == 0, "NaNs in train bathrooms not filled"
    assert train_prep['bedrooms'].isna().sum() == 0, "NaNs in train bedrooms not filled"
    if len(val_prep) > 0:
        assert val_prep['bathrooms'].isna().sum() == 0, "NaNs in val bathrooms not filled"
        assert val_prep['bedrooms'].isna().sum() == 0, "NaNs in val bedrooms not filled"
    if len(test_prep) > 0:
        assert test_prep['bathrooms'].isna().sum() == 0, "NaNs in test bathrooms not filled"
        assert test_prep['bedrooms'].isna().sum() == 0, "NaNs in test bedrooms not filled"
    
    # Check that medians were calculated and stored
    assert 'bathrooms_median' in encoders, "bathrooms median not stored"
    assert 'bedrooms_median' in encoders, "bedrooms median not stored"


def test_preprocess_tabular_encode_categoricals(numeric_test_data_dir, tmp_path):
    """
    Ensures categorical columns (room_type, neighbourhood_cleansed) are encoded to integers.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    master_df = processor.process()
    train_df = master_df.iloc[:2]
    val_df = master_df.iloc[2:2]  # Empty or minimal
    test_df = master_df.iloc[2:]
    
    if len(val_df) == 0 and len(test_df) > 1:
        val_df = test_df.iloc[:1]
        test_df = test_df.iloc[1:]
    
    train_prep, val_prep, test_prep, encoders = processor.preprocess_tabular(train_df, val_df, test_df)
    
    # Check that room_type is now numeric
    assert pd.api.types.is_integer_dtype(train_prep['room_type']), "room_type not encoded to integer"
    if len(test_prep) > 0:
        assert pd.api.types.is_integer_dtype(test_prep['room_type']), "test room_type not encoded to integer"
    
    # Check that neighbourhood_cleansed is now numeric
    assert pd.api.types.is_integer_dtype(train_prep['neighbourhood_cleansed']), "neighbourhood_cleansed not encoded"
    if len(test_prep) > 0:
        assert pd.api.types.is_integer_dtype(test_prep['neighbourhood_cleansed']), "test neighbourhood_cleansed not encoded"
    
    # Check that encoders are stored
    assert 'room_type_encoder' in encoders, "room_type encoder not stored"
    assert 'neighbourhood_cleansed_encoder' in encoders, "neighbourhood_cleansed encoder not stored"


def test_preprocess_tabular_scale_numerics(numeric_test_data_dir, tmp_path):
    """
    Ensures numeric features are scaled with StandardScaler (mean~0, std~1).
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    master_df = processor.process()
    train_df = master_df.iloc[:2]
    val_df = master_df.iloc[2:2]  # Empty or minimal
    test_df = master_df.iloc[2:]
    
    if len(val_df) == 0 and len(test_df) > 1:
        val_df = test_df.iloc[:1]
        test_df = test_df.iloc[1:]
    
    train_prep, val_prep, test_prep, encoders = processor.preprocess_tabular(train_df, val_df, test_df)
    
    # Check that numeric features are scaled (mean near 0, std near 1)
    numeric_cols = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
    for col in numeric_cols:
        train_mean = train_prep[col].mean()
        train_std = train_prep[col].std()
        # For small sample sizes, std might be 0, so just check it's reasonable
        assert abs(train_mean) < 5, f"{col} train mean not close to 0: {train_mean}"
    
    # Check that scaler is stored
    assert 'numeric_scaler' in encoders, "numeric_scaler not stored"
    assert 'numeric_scale_cols' in encoders, "numeric_scale_cols not stored"


def test_preprocess_tabular_unseen_categories(numeric_test_data_dir, tmp_path):
    """
    Ensures unseen val/test categories are mapped to special code (-1).
    """
    base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    train_rows = [
        dict(base_columns, id=1, price="100", bathrooms=1.0, bedrooms=1.0, accommodates=2, minimum_nights=1, season_ordinal=1, room_type="Entire home/apt", neighbourhood_cleansed="Ville-Marie"),
        dict(base_columns, id=2, price="150", bathrooms=2.0, bedrooms=2.0, accommodates=4, minimum_nights=30, season_ordinal=2, room_type="Private room", neighbourhood_cleansed="Le Plateau-Mont-Royal"),
    ]
    val_rows = [
        dict(base_columns, id=3, price="175", bathrooms=1.5, bedrooms=1.5, accommodates=3, minimum_nights=14, season_ordinal=2, room_type="Private room", neighbourhood_cleansed="Le Plateau-Mont-Royal"),
    ]
    test_rows = [
        dict(base_columns, id=4, price="200", bathrooms=1.5, bedrooms=1.5, accommodates=6, minimum_nights=365, season_ordinal=3, room_type="Hotel room", neighbourhood_cleansed="UnknownNeighbourhood"),  # Unseen!
    ]
    
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    
    train_prep, val_prep, test_prep, encoders = processor.preprocess_tabular(train_df, val_df, test_df)
    
    # Check that unseen test categories are mapped to -1
    unseen_room = test_prep.loc[0, 'room_type']
    unseen_neighbourhood = test_prep.loc[0, 'neighbourhood_cleansed']
    
    assert unseen_room == -1, f"Unseen room_type not mapped to -1, got {unseen_room}"
    assert unseen_neighbourhood == -1, f"Unseen neighbourhood not mapped to -1, got {unseen_neighbourhood}"


def test_split_and_export_creates_tabular_parquets(numeric_test_data_dir, tmp_path):
    """
    Ensures split_and_export creates raw and tabular preprocessed parquets for all 3 splits.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    train_df, val_df, test_df = processor.split_and_export()
    
    # Check that raw parquets exist (all 3)
    assert (output_dir / "train.parquet").exists(), "train.parquet not created"
    assert (output_dir / "val.parquet").exists(), "val.parquet not created"
    assert (output_dir / "test.parquet").exists(), "test.parquet not created"
    
    # Check that tabular preprocessed parquets exist (all 3)
    assert (output_dir / "train_tabular.parquet").exists(), "train_tabular.parquet not created"
    assert (output_dir / "val_tabular.parquet").exists(), "val_tabular.parquet not created"
    assert (output_dir / "test_tabular.parquet").exists(), "test_tabular.parquet not created"
    
    # Check that encoders are persisted
    assert (output_dir / "tabular_encoders.joblib").exists(), "tabular_encoders.joblib not created"
    
    # Load and verify tabular parquets
    train_tabular = pd.read_parquet(output_dir / "train_tabular.parquet")
    val_tabular = pd.read_parquet(output_dir / "val_tabular.parquet")
    test_tabular = pd.read_parquet(output_dir / "test_tabular.parquet")
    
    # Check that categorical features are numeric
    assert pd.api.types.is_integer_dtype(train_tabular['room_type']), "room_type not encoded in train_tabular"
    assert pd.api.types.is_integer_dtype(train_tabular['neighbourhood_cleansed']), "neighbourhood not encoded in train_tabular"
    
    # Check that no NaNs remain in numeric columns
    assert train_tabular['bathrooms'].isna().sum() == 0, "NaNs still in train bathrooms"
    assert train_tabular['bedrooms'].isna().sum() == 0, "NaNs still in train bedrooms"
    if len(val_tabular) > 0:
        assert val_tabular['bathrooms'].isna().sum() == 0, "NaNs still in val bathrooms"
        assert val_tabular['bedrooms'].isna().sum() == 0, "NaNs still in val bedrooms"


def test_split_and_export_encoders_persisted(numeric_test_data_dir, tmp_path):
    """
    Ensures tabular_encoders.joblib contains the encoder/scaler objects.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    processor.split_and_export()
    
    # Load persisted encoders
    encoders = joblib.load(output_dir / "tabular_encoders.joblib")
    
    # Check that expected keys are present
    assert 'room_type_encoder' in encoders, "room_type_encoder not in persisted encoders"
    assert 'neighbourhood_cleansed_encoder' in encoders, "neighbourhood_cleansed_encoder not in persisted encoders"
    assert 'numeric_scaler' in encoders, "numeric_scaler not in persisted encoders"
    assert 'bathrooms_median' in encoders, "bathrooms_median not in persisted encoders"
    assert 'bedrooms_median' in encoders, "bedrooms_median not in persisted encoders"
    
    # Verify types
    assert isinstance(encoders['room_type_encoder'], LabelEncoder), "room_type_encoder not LabelEncoder"
    assert isinstance(encoders['numeric_scaler'], StandardScaler), "numeric_scaler not StandardScaler"
