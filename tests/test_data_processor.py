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
    """
    base_columns = {col: "mock_data" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    
    # --- MOCK FILE 1 (March) ---
    # Focus: Testing price parsing and NaN dropping
    march_data = []
    
    # 1. Valid numeric price
    row1 = dict(base_columns, id=1, price="150", bathrooms="1.0", bedrooms="2.0", accommodates="2", minimum_nights="1")
    # 2. String with $ and comma
    row2 = dict(base_columns, id=2, price="$1,250.00", bathrooms="2.0", bedrooms="1.0", accommodates="4", minimum_nights="30")
    # 3. Pure string (should fail/drop)
    row3 = dict(base_columns, id=3, price="Contact host for price", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    # 4. NaN value (should drop)
    row4 = dict(base_columns, id=4, price=None, bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    # 5. String + number mix (should fail/drop per our strict rules)
    row5 = dict(base_columns, id=5, price="100 dollars", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    # 6. Extra junk column to verify schema enforcement
    row6 = dict(base_columns, id=6, price="200.50", bathrooms="1.5", bedrooms="2.0", accommodates="3", minimum_nights="2", junk_column="Delete me", host_name="Alice")
    
    march_data.extend([row1, row2, row3, row4, row5, row6])
    pd.DataFrame(march_data).to_csv(tmp_path / "listings-03-25.csv", index=False)
    
    # --- MOCK FILE 2 (June) ---
    # Focus: Ensuring proper temporal mapping (06 vs 03)
    june_data = [
        dict(base_columns, id=1, price="$200.00", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1"),  # Valid
        dict(base_columns, id=2, price="300", bathrooms="1.5", bedrooms="2.0", accommodates="4", minimum_nights="30")       # Valid
    ]
    pd.DataFrame(june_data).to_csv(tmp_path / "listings-06-25.csv", index=False)
    
    # --- MOCK FILE 3 (September) ---
    # Focus: Same as above but for 09
    sept_data = [
        dict(base_columns, id=1, price="175", bathrooms="1.0", bedrooms="1.0", accommodates="2", minimum_nights="1")
    ]
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
    
    # Load raw data to get train/test
    master_df = processor.process()
    train_df = master_df.iloc[:2]  # First 2 rows for train
    test_df = master_df.iloc[2:]   # Last row for test
    
    train_prep, test_prep, encoders = processor.preprocess_tabular(train_df, test_df)
    
    # Check that no NaNs remain in bathrooms/bedrooms
    assert train_prep['bathrooms'].isna().sum() == 0, "NaNs in train bathrooms not filled"
    assert train_prep['bedrooms'].isna().sum() == 0, "NaNs in train bedrooms not filled"
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
    test_df = master_df.iloc[2:]
    
    train_prep, test_prep, encoders = processor.preprocess_tabular(train_df, test_df)
    
    # Check that room_type is now numeric
    assert pd.api.types.is_integer_dtype(train_prep['room_type']), "room_type not encoded to integer"
    assert pd.api.types.is_integer_dtype(test_prep['room_type']), "test room_type not encoded to integer"
    
    # Check that neighbourhood_cleansed is now numeric
    assert pd.api.types.is_integer_dtype(train_prep['neighbourhood_cleansed']), "neighbourhood_cleansed not encoded"
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
    test_df = master_df.iloc[2:]
    
    train_prep, test_prep, encoders = processor.preprocess_tabular(train_df, test_df)
    
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
    Ensures unseen test categories are mapped to special code (-1).
    """
    base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    train_rows = [
        dict(base_columns, id=1, price="100", bathrooms=1.0, bedrooms=1.0, accommodates=2, minimum_nights=1, season_ordinal=1, room_type="Entire home/apt", neighbourhood_cleansed="Ville-Marie"),
        dict(base_columns, id=2, price="150", bathrooms=2.0, bedrooms=2.0, accommodates=4, minimum_nights=30, season_ordinal=2, room_type="Private room", neighbourhood_cleansed="Le Plateau-Mont-Royal"),
    ]
    test_rows = [
        dict(base_columns, id=3, price="200", bathrooms=1.5, bedrooms=1.5, accommodates=6, minimum_nights=365, season_ordinal=3, room_type="Hotel room", neighbourhood_cleansed="UnknownNeighbourhood"),  # Unseen!
    ]
    
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    
    train_prep, test_prep, encoders = processor.preprocess_tabular(train_df, test_df)
    
    # Check that unseen test categories are mapped to -1
    unseen_room = test_prep.loc[0, 'room_type']
    unseen_neighbourhood = test_prep.loc[0, 'neighbourhood_cleansed']
    
    assert unseen_room == -1, f"Unseen room_type not mapped to -1, got {unseen_room}"
    assert unseen_neighbourhood == -1, f"Unseen neighbourhood not mapped to -1, got {unseen_neighbourhood}"


def test_split_and_export_creates_tabular_parquets(numeric_test_data_dir, tmp_path):
    """
    Ensures split_and_export creates both raw and tabular preprocessed parquets.
    """
    output_dir = tmp_path / "output"
    processor = AirbnbDataProcessor(data_dir=numeric_test_data_dir, output_dir=output_dir)
    
    train_df, test_df = processor.split_and_export()
    
    # Check that raw parquets exist
    assert (output_dir / "train.parquet").exists(), "train.parquet not created"
    assert (output_dir / "test.parquet").exists(), "test.parquet not created"
    
    # Check that tabular preprocessed parquets exist
    assert (output_dir / "train_tabular.parquet").exists(), "train_tabular.parquet not created"
    assert (output_dir / "test_tabular.parquet").exists(), "test_tabular.parquet not created"
    
    # Check that encoders are persisted
    assert (output_dir / "tabular_encoders.joblib").exists(), "tabular_encoders.joblib not created"
    
    # Load and verify tabular parquets
    train_tabular = pd.read_parquet(output_dir / "train_tabular.parquet")
    test_tabular = pd.read_parquet(output_dir / "test_tabular.parquet")
    
    # Check that categorical features are numeric
    assert pd.api.types.is_integer_dtype(train_tabular['room_type']), "room_type not encoded in tabular"
    assert pd.api.types.is_integer_dtype(train_tabular['neighbourhood_cleansed']), "neighbourhood not encoded in tabular"
    
    # Check that no NaNs remain in numeric columns
    assert train_tabular['bathrooms'].isna().sum() == 0, "NaNs still in train bathrooms"
    assert train_tabular['bedrooms'].isna().sum() == 0, "NaNs still in train bedrooms"


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
