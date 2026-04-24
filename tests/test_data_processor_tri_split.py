"""
Test suite for train-validation-test split functionality.
These tests define the expected behavior BEFORE implementation.
Run these first (they should fail), then implement data_processor.py changes until they pass.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_processor import AirbnbDataProcessor


@pytest.fixture
def numeric_test_data_dir_tri_split(tmp_path):
    """
    Creates test data for tri-split testing.
    Mock data with sufficient records to test 80/10/10 split.
    """
    base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}
    
    # Create 110 rows to test proportions (80% = 88, 10% = 11, 10% = 11)
    rows = []
    for i in range(110):
        season = (i % 3) + 1  # Cycle through seasons 1, 2, 3
        row = dict(
            base_columns,
            id=i+1,
            price=str(100 + i),
            bathrooms=float(i % 5 + 1),
            bedrooms=float(i % 4 + 1),
            accommodates=i % 6 + 1,
            minimum_nights=1,
            room_type=["Entire home/apt", "Private room", "Shared room"][i % 3],
            neighbourhood_cleansed=["Ville-Marie", "Le Plateau-Mont-Royal", "Griffintown"][i % 3],
            season_ordinal=season
        )
        rows.append(row)
    
    df = pd.DataFrame(rows)
    for fname in ["listings-03-25.csv", "listings-06-25.csv", "listings-09-25.csv"]:
        df.to_csv(tmp_path / fname, index=False)
    
    return tmp_path


class TestTriSplitProportions:
    """Test that splits achieve correct 80/10/10 proportions."""
    
    def test_split_80_10_10_exact_proportions(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify 80/10/10 split ratios."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        train_df, val_df, test_df = processor.split_and_export()
        
        total = len(train_df) + len(val_df) + len(test_df)
        train_pct = len(train_df) / total
        val_pct = len(val_df) / total
        test_pct = len(test_df) / total
        
        # Allow 1% tolerance for rounding
        assert 0.79 <= train_pct <= 0.81, f"Train proportion {train_pct:.2%} not ~80%"
        assert 0.09 <= val_pct <= 0.11, f"Val proportion {val_pct:.2%} not ~10%"
        assert 0.09 <= test_pct <= 0.11, f"Test proportion {test_pct:.2%} not ~10%"
    
    def test_split_covers_all_data(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify train + val + test covers all original data."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        master_df = processor.process()
        train_df, val_df, test_df = processor.split_and_export()
        
        total_split = len(train_df) + len(val_df) + len(test_df)
        assert total_split == len(master_df), \
            f"Split total ({total_split}) != master ({len(master_df)})"


class TestTriSplitNoOverlap:
    """Test that splits have no row overlap or loss."""
    
    def test_splits_are_disjoint(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify no row appears in multiple splits."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        train_df, val_df, test_df = processor.split_and_export()
        
        train_ids = set(train_df['id'].tolist())
        val_ids = set(val_df['id'].tolist())
        test_ids = set(test_df['id'].tolist())
        
        # Check pairwise disjointness
        assert len(train_ids & val_ids) == 0, "Train and Val have overlapping IDs"
        assert len(train_ids & test_ids) == 0, "Train and Test have overlapping IDs"
        assert len(val_ids & test_ids) == 0, "Val and Test have overlapping IDs"
    
    def test_all_ids_accounted_for(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify every record appears in exactly one split."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        master_df = processor.process()
        train_df, val_df, test_df = processor.split_and_export()
        
        master_ids = set(master_df['id'].tolist())
        split_ids = set(train_df['id'].tolist()) | set(val_df['id'].tolist()) | set(test_df['id'].tolist())
        
        assert master_ids == split_ids, "Some records missing or extra records present"


class TestTriSplitDeterministic:
    """Test that splits are reproducible with same seed."""
    
    def test_tri_split_deterministic(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify same seed produces identical splits."""
        output_dir1 = tmp_path / "output1"
        output_dir2 = tmp_path / "output2"
        
        processor1 = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir1
        )
        processor2 = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir2
        )
        
        train1, val1, test1 = processor1.split_and_export()
        train2, val2, test2 = processor2.split_and_export()
        
        # Compare IDs (order-independent)
        assert set(train1['id']) == set(train2['id']), "Train splits differ"
        assert set(val1['id']) == set(val2['id']), "Val splits differ"
        assert set(test1['id']) == set(test2['id']), "Test splits differ"


class TestTriSplitParquetExports:
    """Test that all 6 parquet files are created correctly."""
    
    def test_all_six_parquets_created(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify all parquet files are exported."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        required_files = [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "train_tabular.parquet",
            "val_tabular.parquet",
            "test_tabular.parquet",
        ]
        
        for fname in required_files:
            fpath = output_dir / fname
            assert fpath.exists(), f"Expected parquet not found: {fname}"
    
    def test_parquet_row_counts_match_splits(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify parquet files contain correct row counts."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        train_df, val_df, test_df = processor.split_and_export()
        
        # Load from parquets and compare
        train_parquet = pd.read_parquet(output_dir / "train.parquet")
        val_parquet = pd.read_parquet(output_dir / "val.parquet")
        test_parquet = pd.read_parquet(output_dir / "test.parquet")
        
        assert len(train_parquet) == len(train_df), "Train parquet row count mismatch"
        assert len(val_parquet) == len(val_df), "Val parquet row count mismatch"
        assert len(test_parquet) == len(test_df), "Test parquet row count mismatch"
    
    def test_tabular_parquets_have_encoded_features(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify tabular parquets have encoded/scaled features."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        train_tabular = pd.read_parquet(output_dir / "train_tabular.parquet")
        val_tabular = pd.read_parquet(output_dir / "val_tabular.parquet")
        test_tabular = pd.read_parquet(output_dir / "test_tabular.parquet")
        
        # Check that categoricals are numeric (encoded)
        assert pd.api.types.is_integer_dtype(train_tabular['room_type']), \
            "room_type not encoded in train_tabular"
        assert pd.api.types.is_integer_dtype(val_tabular['room_type']), \
            "room_type not encoded in val_tabular"
        assert pd.api.types.is_integer_dtype(test_tabular['room_type']), \
            "room_type not encoded in test_tabular"


class TestTaggedExports:
    """Test optional tagged export outputs (e.g., cleaned datasets) are created."""

    def test_split_and_export_with_tag_creates_suffixed_files(self, numeric_test_data_dir_tri_split, tmp_path):
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )

        processor.split_and_export(max_price=5000, file_tag="cleaned")

        required_files = [
            "train_cleaned.parquet",
            "val_cleaned.parquet",
            "test_cleaned.parquet",
            "train_tabular_cleaned.parquet",
            "val_tabular_cleaned.parquet",
            "test_tabular_cleaned.parquet",
            "price_transformer_cleaned.joblib",
            "tabular_encoders_cleaned.joblib",
        ]

        for fname in required_files:
            assert (output_dir / fname).exists(), f"Expected tagged artifact not found: {fname}"


class TestEncoderNoLeakage:
    """Test that encoders are fit on train set only (critical for preventing data leakage)."""
    
    def test_encoder_fit_on_train_only_categories(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify encoder classes match only training set categories."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        train_df, val_df, test_df = processor.split_and_export()
        
        # Load encoders
        encoders = joblib.load(output_dir / "tabular_encoders.joblib")
        room_type_encoder = encoders['room_type_encoder']
        
        # Check that encoder was fit on training set categories
        train_categories = set(train_df[train_df['room_type'].notna()]['room_type'].unique())
        encoder_categories = set(room_type_encoder.classes_)
        
        # All encoder classes should come from training set
        assert encoder_categories == train_categories, \
            f"Encoder has unseen categories: {encoder_categories - train_categories}"
    
    def test_unseen_categories_in_val_mapped_to_minus_one(self, tmp_path):
        """Verify validation/test categories not seen in training map to -1."""
        base_columns = {col: "mock" for col in AirbnbDataProcessor.REQUIRED_COLUMNS}

        processor = AirbnbDataProcessor(
            data_dir=tmp_path,
            output_dir=tmp_path / "output"
        )
        
        # Create deliberately mismatched data for train vs val/test
        train_rows = [
            dict(base_columns, id=i, price=str(100+i), bathrooms="1", bedrooms="1", 
                 accommodates=2, minimum_nights=1, room_type="Entire home/apt",
                 neighbourhood_cleansed="Ville-Marie", season_ordinal=1)
            for i in range(1, 81)  # 80 train rows with only "Entire home/apt"
        ]
        
        # Val/test will have unseen category
        val_test_rows = [
            dict(base_columns, id=i+100, price=str(100+i), bathrooms="1", bedrooms="1",
                 accommodates=2, minimum_nights=1, room_type="Hotel room",  # UNSEEN
                 neighbourhood_cleansed="Ville-Marie", season_ordinal=1)
            for i in range(1, 11)  # 10 val rows
        ]
        
        df_train = pd.DataFrame(train_rows)
        df_val = pd.DataFrame(val_test_rows)
        df_test = df_val.copy()
        
        # Preprocess: train encoder should only see "Entire home/apt"
        train_prep, val_prep, _test_prep, encoders = processor.preprocess_tabular(df_train, df_val, df_test)
        
        # Check that unseen "Hotel room" becomes -1 in validation
        assert (val_prep['room_type'] == -1).all(), \
            "Unseen room_type 'Hotel room' should map to -1, got: " + str(val_prep['room_type'].unique())


class TestTriSplitTemporalBalance:
    """Test that all three splits contain records from all seasons."""
    
    def test_all_splits_have_all_seasons(self, numeric_test_data_dir_tri_split, tmp_path):
        """Ensure temporal data is balanced across splits."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        train_df, val_df, test_df = processor.split_and_export()
        
        train_seasons = set(train_df['season_ordinal'].unique())
        val_seasons = set(val_df['season_ordinal'].unique())
        test_seasons = set(test_df['season_ordinal'].unique())
        
        expected_seasons = {1, 2, 3}
        
        assert train_seasons == expected_seasons, \
            f"Train missing seasons: {expected_seasons - train_seasons}"
        assert val_seasons == expected_seasons, \
            f"Val missing seasons: {expected_seasons - val_seasons}"
        assert test_seasons == expected_seasons, \
            f"Test missing seasons: {expected_seasons - test_seasons}"


class TestEncoderPersistence:
    """Test that encoders and scalers are persisted correctly."""
    
    def test_encoders_persisted_correctly(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify tabular_encoders.joblib contains expected objects."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        encoders = joblib.load(output_dir / "tabular_encoders.joblib")
        
        # Check required keys
        required_keys = [
            'room_type_encoder',
            'neighbourhood_cleansed_encoder',
            'numeric_scaler',
            'bathrooms_median',
            'bedrooms_median',
            'numeric_scale_cols'
        ]
        
        for key in required_keys:
            assert key in encoders, f"Missing key in encoders: {key}"
        
        # Check types
        assert isinstance(encoders['room_type_encoder'], LabelEncoder)
        assert isinstance(encoders['neighbourhood_cleansed_encoder'], LabelEncoder)
        assert isinstance(encoders['numeric_scaler'], StandardScaler)
        assert isinstance(encoders['bathrooms_median'], (float, np.floating))
        assert isinstance(encoders['bedrooms_median'], (float, np.floating))
        assert isinstance(encoders['numeric_scale_cols'], list)


class TestNumericFeatureScaling:
    """Test that numeric features are scaled correctly (train only)."""
    
    def test_numeric_scaled_using_train_statistics(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify ScalerStandard is fit on train, applied to val/test."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        train_tabular = pd.read_parquet(output_dir / "train_tabular.parquet")
        val_tabular = pd.read_parquet(output_dir / "val_tabular.parquet")
        
        numeric_cols = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
        
        # Train mean should be close to 0, std close to 1
        train_means = train_tabular[numeric_cols].mean()
        train_stds = train_tabular[numeric_cols].std()
        
        for col in numeric_cols:
            assert abs(train_means[col]) < 1.0, \
                f"Train {col} mean {train_means[col]:.4f} not close to 0"
            # Some synthetic fixtures may produce constant columns (std=0); StandardScaler will output zeros.
            if train_stds[col] != 0:
                assert 0.8 < train_stds[col] < 1.2, \
                    f"Train {col} std {train_stds[col]:.4f} not close to 1"
        
        # Val mean and std should NOT be 0 and 1
        # (because scaler was fit on train, val gets different statistics)
        val_means = val_tabular[numeric_cols].mean()
        
        # At least one column should have non-zero mean in val
        # (very unlikely all would be exactly 0)
        assert any(abs(val_means) > 0.1), \
            "Val set has suspiciously zero-mean features (scaler may not be independent)"


class TestPriceTransformerPersistence:
    """Test that Box-Cox price transformer is persisted."""
    
    def test_price_transformer_persisted(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify price_transformer.joblib is saved."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        transformer_path = output_dir / "price_transformer.joblib"
        assert transformer_path.exists(), "price_transformer.joblib not found"
        
        transformer = joblib.load(transformer_path)
        assert hasattr(transformer, 'transform'), "Transformer missing transform method"
    
    def test_price_bc_column_in_all_splits(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify price_bc (Box-Cox transformed price) appears in all splits."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        processor.split_and_export()
        
        train = pd.read_parquet(output_dir / "train.parquet")
        val = pd.read_parquet(output_dir / "val.parquet")
        test = pd.read_parquet(output_dir / "test.parquet")
        
        assert 'price_bc' in train.columns, "price_bc missing in train"
        assert 'price_bc' in val.columns, "price_bc missing in val"
        assert 'price_bc' in test.columns, "price_bc missing in test"


# ============================================================================
# UPDATE EXISTING split_and_export METHOD SIGNATURE
# ============================================================================

class TestSplitAndExportReturnValue:
    """Test that split_and_export() returns THREE dataframes instead of two."""
    
    def test_split_and_export_returns_three_dataframes(self, numeric_test_data_dir_tri_split, tmp_path):
        """Verify updated method signature: (train, val, test) not (train, test)."""
        output_dir = tmp_path / "output"
        processor = AirbnbDataProcessor(
            data_dir=numeric_test_data_dir_tri_split,
            output_dir=output_dir
        )
        
        result = processor.split_and_export()
        
        # Should be a tuple of 3 DataFrames
        assert isinstance(result, tuple), "split_and_export should return tuple"
        assert len(result) == 3, f"split_and_export should return 3 items, got {len(result)}"
        
        train_df, val_df, test_df = result
        assert isinstance(train_df, pd.DataFrame), "First return value should be DataFrame"
        assert isinstance(val_df, pd.DataFrame), "Second return value should be DataFrame"
        assert isinstance(test_df, pd.DataFrame), "Third return value should be DataFrame"
