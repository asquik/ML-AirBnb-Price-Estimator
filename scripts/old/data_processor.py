import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

class AirbnbDataProcessor:
    """
    A unified data processor that loads raw Airbnb monthly snapshots,
    applies universal cleaning rules, splits into deterministic train/test sets,
    and exports as compressed Parquet files.
    
    This ensures all downstream models (tabular, text, image, multi-modal) 
    are evaluated on the exact same train/test split.
    
    Seasons are encoded ordinally to prevent spurious correlation with raw month numbers.
    Mapping: Winter=1 (Oct-Apr), Spring=2 (Apr-Jun), Summer=3 (Jun-Oct).
    """
    
    # We load a fixed set of columns used across modalities.
    # If you add a column here, update tests and downstream training scripts accordingly.
    REQUIRED_COLUMNS = [
        "id",                       # Needed for split grouping & identification
        "description",              # Text modality
        "amenities",                # Text modality
        "picture_url",              # Image modality
        "room_type",                # Tabular modality
        "neighbourhood_cleansed",   # Tabular modality
        "accommodates",             # Tabular modality
        "bathrooms",                # Tabular modality
        "bedrooms",                 # Tabular modality
        "minimum_nights",           # Tabular modality
        # Expanded tabular feature set (April 2026)
        "beds",
        "host_total_listings_count",
        "latitude",
        "longitude",
        "property_type",
        "instant_bookable",
        "availability_365",
        "number_of_reviews",
        "price"                     # Target variable
    ]

    # Seasonal mapping: month strings (from filename) → ordinal season codes
    MONTH_TO_SEASON = {
        "03": 1,  # March → Winter (October-April)
        "06": 2,  # June → Spring (April-June)
        "09": 3,  # September → Summer (June-October)
    }
    SEASON_NAMES = {1: "Winter", 2: "Spring", 3: "Summer"}

    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(data_dir) / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_files = [
            "listings-03-25.csv",
            "listings-06-25.csv",
            "listings-09-25.csv"
        ]
        # Fixed seed for reproducible train/val/test split
        self.RANDOM_STATE = 42
        self.TRAIN_SPLIT = 0.8
        self.VAL_SPLIT = 0.1
        self.TEST_SPLIT = 0.1

    def _load_snapshots(self) -> pd.DataFrame:
        """Loads and concatenates the monthly snapshot CSVs."""
        dataframes = []
        for file_name in self.snapshot_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Expected data file not found: {file_path}")
            
            # Extract month (03, 06, 09) from filename and map to semantic season
            month = file_name.split("-")[1]
            season_ordinal = self.MONTH_TO_SEASON.get(month)
            if season_ordinal is None:
                raise ValueError(f"Unknown month code in file {file_name}: {month}")
            
            df = pd.read_csv(file_path, usecols=lambda c: c in self.REQUIRED_COLUMNS)
            df['season_ordinal'] = season_ordinal  # Clean semantic feature for models
            dataframes.append(df)
            
        return pd.concat(dataframes, ignore_index=True)

    def _clean_price(self, df: pd.DataFrame, max_price: float = None) -> pd.DataFrame:
        """Cleans the target price column and removes invalid/missing values. Keeps only raw price.
        
        Args:
            df: DataFrame with price column
            max_price: Optional upper bound for price filtering (e.g., 5000 to remove outliers)
        """
        df = df.copy()
        # Strip currency symbols and commas, convert to float
        df['price'] = (
            df['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # Drop rows where price is missing or non-positive
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]
        
        # Optional: filter upper bound for outlier removal
        if max_price is not None:
            df = df[df['price'] <= max_price]
        
        return df

    def process(self, max_price: float = None) -> pd.DataFrame:
        """
        Executes the universal data pipeline.
        Returns the clean, master DataFrame ready for any downstream modeling.
        
        Args:
            max_price: Optional upper bound to filter out extreme outliers
        """
        raw_df = self._load_snapshots()
        clean_df = self._clean_price(raw_df, max_price=max_price)
        
        # Return the master "signature object"
        return clean_df

    def preprocess_tabular(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Preprocesses tabular features for all downstream models (trees, MLPs, text/image branches).
        - Fills missing values in numeric columns (bathrooms, bedrooms) with median from train
        - Encodes categorical columns (room_type, neighbourhood_cleansed) with LabelEncoder
        - Handles unseen val/test categories by mapping to special code (-1)
        - Scales all numeric features with StandardScaler (fit on train only)
        
        Fit process:
        - Fit encoders on TRAIN set only
        - Apply fitted encoders to VAL and TEST sets (no leakage)
        - Unseen categories in val/test map to -1
        
        Returns: (train_preprocessed, val_preprocessed, test_preprocessed, encoders_scalers_dict)
        """
        train = train_df.copy()
        val = val_df.copy()
        test = test_df.copy()
        
        # Dictionary to store all encoders and scalers for reproducibility
        encoders_scalers = {}
        
        # Define preprocessing columns
        categorical_cols = [
            'room_type',
            'neighbourhood_cleansed',
            'property_type',
            'instant_bookable',
        ]
        numeric_scale_cols = [
            'accommodates',
            'bathrooms',
            'bedrooms',
            'minimum_nights',
            'season_ordinal',
            'beds',
            'host_total_listings_count',
            'latitude',
            'longitude',
            'availability_365',
            'number_of_reviews',
        ]

        # Coerce numerics to numeric dtype (many raw CSVs provide strings)
        for col in numeric_scale_cols:
            train[col] = pd.to_numeric(train[col], errors='coerce')
            val[col] = pd.to_numeric(val[col], errors='coerce')
            test[col] = pd.to_numeric(test[col], errors='coerce')

        # Normalize categorical missing values for safe encoding
        for col in categorical_cols:
            train[col] = train[col].astype('object').fillna('Unknown')
            val[col] = val[col].astype('object').fillna('Unknown')
            test[col] = test[col].astype('object').fillna('Unknown')

        # 1. FILL MISSING VALUES in numeric columns (fit on train, apply to all)
        # For robustness, impute every numeric feature used downstream.
        for col in numeric_scale_cols:
            median_val = train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            train[col] = train[col].fillna(median_val)
            val[col] = val[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
            encoders_scalers[f'{col}_median'] = median_val
        
        # 2. ENCODE CATEGORICAL COLUMNS with LabelEncoder (fit on train only)
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on train set ONLY
            le.fit(train[col])
            # Transform train
            train[col] = le.transform(train[col])
            
            # Transform val: map unseen to -1
            val_encoded = []
            for val_item in val[col]:
                if val_item in le.classes_:
                    val_encoded.append(le.transform([val_item])[0])
                else:
                    # Unseen category → special code
                    val_encoded.append(-1)
            val[col] = val_encoded
            
            # Transform test: map unseen to -1
            test_encoded = []
            for test_item in test[col]:
                if test_item in le.classes_:
                    test_encoded.append(le.transform([test_item])[0])
                else:
                    # Unseen category → special code
                    test_encoded.append(-1)
            test[col] = test_encoded
            
            encoders_scalers[f'{col}_encoder'] = le
        
        # 3. SCALE NUMERIC FEATURES with StandardScaler (fit on train only)
        scaler = StandardScaler()
        scaler.fit(train[numeric_scale_cols])
        train[numeric_scale_cols] = scaler.transform(train[numeric_scale_cols])
        if len(val) > 0:
            val[numeric_scale_cols] = scaler.transform(val[numeric_scale_cols])
        if len(test) > 0:
            test[numeric_scale_cols] = scaler.transform(test[numeric_scale_cols])
        encoders_scalers['numeric_scaler'] = scaler
        encoders_scalers['numeric_scale_cols'] = numeric_scale_cols
        
        return train, val, test, encoders_scalers

    def _suffix(self, file_tag: Optional[str]) -> str:
        return "" if not file_tag else f"_{file_tag}"

    def _parquet_path(self, base_name: str, file_tag: Optional[str]) -> Path:
        return self.output_dir / f"{base_name}{self._suffix(file_tag)}.parquet"

    def _joblib_path(self, base_name: str, file_tag: Optional[str]) -> Path:
        return self.output_dir / f"{base_name}{self._suffix(file_tag)}.joblib"

    def split_and_export(self, max_price: float = None, file_tag: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads, cleans, splits into train/val/test (80/10/10), applies Box-Cox to price (fit on train only), 
        and exports as Parquet files. Also preprocesses tabular features and exports preprocessed parquets.
        
        Split strategy (two-stage):
        1. Split master into train (80%) and remaining (20%)
        2. Split remaining into val (50% of 20% = 10%) and test (50% of 20% = 10%)
        
        Result: train=80%, val=10%, test=10%
        
        Args:
            max_price: Optional upper bound to filter out extreme outliers
            file_tag: Optional suffix for exported artifacts (e.g., 'cleaned' -> train_cleaned.parquet)
        
        Returns (train_df, val_df, test_df).
        """
        master_df = self.process(max_price=max_price)
        
        # Split by unique listing id to prevent leakage across time snapshots.
        # A listing appears in multiple monthly snapshots; all rows for a given id must stay in the same split.
        unique_ids = pd.Series(master_df['id'].dropna().unique())

        # Stage 1: Split IDs into train (80%) and remaining (20%) deterministically
        train_ids = unique_ids.sample(frac=self.TRAIN_SPLIT, random_state=self.RANDOM_STATE)
        remaining_ids = unique_ids[~unique_ids.isin(train_ids)]

        # Stage 2: Split remaining IDs into val (50% of 20% = 10%) and test (50% of 20% = 10%)
        val_ids = remaining_ids.sample(frac=0.5, random_state=self.RANDOM_STATE)
        test_ids = remaining_ids[~remaining_ids.isin(val_ids)]

        train_df = master_df[master_df['id'].isin(set(train_ids))]
        val_df = master_df[master_df['id'].isin(set(val_ids))]
        test_df = master_df[master_df['id'].isin(set(test_ids))]

        # Verify split proportions
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(master_df), f"Split rows ({total}) != master ({len(master_df)})"
        assert len(train_df) + len(val_df) + len(test_df) == len(master_df), "Data lost in split"

        # Fit Box-Cox transformer on train only; add price_bc to all splits
        pt = PowerTransformer(method='box-cox', standardize=False)
        train_prices = train_df['price'].values.reshape(-1, 1)
        pt.fit(train_prices)
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['price_bc'] = pt.transform(train_prices).ravel()
        if len(val_df) > 0:
            val_df['price_bc'] = pt.transform(val_df['price'].values.reshape(-1, 1)).ravel()
        else:
            val_df['price_bc'] = np.array([], dtype=float)

        if len(test_df) > 0:
            test_df['price_bc'] = pt.transform(test_df['price'].values.reshape(-1, 1)).ravel()
        else:
            test_df['price_bc'] = np.array([], dtype=float)
        self._price_transformer = pt

        # Export raw parquets (compressed with gzip)
        train_path = self._parquet_path("train", file_tag)
        val_path = self._parquet_path("val", file_tag)
        test_path = self._parquet_path("test", file_tag)
        
        train_df.to_parquet(train_path, compression='gzip', index=False)
        val_df.to_parquet(val_path, compression='gzip', index=False)
        test_df.to_parquet(test_path, compression='gzip', index=False)
        
        # Persist the fitted price transformer for reproducibility
        transformer_path = self._joblib_path('price_transformer', file_tag)
        joblib.dump(self._price_transformer, transformer_path)
        tag_label = file_tag or "default"
        print(f"✅ Persisted price transformer ({tag_label}): {transformer_path}")
        print(f"✅ Train set exported ({tag_label}): {train_path} ({len(train_df)} rows, {100*len(train_df)/total:.1f}%)")
        print(f"✅ Val set exported ({tag_label}): {val_path} ({len(val_df)} rows, {100*len(val_df)/total:.1f}%)")
        print(f"✅ Test set exported ({tag_label}): {test_path} ({len(test_df)} rows, {100*len(test_df)/total:.1f}%)")

        # PREPROCESS TABULAR FEATURES for all downstream models (fit on train only, apply to all)
        print("\nPreprocessing tabular features (fill NaNs, encode categoricals, scale numerics)...")
        print("  Fit encoders/scalers on TRAIN set only")
        train_tabular, val_tabular, test_tabular, encoders_scalers = self.preprocess_tabular(
            train_df, val_df, test_df
        )
        
        # Export preprocessed tabular parquets
        train_tabular_path = self._parquet_path("train_tabular", file_tag)
        val_tabular_path = self._parquet_path("val_tabular", file_tag)
        test_tabular_path = self._parquet_path("test_tabular", file_tag)
        
        train_tabular.to_parquet(train_tabular_path, compression='gzip', index=False)
        val_tabular.to_parquet(val_tabular_path, compression='gzip', index=False)
        test_tabular.to_parquet(test_tabular_path, compression='gzip', index=False)
        
        # Persist encoders and scalers for reproducibility
        encoders_path = self._joblib_path('tabular_encoders', file_tag)
        joblib.dump(encoders_scalers, encoders_path)
        print(f"✅ Persisted tabular encoders & scalers ({tag_label}): {encoders_path}")
        print(f"✅ Train tabular exported ({tag_label}): {train_tabular_path} ({len(train_tabular)} rows)")
        print(f"✅ Val tabular exported ({tag_label}): {val_tabular_path} ({len(val_tabular)} rows)")
        print(f"✅ Test tabular exported ({tag_label}): {test_tabular_path} ({len(test_tabular)} rows)")
        
        print(f"\n📊 Preprocessing summary:")
        print(f"   - Filled NaNs in bathrooms, bedrooms with median (from train only)")
        print(f"   - Encoded room_type and neighbourhood_cleansed with LabelEncoder (fit on train only)")
        print(f"   - Scaled numeric features with StandardScaler (fit on train only)")
        print(f"   - Unseen val/test categories mapped to -1 (no data leakage)")
        print(f"\n✅ Train-Validation-Test split complete (80/10/10)")

        return train_df, val_df, test_df

    def split_and_export_both(self, cleaned_max_price: float = 5000, cleaned_tag: str = "cleaned") -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Exports BOTH the default dataset and a cleaned dataset (price filtered) in one run.

        - Default artifacts keep the existing filenames (backward compatible).
        - Cleaned artifacts are exported with a suffix, e.g. train_cleaned.parquet.

        Returns: ((train, val, test) default, (train, val, test) cleaned)
        """
        default_splits = self.split_and_export(max_price=None, file_tag=None)
        cleaned_splits = self.split_and_export(max_price=cleaned_max_price, file_tag=cleaned_tag)
        return default_splits, cleaned_splits


if __name__ == "__main__":
    # Smoke test to ensure it runs independently
    project_root = Path(__file__).resolve().parents[1]
    
    print("Initializing Data Processor...")
    processor = AirbnbDataProcessor(data_dir=project_root)
    
    print("Processing and splitting master dataset (default + cleaned)...")
    (train_df, val_df, test_df), (train_clean, val_clean, test_clean) = processor.split_and_export_both(
        cleaned_max_price=5000,
        cleaned_tag="cleaned",
    )
    
    print(f"\n✅ Success! Train/val/test split + tabular preprocessing complete.")
    print(f"   Train: {len(train_df)} records (80%)")
    print(f"   Val:   {len(val_df)} records (10%)")
    print(f"   Test:  {len(test_df)} records (10%)")
    print(f"\n✅ Cleaned variant (price <= $5000) exported.")
    print(f"   Train: {len(train_clean)} records")
    print(f"   Val:   {len(val_clean)} records")
    print(f"   Test:  {len(test_clean)} records")
    print(f"   Parquet files saved to: {processor.output_dir}")
    print(f"\nGenerated files:")
    print(f"   - train.parquet, val.parquet, test.parquet (raw: for text/image branches)")
    print(f"   - train_tabular.parquet, val_tabular.parquet, test_tabular.parquet (preprocessed: for all models)")
    print(f"   - price_transformer.joblib")
    print(f"   - tabular_encoders.joblib")
    print(f"   - train_cleaned.parquet, val_cleaned.parquet, test_cleaned.parquet")
    print(f"   - train_tabular_cleaned.parquet, val_tabular_cleaned.parquet, test_tabular_cleaned.parquet")
    print(f"   - price_transformer_cleaned.joblib")
    print(f"   - tabular_encoders_cleaned.joblib")
    print(f"\nAvailable columns in raw parquets:")
    print(f"   {train_df.columns.tolist()}")
