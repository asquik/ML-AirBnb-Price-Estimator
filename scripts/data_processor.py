import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict

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
    
    # We load only the features identified during EDA that are relevant 
    # to either our tabular, text, or image modalities.
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

    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the target price column and removes invalid/missing values. Keeps only raw price."""
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
        return df

    def process(self) -> pd.DataFrame:
        """
        Executes the universal data pipeline.
        Returns the clean, master DataFrame ready for any downstream modeling.
        """
        raw_df = self._load_snapshots()
        clean_df = self._clean_price(raw_df)
        
        # Return the master "signature object"
        return clean_df

    def preprocess_tabular(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
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
        
        # 1. FILL MISSING VALUES in numeric columns (fit on train, apply to all)
        numeric_fill_cols = ['bathrooms', 'bedrooms']
        for col in numeric_fill_cols:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            val[col] = val[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
            encoders_scalers[f'{col}_median'] = median_val
        
        # 2. ENCODE CATEGORICAL COLUMNS with LabelEncoder (fit on train only)
        categorical_cols = ['room_type', 'neighbourhood_cleansed']
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
        numeric_scale_cols = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
        scaler = StandardScaler()
        scaler.fit(train[numeric_scale_cols])
        train[numeric_scale_cols] = scaler.transform(train[numeric_scale_cols])
        val[numeric_scale_cols] = scaler.transform(val[numeric_scale_cols])
        test[numeric_scale_cols] = scaler.transform(test[numeric_scale_cols])
        encoders_scalers['numeric_scaler'] = scaler
        encoders_scalers['numeric_scale_cols'] = numeric_scale_cols
        
        return train, val, test, encoders_scalers

    def split_and_export(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads, cleans, splits into train/val/test (80/10/10), applies Box-Cox to price (fit on train only), 
        and exports as Parquet files. Also preprocesses tabular features and exports preprocessed parquets.
        
        Split strategy (two-stage):
        1. Split master into train (80%) and remaining (20%)
        2. Split remaining into val (50% of 20% = 10%) and test (50% of 20% = 10%)
        
        Result: train=80%, val=10%, test=10%
        
        Returns (train_df, val_df, test_df).
        """
        master_df = self.process()
        
        # Stage 1: Split into train (80%) and remaining (20%) using deterministic seed
        train_df = master_df.sample(frac=self.TRAIN_SPLIT, random_state=self.RANDOM_STATE)
        remaining_df = master_df.drop(train_df.index)
        
        # Stage 2: Split remaining 20% into val (50% of 20% = 10%) and test (50% of 20% = 10%)
        val_df = remaining_df.sample(frac=0.5, random_state=self.RANDOM_STATE)
        test_df = remaining_df.drop(val_df.index)

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
        val_df['price_bc'] = pt.transform(val_df['price'].values.reshape(-1, 1)).ravel()
        test_df['price_bc'] = pt.transform(test_df['price'].values.reshape(-1, 1)).ravel()
        self._price_transformer = pt

        # Export raw parquets (compressed with gzip)
        train_path = self.output_dir / "train.parquet"
        val_path = self.output_dir / "val.parquet"
        test_path = self.output_dir / "test.parquet"
        
        train_df.to_parquet(train_path, compression='gzip', index=False)
        val_df.to_parquet(val_path, compression='gzip', index=False)
        test_df.to_parquet(test_path, compression='gzip', index=False)
        
        # Persist the fitted price transformer for reproducibility
        transformer_path = self.output_dir / 'price_transformer.joblib'
        joblib.dump(self._price_transformer, transformer_path)
        print(f"✅ Persisted price transformer: {transformer_path}")
        print(f"✅ Train set exported: {train_path} ({len(train_df)} rows, {100*len(train_df)/total:.1f}%)")
        print(f"✅ Val set exported: {val_path} ({len(val_df)} rows, {100*len(val_df)/total:.1f}%)")
        print(f"✅ Test set exported: {test_path} ({len(test_df)} rows, {100*len(test_df)/total:.1f}%)")

        # PREPROCESS TABULAR FEATURES for all downstream models (fit on train only, apply to all)
        print("\nPreprocessing tabular features (fill NaNs, encode categoricals, scale numerics)...")
        print("  Fit encoders/scalers on TRAIN set only")
        train_tabular, val_tabular, test_tabular, encoders_scalers = self.preprocess_tabular(
            train_df, val_df, test_df
        )
        
        # Export preprocessed tabular parquets
        train_tabular_path = self.output_dir / "train_tabular.parquet"
        val_tabular_path = self.output_dir / "val_tabular.parquet"
        test_tabular_path = self.output_dir / "test_tabular.parquet"
        
        train_tabular.to_parquet(train_tabular_path, compression='gzip', index=False)
        val_tabular.to_parquet(val_tabular_path, compression='gzip', index=False)
        test_tabular.to_parquet(test_tabular_path, compression='gzip', index=False)
        
        # Persist encoders and scalers for reproducibility
        encoders_path = self.output_dir / 'tabular_encoders.joblib'
        joblib.dump(encoders_scalers, encoders_path)
        print(f"✅ Persisted tabular encoders & scalers: {encoders_path}")
        print(f"✅ Train tabular exported: {train_tabular_path} ({len(train_tabular)} rows)")
        print(f"✅ Val tabular exported: {val_tabular_path} ({len(val_tabular)} rows)")
        print(f"✅ Test tabular exported: {test_tabular_path} ({len(test_tabular)} rows)")
        
        print(f"\n📊 Preprocessing summary:")
        print(f"   - Filled NaNs in bathrooms, bedrooms with median (from train only)")
        print(f"   - Encoded room_type and neighbourhood_cleansed with LabelEncoder (fit on train only)")
        print(f"   - Scaled numeric features with StandardScaler (fit on train only)")
        print(f"   - Unseen val/test categories mapped to -1 (no data leakage)")
        print(f"\n✅ Train-Validation-Test split complete (80/10/10)")

        return train_df, val_df, test_df


if __name__ == "__main__":
    # Smoke test to ensure it runs independently
    project_root = Path(__file__).resolve().parents[1]
    
    print("Initializing Data Processor...")
    processor = AirbnbDataProcessor(data_dir=project_root)
    
    print("Processing and splitting master dataset...")
    train_df, val_df, test_df = processor.split_and_export()
    
    print(f"\n✅ Success! Train/val/test split + tabular preprocessing complete.")
    print(f"   Train: {len(train_df)} records (80%)")
    print(f"   Val:   {len(val_df)} records (10%)")
    print(f"   Test:  {len(test_df)} records (10%)")
    print(f"   Parquet files saved to: {processor.output_dir}")
    print(f"\nGenerated files:")
    print(f"   - train.parquet, val.parquet, test.parquet (raw: for text/image branches)")
    print(f"   - train_tabular.parquet, val_tabular.parquet, test_tabular.parquet (preprocessed: for all models)")
    print(f"   - price_transformer.joblib")
    print(f"   - tabular_encoders.joblib")
    print(f"\nAvailable columns in raw parquets:")
    print(f"   {train_df.columns.tolist()}")
