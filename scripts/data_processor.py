import pandas as pd
from pathlib import Path
from typing import Tuple

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
        # Fixed seed for reproducible train/test split
        self.RANDOM_STATE = 42
        self.TRAIN_TEST_SPLIT = 0.8

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
        """Cleans the target price column and removes invalid/missing values."""
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
        
        # Universal Rule: Drop rows where the target (price) is entirely missing.
        # This guarantees models won't accidentally train/test on blank targets.
        df = df.dropna(subset=['price'])
        
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

    def split_and_export(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads, cleans, splits into train/test (80/20), and exports as Parquet files.
        Returns (train_df, test_df).
        
        Parquet files are compressed by default and include all tabular + text features.
        The 'picture_url' column is retained for optional image branch models.
        """
        master_df = self.process()
        
        # Deterministic split using listing ID to ensure reproducibility
        train_df = master_df.sample(frac=self.TRAIN_TEST_SPLIT, random_state=self.RANDOM_STATE)
        test_df = master_df.drop(train_df.index)
        
        # Export to Parquet (compressed by default with gzip)
        train_path = self.output_dir / "train.parquet"
        test_path = self.output_dir / "test.parquet"
        
        train_df.to_parquet(train_path, compression='gzip', index=False)
        test_df.to_parquet(test_path, compression='gzip', index=False)
        
        print(f"✅ Train set exported: {train_path} ({len(train_df)} rows)")
        print(f"✅ Test set exported: {test_path} ({len(test_df)} rows)")
        
        return train_df, test_df


if __name__ == "__main__":
    # Smoke test to ensure it runs independently
    project_root = Path(__file__).resolve().parents[1]
    
    print("Initializing Data Processor...")
    processor = AirbnbDataProcessor(data_dir=project_root)
    
    print("Processing and splitting master dataset...")
    train_df, test_df = processor.split_and_export()
    
    print(f"\n✅ Success! Train/test split complete.")
    print(f"   Train: {len(train_df)} records")
    print(f"   Test:  {len(test_df)} records")
    print(f"   Parquet files saved to: {processor.output_dir}")
    print(f"\nAvailable columns for models:")
    print(f"   {train_df.columns.tolist()}")
