import pandas as pd
from pathlib import Path
from typing import List

class AirbnbDataProcessor:
    """
    A unified data processor that loads raw Airbnb monthly snapshots,
    applies universal cleaning rules, and returns a single 'Master' DataFrame.
    This ensures all downstream models (tabular, text, image, multi-modal) 
    are evaluated on the exact same universe of data.
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

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.snapshot_files = [
            "listings-03-25.csv",
            "listings-06-25.csv",
            "listings-09-25.csv"
        ]

    def _load_snapshots(self) -> pd.DataFrame:
        """Loads and concatenates the monthly snapshot CSVs."""
        dataframes = []
        for file_name in self.snapshot_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Expected data file not found: {file_path}")
            
            # Extract month (03, 06, 09) from filename to maintain temporal context
            month = file_name.split("-")[1]
            
            df = pd.read_csv(file_path, usecols=lambda c: c in self.REQUIRED_COLUMNS)
            df['snapshot_month'] = month
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


if __name__ == "__main__":
    # Smoke test to ensure it runs independently
    project_root = Path(__file__).resolve().parents[1]
    
    print("Initializing Data Processor...")
    processor = AirbnbDataProcessor(data_dir=project_root)
    
    print("Processing master dataset...")
    master_df = processor.process()
    
    print(f"Success! Master dataset generated with {len(master_df)} records.")
    print("Available columns for models:", master_df.columns.tolist())
