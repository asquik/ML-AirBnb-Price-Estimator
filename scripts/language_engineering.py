import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import os

# Set seed for deterministic results in langdetect
DetectorFactory.seed = 42

def detect_is_french(text):
    if not text or len(str(text).strip()) < 10:
        return 0
    try:
        lang = detect(str(text))
        return 1 if lang == 'fr' else 0
    except:
        return 0

def process_splits():
    splits = ['train', 'val', 'test']
    for split in splits:
        path = f"data/{split}_tabular.parquet"
        if not os.path.exists(path):
            print(f"Skipping {split}, file not found.")
            continue
            
        print(f"Processing {split} for language detection...")
        df = pd.read_parquet(path)
        
        # We use the 'description' field for language detection
        tqdm.pandas(desc=f"Detecting language in {split}")
        df['is_french'] = df['description'].progress_apply(detect_is_french)
        
        # Save back to the same file
        df.to_parquet(path, index=False)
        print(f"Updated {split} with is_french feature. Counts: {df['is_french'].value_counts().to_dict()}")

if __name__ == "__main__":
    process_splits()
