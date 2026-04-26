import torch
import pandas as pd
import numpy as np
import psutil
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
import time
import gc
from numpy.lib.format import open_memmap

# ============================================================================
# SETTINGS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32 # Increased due to higher RAM (32GB)
MAX_LENGTH = 256

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3) # GB

class AirbnbTextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.listing_ids = df['id'].values
        # Keep only raw columns and tokenize lazily per item to avoid building a giant text list.
        self.descriptions = df['description'].fillna("").astype(str).values
        self.amenities = df['amenities'].fillna("").astype(str).values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.listing_ids)

    def __getitem__(self, idx):
        combined = f"{self.descriptions[idx]} [SEP] {self.amenities[idx]}"
        encoding = self.tokenizer(
            combined,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

def extract_text_features(df, split_name, model_name="distilbert-base-multilingual-cased", output_dir="data/embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[DIAGNOSTIC] Starting extraction for {split_name}")
    print(f"[DIAGNOSTIC] Initial RAM Usage: {get_mem_usage():.2f} GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    print(f"[DIAGNOSTIC] Model loaded. RAM Usage: {get_mem_usage():.2f} GB")
    
    dataset = AirbnbTextDataset(df, tokenizer)
    # Release the original DataFrame as soon as dataset arrays are built.
    del df
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == 'cuda')
    )

    model_tag = model_name.split('/')[-1]
    output_npy = os.path.join(output_dir, f"{split_name}_{model_tag}.npy")
    output_ids = os.path.join(output_dir, f"{split_name}_{model_tag}_ids.csv")

    num_rows = len(dataset)
    hidden_size = int(model.config.hidden_size)
    embeddings_mm = open_memmap(output_npy, mode='w+', dtype=np.float32, shape=(num_rows, hidden_size))
    
    model.eval()
    start_time = time.time()
    cursor = 0
    
    with torch.no_grad():
        for i, (input_ids, att_mask) in enumerate(tqdm(loader)):
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=att_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]

            batch_np = cls_emb.cpu().numpy().astype(np.float32, copy=False)
            batch_size = batch_np.shape[0]
            embeddings_mm[cursor:cursor + batch_size] = batch_np
            cursor += batch_size

            del outputs, cls_emb, input_ids, att_mask, batch_np
            
            # Periodically report diagnostics
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                it_per_sec = (i + 1) / elapsed
                print(f"\n[DIAGNOSTIC] Batch {i+1}/{len(loader)} | RAM: {get_mem_usage():.2f} GB | Speed: {it_per_sec:.2f} batch/s")
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    embeddings_mm.flush()
    pd.DataFrame({'id': dataset.listing_ids}).to_csv(output_ids, index=False)
    
    print(f"[DIAGNOSTIC] Completed {split_name}. Final RAM: {get_mem_usage():.2f} GB")
    print(f"Saved {(num_rows, hidden_size)} features to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument("--model", type=str, default="distilbert-base-multilingual-cased")
    args = parser.parse_args()
    
    df_path = f"data/{args.split}_tabular.parquet"
    if not os.path.exists(df_path):
        print(f"Error: {df_path} not found.")
    else:
        df = pd.read_parquet(df_path)
        extract_text_features(df, args.split, model_name=args.model)
