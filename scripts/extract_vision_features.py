import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse

# ============================================================================
# SETTINGS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

class AirbnbImageDataset(Dataset):
    def __init__(self, df, img_root="images/processed_224", processor=None):
        self.listing_ids = df['id'].values
        self.img_root = img_root
        self.processor = processor
        
        # Verify which listings actually have images in the processed folder
        self.valid_indices = []
        print(f"DEBUG: Checking {len(self.listing_ids)} IDs in {img_root}")
        for i, lid in enumerate(self.listing_ids):
            img_path = os.path.join(img_root, f"{lid}.jpg")
            if os.path.exists(img_path):
                self.valid_indices.append(i)
        
        if len(self.valid_indices) > 0:
            sample_idx = self.valid_indices[0]
            print(f"DEBUG: Sample match found: {self.listing_ids[sample_idx]} at {os.path.join(img_root, f'{self.listing_ids[sample_idx]}.jpg')}")
        else:
            sample_id = self.listing_ids[0]
            print(f"DEBUG: No matches. Sample ID being checked: {sample_id}, Expected Path: {os.path.join(img_root, f'{sample_id}.jpg')}")
            # Try listdir to see what's actually in there
            if os.path.exists(img_root):
                print(f"DEBUG: Directory contents (first 5): {os.listdir(img_root)[:5]}")
            else:
                print(f"DEBUG: Directory {img_root} DOES NOT EXIST")
        
        print(f"Dataset initialized: {len(self.valid_indices)} listings with processed images found out of {len(df)}.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        lid = self.listing_ids[real_idx]
        img_path = os.path.join(self.img_root, f"{lid}.jpg")
        
        image = Image.open(img_path).convert("RGB")
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            return lid, inputs['pixel_values'].squeeze(0)
        return lid, image

def extract_clip_features(df, split_name, img_root="images/processed_224", output_dir="data/embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExtracting CLIP features for {split_name}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = AirbnbImageDataset(df, img_root=img_root, processor=processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    listing_ids = []
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for lids, pixel_values in tqdm(loader):
            pixel_values = pixel_values.to(device)
            # Use the vision_model part of the CLIPModel to get the pooler_output
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            
            # The pooler_output is the [CLS] representation for the image
            features = vision_outputs.pooler_output.cpu().numpy()
            
            listing_ids.extend(lids.tolist())
            embeddings.append(features)
            
    embeddings = np.vstack(embeddings)
    
    # Save as .npy + a mapping CSV
    np.save(os.path.join(output_dir, f"{split_name}_clip_vision.npy"), embeddings)
    pd.DataFrame({'id': listing_ids}).to_csv(os.path.join(output_dir, f"{split_name}_clip_ids.csv"), index=False)
    print(f"Saved {embeddings.shape} features to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    df_path = f"data/{args.split}_tabular.parquet"
    if not os.path.exists(df_path):
        print(f"Error: {df_path} not found.")
    else:
        df = pd.read_parquet(df_path)
        extract_clip_features(df, args.split)
