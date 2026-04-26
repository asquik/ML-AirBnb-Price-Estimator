"""
Late-Fusion Multimodal Model with LoRA Fine-Tuning
- Baseline: Frozen encoders + trainable fusion head
- LoRA: LoRA adapters on encoders + trainable fusion head
- Memory-optimized for 6GB GPU: micro-batching, gradient accumulation, mixed precision

VRAM usage tracking built-in to validate memory assumptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import (
    AutoModel, 
    AutoTokenizer,
    CLIPProcessor,
    CLIPVisionModel,
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import os
from PIL import Image
import warnings
from contextlib import nullcontext
warnings.filterwarnings('ignore')


# ============================================================================
# VRAM MONITORING UTILITIES
# ============================================================================

class VRAMMonitor:
    """Track GPU memory usage"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.peak_memory = 0
        self.memory_log = []
    
    def reset(self):
        """Clear VRAM cache and reset counters"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0
        self.memory_log = []
    
    def log(self, label=""):
        """Snapshot current memory usage"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1e9  # GB
            peak = torch.cuda.max_memory_allocated() / 1e9  # GB
            self.peak_memory = max(self.peak_memory, peak)
            self.memory_log.append({
                'label': label,
                'current_gb': round(current, 3),
                'peak_gb': round(peak, 3),
            })
            return current, peak
        return 0, 0
    
    def summary(self):
        """Print memory summary"""
        if torch.cuda.is_available():
            print("\n" + "="*70)
            print("VRAM USAGE SUMMARY")
            print("="*70)
            for entry in self.memory_log:
                print(f"{entry['label']:<40} | Current: {entry['current_gb']:>7.3f}GB | Peak: {entry['peak_gb']:>7.3f}GB")
            print(f"{'TOTAL PEAK MEMORY':<40} | {self.peak_memory:>7.3f}GB")
            print("="*70 + "\n")


# ============================================================================
# DATASET CLASS
# ============================================================================

class MultimodalAirbnbDataset(Dataset):
    """
    Load: text (from parquet description), images (from picture_url), tabular features, price
    """
    
    def __init__(self, 
                 parquet_path,
                 tokenizer,
                 image_dir=None,
                 max_text_length=128,
                 image_size=224):
        """
        Args:
            parquet_path: path to train/val/test parquet with raw data + images
            tokenizer: HuggingFace tokenizer for text
            image_dir: directory with downloaded images (or None to download on-the-fly)
            max_text_length: max tokens for text
            image_size: size to resize images to (224x224 for CLIP)
        """
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.image_index = {}
        
        # Check which tabular columns are available
        self.tabular_cols = [
            'room_type', 'neighbourhood_cleansed', 'accommodates', 'bathrooms',
            'bedrooms', 'minimum_nights', 'season_ordinal', 'beds',
            'host_total_listings_count', 'latitude', 'longitude',
            'property_type', 'instant_bookable', 'availability_365', 'number_of_reviews'
        ]
        # Filter to columns that exist in parquet
        self.tabular_cols = [c for c in self.tabular_cols if c in self.df.columns]

        # Build a fast lookup for flat image folders (e.g., processed_224/*.jpg)
        if self.image_dir and self.image_dir.exists() and self.image_dir.is_dir():
            jpg_files = list(self.image_dir.glob("*.jpg"))
            if jpg_files:
                for p in jpg_files:
                    stem = p.stem
                    # Handle both "<id>.jpg" and "<id>_...jpg"
                    key = stem.split("_")[0]
                    # Keep first occurrence for deterministic behavior
                    if key not in self.image_index:
                        self.image_index[key] = p
        
        print(f"Dataset size: {len(self.df)}")
        print(f"Tabular columns: {self.tabular_cols}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text: concatenate description + amenities + selected listing attributes.
        # This gives the text branch additional context that can influence price.
        attr_parts = [
            f"room_type={row.get('room_type', '')}",
            f"property_type={row.get('property_type', '')}",
            f"neighbourhood={row.get('neighbourhood_cleansed', '')}",
            f"accommodates={row.get('accommodates', '')}",
            f"bathrooms={row.get('bathrooms', '')}",
            f"bedrooms={row.get('bedrooms', '')}",
            f"beds={row.get('beds', '')}",
            f"instant_bookable={row.get('instant_bookable', '')}",
            f"minimum_nights={row.get('minimum_nights', '')}",
            f"availability_365={row.get('availability_365', '')}",
            f"number_of_reviews={row.get('number_of_reviews', '')}",
            f"season_ordinal={row.get('season_ordinal', '')}",
        ]
        text = (
            f"Description: {str(row.get('description', '')).strip()} "
            f"Amenities: {str(row.get('amenities', '')).strip()} "
            f"Listing attributes: {'; '.join(attr_parts)}"
        ).strip()[:1200]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Image: try flat processed folder first, then legacy per-listing folder, else fallback placeholder
        listing_id = str(row.get('id', ''))
        try:
            if self.image_dir and listing_id:
                # Fast path for processed_224 flat files
                img_path = self.image_index.get(listing_id)
                if img_path and img_path.exists():
                    image = Image.open(img_path).convert('RGB')
                else:
                    # Legacy path: images/all/<listing_id>/*.jpg
                    listing_dir = self.image_dir / listing_id
                    if listing_dir.exists() and listing_dir.is_dir():
                        candidates = sorted([
                            p for p in listing_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                        ])
                        if candidates:
                            image = Image.open(candidates[0]).convert('RGB')
                        else:
                            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
                    else:
                        image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
            else:
                image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
            # processed_224 images are already sized; resize remains safe for mixed sources
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # CHW
        except Exception as e:
            # Fallback: gray image
            image = torch.ones(3, self.image_size, self.image_size) * 0.5
        
        # Tabular features (assumed already scaled if from train_tabular.parquet)
        tabular = torch.tensor([
            float(row.get(col, 0.0)) for col in self.tabular_cols
        ], dtype=torch.float32)
        
        # Price
        price = torch.tensor(float(row['price']), dtype=torch.float32)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'image': image,
            'tabular': tabular,
            'price': price,
        }


class CachedMultimodalDataset(Dataset):
    """Dataset backed by precomputed .npy cache (mmap) for faster training."""

    def __init__(self, cache_dir, split_name):
        cache_dir = Path(cache_dir)
        self.input_ids = np.load(cache_dir / f"{split_name}_input_ids.npy", mmap_mode="r")
        self.attention_mask = np.load(cache_dir / f"{split_name}_attention_mask.npy", mmap_mode="r")
        self.images_uint8 = np.load(cache_dir / f"{split_name}_images_uint8.npy", mmap_mode="r")
        self.tabular = np.load(cache_dir / f"{split_name}_tabular.npy", mmap_mode="r")
        self.price = np.load(cache_dir / f"{split_name}_price.npy", mmap_mode="r")
        self.n = self.input_ids.shape[0]
        self.tabular_dim = int(self.tabular.shape[1])
        print(f"Cached dataset {split_name}: {self.n} samples")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Convert uint8 image to float CHW tensor in [0,1]
        image = torch.from_numpy(np.array(self.images_uint8[idx], copy=False)).float() / 255.0
        return {
            'input_ids': torch.from_numpy(np.array(self.input_ids[idx], copy=False)).long(),
            'attention_mask': torch.from_numpy(np.array(self.attention_mask[idx], copy=False)).long(),
            'image': image,
            'tabular': torch.from_numpy(np.array(self.tabular[idx], copy=False)).float(),
            'price': torch.tensor(float(self.price[idx]), dtype=torch.float32),
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LateFusionMultimodal(nn.Module):
    """
    Late-fusion architecture: frozen encoders + trainable fusion head
    Ready for LoRA adaptation via peft.
    """
    
    def __init__(self,
                 text_model="distilbert-base-multilingual-cased",
                 image_model="openai/clip-vit-base-patch32",
                 tabular_dim=15,
                 fusion_hidden=512):
        super().__init__()
        
        # Load text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model, use_safetensors=True)
        self.text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Load image encoder (CLIP Vision)
        self.image_processor = CLIPProcessor.from_pretrained(image_model)
        self.image_encoder = CLIPVisionModel.from_pretrained(image_model, use_safetensors=True)
        self.image_dim = self.image_encoder.config.hidden_size  # 768
        
        # Freeze encoders by default (will unfreeze if using LoRA)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Tabular embedding (small, always trainable)
        self.tabular_embedding = nn.Linear(tabular_dim, 128)
        self.tabular_dim_out = 128
        
        # Late fusion MLP (trainable)
        fusion_input_dim = self.text_dim + self.image_dim + self.tabular_dim_out
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids=None, attention_mask=None, images=None, tabular=None, **kwargs):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            images: (batch, 3, 224, 224)
            tabular: (batch, 15)
        
        Returns:
            prices: (batch, 1)
        """
        # PEFT wrappers may pass extra kwargs (e.g., inputs_embeds). Ignore unsupported extras.
        if input_ids is None:
            raise ValueError("input_ids is required for LateFusionMultimodal.forward")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if images is None:
            raise ValueError("images tensor is required for LateFusionMultimodal.forward")
        if tabular is None:
            raise ValueError("tabular tensor is required for LateFusionMultimodal.forward")

        # Text embedding
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embed = text_out.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Image embedding
        image_out = self.image_encoder(pixel_values=images)
        image_embed = image_out.pooler_output
        
        # Tabular embedding
        tabular_embed = self.tabular_embedding(tabular)
        
        # Concatenate and predict
        fused = torch.cat([text_embed, image_embed, tabular_embed], dim=1)
        prices = self.fusion_head(fused)
        
        return prices


# ============================================================================
# TRAINER CLASS
# ============================================================================

class LoRAMultimodalTrainer:
    """
    Training loop with:
    - Gradient accumulation
    - Mixed precision
    - LoRA adaptation
    - VRAM monitoring
    """
    
    def __init__(self, model, device, learning_rate=1e-4, accumulation_steps=4):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.accumulation_steps = accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.vram = VRAMMonitor()
        
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.results = []

    def _autocast(self):
        if self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()
    
    def apply_lora(self, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        """Apply LoRA adapters to text/image encoder branches only (keeps multimodal forward intact)."""
        text_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_lin", "v_lin"],  # DistilBERT attention
        )
        image_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            # Keep task_type unset for CLIPVisionModel so PEFT does not inject NLP kwargs
            # like `input_ids` into a vision-only forward signature.
            target_modules=["q_proj", "v_proj"],  # CLIP vision attention
        )

        self.model.text_encoder = get_peft_model(self.model.text_encoder, text_lora)
        self.model.image_encoder = get_peft_model(self.model.image_encoder, image_lora)

        # Ensure fusion/tabular heads remain trainable.
        for p in self.model.tabular_embedding.parameters():
            p.requires_grad = True
        for p in self.model.fusion_head.parameters():
            p.requires_grad = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(1, total)
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")
        print(f"\n✅ LoRA applied: rank={lora_rank}, alpha={lora_alpha}")
        self.vram.log("After LoRA init")
    
    def setup_optimizer(self):
        """Create optimizer (do this after LoRA is applied)"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
    
    def train_epoch(self, train_loader):
        """One training epoch with gradient accumulation + mixed precision"""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc="Training", leave=True)
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attn_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            tabular = batch['tabular'].to(self.device, non_blocking=True)
            prices = batch['price'].to(self.device, non_blocking=True).unsqueeze(1)
            
            # Forward pass with mixed precision
            with self._autocast():
                preds = self.model(input_ids, attn_mask, images, tabular)
                loss = self.criterion(preds, prices)
                loss = loss / self.accumulation_steps  # Normalize
            
            # Backward pass
            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.accumulation_steps
            
            # Optimizer step (accumulate every N steps)
            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader, dataset_name="Validation"):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_prices = []
        
        pbar = tqdm(val_loader, desc=f"{dataset_name}", leave=True)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attn_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            tabular = batch['tabular'].to(self.device, non_blocking=True)
            prices = batch['price'].to(self.device, non_blocking=True).unsqueeze(1)
            
            with self._autocast():
                preds = self.model(input_ids, attn_mask, images, tabular)
                loss = self.criterion(preds, prices)
            
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_prices.append(prices.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_prices = np.concatenate(all_prices)
        
        rmse = np.sqrt(np.mean((all_preds - all_prices) ** 2))
        mae = np.mean(np.abs(all_preds - all_prices))
        r2 = 1 - (np.sum((all_prices - all_preds) ** 2) / np.sum((all_prices - all_prices.mean()) ** 2))
        
        return {
            'loss': total_loss / len(val_loader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    
    def train(self, train_loader, val_loader, test_loader, epochs=3, early_stop_patience=2):
        """Full training loop"""
        self.vram.reset()
        self.vram.log("Initial state")
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.vram.log(f"After epoch {epoch+1} training")
            
            # Validate
            val_metrics = self.evaluate(val_loader, "Validation")
            
            # Test
            test_metrics = self.evaluate(test_loader, "Test")
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val RMSE: {val_metrics['rmse']:.2f} | Val R²: {val_metrics['r2']:.4f}")
            print(f"Test RMSE: {test_metrics['rmse']:.2f} | Test R²: {test_metrics['r2']:.4f}")
            
            # Early stopping
            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                print(f"✅ Best validation RMSE: {best_val_rmse:.2f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"⚠️  Early stopping triggered (patience={early_stop_patience})")
                    break
            
            # Log results
            self.results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_rmse': val_metrics['rmse'],
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
            })
        
        self.vram.summary()
        return self.results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late-fusion multimodal baseline + LoRA trainer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--lora-epochs", type=int, default=2)
    parser.add_argument("--image-dir", type=str, default="/mnt/nvme_data/linux_sys/ml_images/processed_224")
    parser.add_argument("--cache-dir", type=str, default="data/cache_multimodal")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # STEP 1: Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    
    cache_dir = Path(args.cache_dir)
    cache_ready = (
        (cache_dir / "train_input_ids.npy").exists() and
        (cache_dir / "val_input_ids.npy").exists() and
        (cache_dir / "test_input_ids.npy").exists()
    )

    if args.use_cache and cache_ready:
        print(f"Using precomputed cache from: {cache_dir}")
        train_dataset = CachedMultimodalDataset(cache_dir, "train")
        val_dataset = CachedMultimodalDataset(cache_dir, "val")
        test_dataset = CachedMultimodalDataset(cache_dir, "test")
    else:
        if args.use_cache and not cache_ready:
            print(f"Cache requested but missing files in {cache_dir}, falling back to on-the-fly processing.")
        train_dataset = MultimodalAirbnbDataset(
            "data/train_tabular.parquet",
            tokenizer=tokenizer,
            image_dir=args.image_dir
        )
        val_dataset = MultimodalAirbnbDataset(
            "data/val_tabular.parquet",
            tokenizer=tokenizer,
            image_dir=args.image_dir
        )
        test_dataset = MultimodalAirbnbDataset(
            "data/test_tabular.parquet",
            tokenizer=tokenizer,
            image_dir=args.image_dir
        )
    
    # STEP 2: Create dataloaders with micro-batching
    batch_size = args.batch_size
    accumulation_steps = max(1, args.effective_batch // max(1, batch_size))
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": args.workers,
        "pin_memory": (device.type == 'cuda'),
        "persistent_workers": (args.workers > 0),
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Gradient accumulation: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")

    if hasattr(train_dataset, "tabular_cols"):
        tabular_dim = len(train_dataset.tabular_cols)
    else:
        tabular_dim = int(getattr(train_dataset, "tabular_dim"))
    
    # STEP 3: Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    model = LateFusionMultimodal(
        text_model="distilbert-base-multilingual-cased",
        image_model="openai/clip-vit-base-patch32",
        tabular_dim=tabular_dim,
        fusion_hidden=512
    )
    
    trainer = LoRAMultimodalTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        accumulation_steps=accumulation_steps
    )
    
    # STEP 4: BASELINE (frozen encoders, trainable fusion only)
    print("\n" + "="*70)
    print("TRAINING BASELINE (FROZEN ENCODERS)")
    print("="*70)
    
    trainer.setup_optimizer()
    baseline_results = trainer.train(train_loader, val_loader, test_loader, epochs=args.baseline_epochs)
    
    # Save baseline results
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv("outputs/baseline_frozen_results.csv", index=False)
    print(f"\n✅ Baseline results saved to outputs/baseline_frozen_results.csv")
    
    # STEP 5: LORA FINE-TUNING
    print("\n" + "="*70)
    print("FINE-TUNING WITH LoRA")
    print("="*70)
    
    # Reinitialize model for LoRA
    model = LateFusionMultimodal(
        text_model="distilbert-base-multilingual-cased",
        image_model="openai/clip-vit-base-patch32",
        tabular_dim=tabular_dim,
        fusion_hidden=512
    )
    
    trainer = LoRAMultimodalTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        accumulation_steps=accumulation_steps
    )
    
    trainer.apply_lora(lora_rank=8, lora_alpha=16, lora_dropout=0.1)
    trainer.setup_optimizer()
    lora_results = trainer.train(train_loader, val_loader, test_loader, epochs=args.lora_epochs)
    
    # Save LoRA results
    lora_df = pd.DataFrame(lora_results)
    lora_df.to_csv("outputs/lora_finetuned_results.csv", index=False)
    print(f"\n✅ LoRA results saved to outputs/lora_finetuned_results.csv")
    
    # STEP 6: COMPARISON
    print("\n" + "="*70)
    print("FINAL COMPARISON: BASELINE vs. LoRA")
    print("="*70)
    
    baseline_test_rmse = baseline_df['test_rmse'].iloc[-1]
    lora_test_rmse = lora_df['test_rmse'].iloc[-1]
    improvement_pct = ((baseline_test_rmse - lora_test_rmse) / baseline_test_rmse) * 100
    
    print(f"\nBaseline Test RMSE: ${baseline_test_rmse:.2f}")
    print(f"LoRA Test RMSE:     ${lora_test_rmse:.2f}")
    print(f"Improvement:        {improvement_pct:+.2f}%")
