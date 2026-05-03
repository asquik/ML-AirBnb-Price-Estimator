# Multi‑Modal Airbnb Price Predictor — Montreal

A compact, reproducible project that predicts nightly Airbnb prices using a late‑fusion multimodal model (images + listing text + tabular features). Implementation focuses on clarity, reproducibility, and fast iteration utilizing parameter-efficient fine-tuning (LoRA).

## 🚀 Quickstart: Local Python Environment

This project is designed to run natively in a Python virtual environment. 

### 1. Environment Setup
Initialize your environment and install the required dependencies. The LoRA requirements contain the full stack for multimodal training.

```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows Git Bash/CMD
pip install -r requirements-lora.txt
```

*(Tip: You can append `--help` to any of the Python scripts below to see a full list of accepted arguments and configuration options.)*

### 2. Download Raw Images
Before processing the dataset, fetch the images from the URLs provided in the raw insideAirbnb data. The downloader supports resuming and pacing to avoid rate limits.

```bash
# Basic run (applies EDA price filter by default)
python scripts/download_images.py

# Example: Spread downloads evenly across 24 hours
python scripts/download_images.py --duration-hours 24 --min-interval 10
```

### 3. Data Processing (The "Pure Function" Pipeline)
The data processor acts as the source of truth, converting raw CSV snapshots into ready-to-train deterministic splits.

```bash
python scripts/data_processor.py
```
**Outputs:** This script generates over 10 distinct `.parquet` files in the `data/` directory, representing an 80/10/10 split (Train/Val/Test) across both the normal dataset and a `cleaned` variant (filtered for prices strictly between $50 and $1000). It also exports corresponding `_tabular.parquet` versions with pre-scaled numeric features and encoded categoricals.

### 4. Image Preprocessing
Standardize the downloaded images for the vision model. This script creates dual-resolution outputs and generates neutral, standardized placeholders for any missing or corrupt image files to ensure pipeline integrity.

```bash
python scripts/image_processor.py
```

### 5. Model Training (Ablation & Fusion)
Train models ranging from simple baselines to complex multi-modal networks. Metrics (RMSE, MAE, R²) are automatically logged to `outputs/model_runs.csv`.

**Example: Run the Late-Fusion LoRA Model (Priority 1)**
```bash
python scripts/models/fusion_lora.py \
  --variant normal_bc \
  --image-size 224 \
  --lora-rank 16 \
  --fusion-head deep_256 \
  --batch-size 16 \
  --accum-steps 2 \
  --workers 4 \
  --run-name priority1_local
```

---

## 🧠 Available Models Catalog

The `scripts/models/` directory contains isolated execution scripts for various architectures, allowing for clean ablation studies.

### Baseline & Tabular Models
*   **`decision_tree.py`**: Fast, interpretable baseline regression.
*   **`random_forest.py`**: Ensemble tree baseline.
*   **`gradient_boosting.py`**: Standard GBM regression.
*   **`lightgbm_model.py`**: High-performance gradient boosting.
*   **`ridge_model.py` / `polynomial_ridge.py`**: Linear baselines.
*   **`tabular_mlp.py`**: Deep learning baseline using only tabular features.

### Unimodal Deep Learning
*   **`text_mlp.py`**: Trains on text embeddings (DistilBERT) with frozen weights.
*   **`text_lora.py`**: Fine-tunes the text backbone using Low-Rank Adaptation.
*   **`image_mlp.py`**: Trains on image embeddings (CLIP) with frozen weights.
*   **`image_lora.py`**: Fine-tunes the vision backbone using Low-Rank Adaptation.

### Multimodal Late Fusion
*   **`fusion_mlp.py`**: Concatenates frozen text, vision, and tabular embeddings into a trainable head.
*   **`fusion_lora.py`**: The flagship model. Performs end-to-end training by applying LoRA adapters to both the DistilBERT and CLIP backbones simultaneously while training the late-fusion head.