# System Context & Data Architecture Manifesto
## Project: Multi-Modal Airbnb Price Predictor (Montreal)

**WARNING TO AI AGENTS:** This document is the absolute source of truth regarding data formats, preprocessing rules, and theoretical constraints for this project. Read thoroughly before initializing any DataLoaders, feature extractors, or neural network architectures. Do not attempt to re-clean, re-split, or alter the fundamental data structures provided in the repository.

---

## 1. The Core Relational Contract (The `listing_id`)
In this multimodal system, data is divided across tabular parquets, text embeddings, and image files.
* **The Absolute Anchor:** The `listing_id` (derived from the raw CSV `id`) is the sole primary key across all modalities.
* **No Array Assumption:** You must *never* assume that row index `i` in the tabular dataset corresponds to file `i` in a directory. All cross-modality loading (e.g., loading an image to match a tabular row) must be executed explicitly by querying the file named `{listing_id}.jpg` or the embedding keyed by `listing_id`.

---

## 2. The Dual-Universe Data Strategy (Normal vs. Cleaned)
The Montreal dataset contains massive price outliers (e.g., $5,000+/night). EDA revealed that many of these are not errors, but "Monthly Leases" (where `minimum_nights >= 30`) that are improperly recorded as nightly rates. To allow rigorous ablation studies, the data has been processed into two completely isolated universes. 

* **The Normal Universe (Default):** * Keeps 100% of the valid price data, including the massive outliers. 
  * *Constraint:* Preprocessing artifacts (scalers, encoders, Box-Cox lambdas) for this universe were fitted *including* the outliers.
* **The Cleaned Universe (Ablation/Comparison):** * Strictly truncates the dataset to remove rows where `price < 50` or `price > 1000`.
  * *Constraint:* This universe is mathematically parallel. It has its *own* separate preprocessing artifacts (scalers, lambdas) fitted strictly on the cleaned training distribution to prevent the outliers of the Normal universe from skewing the scaling of the Cleaned universe.

---

## 3. Target Variables & The Regression Loss Strategy
Because we preserved extreme right-skewed anomalies in the Normal universe, the target variable requires careful handling during deep learning. Every tabular dataset contains **two** target columns. Future modeling agents must explicitly choose which one to train on and select the appropriate loss function.

* **Target Option A: `price_bc` (Box-Cox Transformed)**
  * *What it is:* The raw price transformed into a perfectly symmetrical, normal distribution (bell curve) using a Box-Cox Power Transformation.
  * *When to use it:* For Late Fusion MLPs using standard **Mean Squared Error (MSE)** loss.
  * *Inference Rule:* Models trained on `price_bc` will output predictions in "Box-Cox space" (e.g., 2.8). You must load the persisted `price_transformer.joblib` artifact and apply `inverse_transform` to convert predictions back to raw Canadian Dollars before calculating the final RMSE/MAE metrics.
* **Target Option B: `price` (Raw Dollars)**
  * *What it is:* The unmodified nightly price in raw dollars.
  * *When to use it:* When training models using robust, outlier-resistant loss functions like **Huber Loss (Smooth L1)**. Huber loss acts as MSE for small errors but shifts to linear absolute error for extreme spikes, preventing the network weights from exploding.
  * *Inference Rule:* Models trained on this output native dollars. No inverse transformation is needed.

---

## 4. The 80/10/10 Split Isolation
The data has been deterministically split using `seed=42`.
* **Train (80%):** The *only* data used to learn scaling means, variances, vocabulary indices, class frequencies, and Box-Cox lambdas.
* **Validation (10%):** Used strictly for hyperparameter tuning and early stopping during neural network training.
* **Test (10%):** The mathematically quarantined hold-out set. Used *only* for the final ablation comparison table.

---

## 5. Tabular Feature Engineering & PyTorch Constraints
Tabular pre-processing has been hyper-optimized for immediate ingestion by PyTorch MLPs and Decision Trees.

### A. Missing Values
* All missing numeric values have been imputed using the **Train Set Median**. No `NaN` values exist in the tabular features.

### B. PyTorch-Safe Categorical Encoding
Categorical features (`room_type`, `neighbourhood_cleansed`, `property_type`, `instant_bookable`) have been integer-encoded with a strict rule to prevent `IndexOutOfBounds` crashes in PyTorch `nn.Embedding` layers:
* **`0` is strictly reserved for "Unknown/Unseen/Missing"**. If a neighborhood exists in the Test set but wasn't in the Train set, it is mapped to `0`.
* All known categories fitted on the training set start at index `1` and count upward.

### C. Numeric Scaling
Numeric features (`accommodates`, `bathrooms`, `bedrooms`, `beds`, `host_total_listings_count`, `latitude`, `longitude`, `minimum_nights`, `availability_365`, `number_of_reviews`) are standardized (Mean=0, Std=1) via `StandardScaler`.

### D. Seasonality
Raw snapshot months (03, 06, 09) create false ordinal correlations. They have been replaced with a continuous integer feature: `season_ordinal` (1 = Winter, 2 = Spring, 3 = Summer).

### E. Imbalance Handling: The `sample_weight` Column
Instead of physically duplicating rows (which ruins continuous regression distributions), class imbalance (e.g., rare "Shared rooms") is handled via PyTorch loss weighting.
* Every row includes a `sample_weight` float column.
* This weight is calculated using inverse-frequency based strictly on the `room_type` distribution in the *Training Set*. 
* *Modeling Agent Instruction:* When calculating your loss (MSE or Huber), multiply the raw error of each batch item by its `sample_weight` to softly penalize mistakes on rare listing types.

---

## 6. Text & Language Modality
The textual features of the listings have been processed to simplify extraction via models like DistilBERT.

* **Concatenation:** The raw `description` and `amenities` have been combined into a single string column: `full_text`, separated by a `[SEP]` token.
* **Bilingual Context (Montreal):** Montreal is heavily bilingual. The dataset includes a boolean column `is_french` generated via text-detection markers. 
  * *Modeling Agent Instruction:* This allows you to run ablation studies comparing a Multilingual Text Backbone against an English-Only Backbone that relies on the `is_french` tabular flag to provide context.

---

## 7. Image Modality & The Deep Learning Placeholder Trap
Images were downloaded via URLs. Because not all URLs were valid, the image dataset requires specific handling to maintain matrix dimensions during multimodal late fusion.

### A. Dual Resolutions
To allow experimentation with different Vision Transformer patch sizes, every valid listing has two mathematically cropped images:
* `images/processed_224/` : 224x224 (Standard for CLIP ViT-B/32)
* `images/processed_336/` : 336x336 (High-Res for CLIP ViT-L/14)
* *Cropping Math:* Images are resized preserving aspect ratio using Lanczos resampling, then center-cropped to a perfect square.

### B. The Missing Image Strategy (ImageNet Mean Placeholder)
If a listing had a missing or corrupt image URL, it was **not** dropped from the dataset. Instead, a placeholder image was generated.
* *The Deep Learning Math:* The placeholder is **not** pure black (`[0,0,0]`). Because PyTorch `transforms.Normalize` uses ImageNet statistics, pure black results in massive negative activations (e.g., -2.1) which the neural network will hyper-fixate on. 
* *The Solution:* The generated placeholders are a neutral gray filled with exactly the ImageNet mean RGB values (`[122, 116, 104]`). When passed through normalization, this results in a tensor of pure `0.0`s, making the missing image "mathematically invisible" to the fusion head.

### C. The `has_valid_image` Tabular Flag
Every row in the tabular dataset possesses a `has_valid_image` boolean column. 
* *Modeling Agent Instruction:* You can train on the full dataset (allowing the model to learn that `has_valid_image=False` + `0.0 tensor` = no curb appeal), OR you can filter the PyTorch Dataset dynamically by this flag to train exclusively on listings with real photos.

---

## 8. File System Manifest & Nomenclature
AI agents should expect the following file structure generated by the processors:

* **`/data/`**
  * `train.parquet`, `val.parquet`, `test.parquet` *(Raw string/numeric data, useful for Text Extractors)*
  * `train_tabular.parquet`, `val_tabular.parquet`, `test_tabular.parquet` *(Fully encoded, scaled, and weighted data for MLPs/Trees)*
  * `train_cleaned_tabular.parquet`, etc. *(The parallel Cleaned universe data)*
* **`/data/artifacts/`** (Contains the `.joblib` dictionaries/objects for `inverse_transform` and inference).
  * `price_transformer.joblib` / `price_transformer_cleaned.joblib`
  * `tabular_encoders.joblib` / `tabular_encoders_cleaned.joblib`
  * `numeric_scaler.joblib` / `numeric_scaler_cleaned.joblib`
  * `room_type_weights.joblib` / `room_type_weights_cleaned.joblib`
* **`/images/`**
  * `processed_224/{listing_id}.jpg`
  * `processed_336/{listing_id}.jpg`

**End of Context Document.** Future modeling agents: use this architectural blueprint to construct your PyTorch Datasets, Dataloaders, and Modality Fusion Heads.