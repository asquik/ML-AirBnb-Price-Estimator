# Model Training Specification
## Project: Multi-Modal Airbnb Price Predictor (Montreal)

**WARNING TO AI AGENTS:** This document is the absolute source of truth for all model training, experiment tracking, and artifact output in this project. Read it in full before writing or modifying any training script. The Data Architecture Manual describes what data exists and how it was built. This document describes how to consume it and how to record results.

---

## 0. The Cardinal Rules

These rules are non-negotiable. Violating any one of them corrupts the experiment ledger and makes the final notebook comparison impossible.

1. **Pre-computation is mandatory, on-the-fly is forbidden.** Text embeddings (DistilBERT) and image embeddings (CLIP) must be extracted and saved to disk as `.npy` files before any model training script is run. Training scripts load static arrays from disk. They never instantiate a backbone encoder.
2. **All metrics are reported in raw Canadian dollars.** A model trained on `price_bc` must inverse-transform its predictions before computing RMSE, MAE, and R². This makes every row in `master_runs_log.csv` directly comparable regardless of training target.
3. **`sample_weight` is always applied.** Every model that supports sample weighting must use the `sample_weight` column from the tabular parquet. For sklearn models: pass as `sample_weight=` in `.fit()`. For PyTorch models: multiply the per-sample loss by the sample weight before taking the batch mean. Never average raw loss values.
4. **No training script touches test data until the final evaluation call.** Hyperparameter search and early stopping use val only. The test set is evaluated exactly once per run, at the end.
5. **Every completed run appends exactly one row to `outputs/master_runs_log.csv`.** If a script crashes before finishing, it appends nothing. Partial rows are never written.
6. **Skipping runs is explicitly allowed.** The full matrix of ~40 runs is aspirational. If compute time runs out, any subset of runs is valid as long as the runs that do complete are logged correctly. The design never assumes all runs exist.
8. **In-Memory Dataset Preloading is mandatory.** Because the host machines have ample RAM (32GB+), all data—including `.npy` embeddings, tabular features, and raw images/tokens for LoRA—MUST be fully loaded into system RAM during the PyTorch `Dataset` initialization (`__init__`). The `__getitem__` method must only slice and return tensors that already reside in memory. Performing disk I/O inside `__getitem__` during the training loop is strictly forbidden to prevent GPU starvation.
7. **DataLoader I/O optimization is mandatory.** All PyTorch `DataLoader` instantiations must expose `num_workers` as a configurable argument (tracked in `config.json`) and must enforce `pin_memory=True` whenever training on a GPU. This is critical to prevent the CPU/disk pipeline from starving the GPU. Additionally, **all pre-computed embedding arrays and tabular data must be loaded into RAM in full before the training loop begins** — do not read from disk inside `__getitem__`. The dataset is small enough that full in-memory storage is always feasible on the available hardware (both machines use SSD, but disk I/O was observed to be the primary GPU starvation bottleneck during earlier experiments). Concretely: load the `.npy` arrays once with `np.load(..., mmap_mode=None)` and store the result as a class attribute on the `Dataset`. Use `mmap_mode='r'` only as a last resort for debugging, never in production training runs.

---

## 1. Data Contract

### 1.1 Source Files

All training scripts load from `/data/`. The Data Architecture Manual describes what these files contain. Do not re-derive or re-clean anything.

**Tabular parquets (for tree models and tabular MLP):**
- Normal universe: `train_tabular.parquet`, `val_tabular.parquet`, `test_tabular.parquet`
- Cleaned universe: `train_cleaned_tabular.parquet`, `val_cleaned_tabular.parquet`, `test_cleaned_tabular.parquet`

**Raw parquets (for text extraction scripts only — not for model training):**
- `train.parquet`, `val.parquet`, `test.parquet`
- `train_cleaned.parquet`, `val_cleaned.parquet`, `test_cleaned.parquet`

**Preprocessing artifacts (in `/data/`, not `/data/artifacts/` — artifacts are in the data root):**
- `price_transformer.joblib` / `price_transformer_cleaned.joblib`
- `tabular_encoders.joblib` / `tabular_encoders_cleaned.joblib`
- `numeric_scaler.joblib` / `numeric_scaler_cleaned.joblib`
- `room_type_weights.joblib` / `room_type_weights_cleaned.joblib`

**Pre-computed embeddings (in `/data/embeddings/` — created by extraction scripts, never by training scripts):**
- `train_text_normal.npy`, `val_text_normal.npy`, `test_text_normal.npy` — DistilBERT embeddings, shape `(N, 768)`, aligned to normal universe tabular parquet by `listing_id`
- `train_text_cleaned.npy`, `val_text_cleaned.npy`, `test_text_cleaned.npy` — same, aligned to cleaned universe
- `train_image_normal_224.npy`, `val_image_normal_224.npy`, `test_image_normal_224.npy` — CLIP embeddings at 224px, shape `(N, 512)`, listing-level (averaged over all valid images per listing)
- `train_image_cleaned_224.npy`, `val_image_cleaned_224.npy`, `test_image_cleaned_224.npy` — same, cleaned universe
- `train_image_normal_336.npy`, `val_image_normal_336.npy`, `test_image_normal_336.npy` — CLIP embeddings at 336px, shape `(N, 768)` (ViT-L/14), normal universe
- `train_image_cleaned_336.npy`, `val_image_cleaned_336.npy`, `test_image_cleaned_336.npy` — same, cleaned universe
- Companion ID files: `train_text_normal_ids.npy`, etc. — the `listing_id` array in the same row order as the embedding matrix

> **Note on 336px files:** These are generated only if VRAM allows running ViT-L/14. If they do not exist, image models default to 224px (ViT-B/32). The model registry (Section 5) notes which resolution each run targets.

> **Alignment guarantee:** The extraction scripts guarantee that row `i` in `train_text_normal.npy` corresponds to row `i` in `train_tabular.parquet`. The companion `_ids.npy` file holds the `listing_id` for each row and must be verified at load time. If the IDs do not match the tabular parquet IDs exactly, abort with an error — do not silently proceed with misaligned data.

### 1.2 Feature Column Definitions

These are fixed. Do not change them inside a model script. If an experiment requires different features, create a new script.

**Tabular features for tree models and tabular MLP (16 columns, all pre-encoded and pre-scaled):**
```
room_type, neighbourhood_cleansed, property_type, instant_bookable,
accommodates, bathrooms, bedrooms, beds, host_total_listings_count,
latitude, longitude, minimum_nights, availability_365, number_of_reviews,
season_ordinal, has_valid_image
```

**Tabular features for fusion models (same 16 columns, concatenated with embeddings):**
Same as above. `full_text` and `is_french` are NOT included as tabular columns — `full_text` is consumed by the extraction script to produce the text embedding; `is_french` is available as an optional additional tabular feature for ablation but is not included in the default feature set.

**Targets:**
- `price` — raw Canadian dollars. Use with Huber loss (SmoothL1) for DL; use directly for tree/sklearn models.
- `price_bc` — Box-Cox transformed. Use with MSE loss for DL. Requires inverse transform for metric reporting.

### 1.3 The Universe × Target Matrix

Every model run lives in exactly one cell of this matrix:

| | `normal` universe | `cleaned` universe |
|---|---|---|
| `price` target | variant: `normal_raw` | variant: `cleaned_raw` |
| `price_bc` target | variant: `normal_bc` | variant: `cleaned_bc` |

The four variant identifiers (`normal_raw`, `normal_bc`, `cleaned_raw`, `cleaned_bc`) are the canonical strings used in CLI args, folder names, and CSV rows throughout the entire project.

---

## 2. The CLI Contract

Every training script must accept these command-line arguments. No exceptions.

```
--variant     {normal_raw, normal_bc, cleaned_raw, cleaned_bc}   (required)
--run-name    <string>   optional human label appended to the run folder name
                         e.g. "depth15_minleaf5" or "head_deep256"
```

Fusion and image model scripts additionally accept:
```
--image-size  {224, 336}   default: 224
```

**How the script uses `--variant`:**
- `normal_*` → load `train_tabular.parquet`, `price_transformer.joblib`, `room_type_weights.joblib`, embeddings with `_normal` suffix
- `cleaned_*` → load `train_cleaned_tabular.parquet`, `price_transformer_cleaned.joblib`, `room_type_weights_cleaned.joblib`, embeddings with `_cleaned` suffix
- `*_raw` → use `price` column as target; use Huber loss for DL; no inverse transform needed for reporting
- `*_bc` → use `price_bc` column as target; use MSE loss for DL; load the appropriate `price_transformer` and apply `inverse_transform` before computing all metrics

Every script also accepts:
```
--smoke-test    optional flag. If passed, triggers rapid debugging mode.
```

**Smoke Test Behavior:** If `--smoke-test` is passed, the script must perform rapid debugging:
1. Truncate all loaded data splits (train, val, test) to exactly 100 rows immediately after loading, before any other processing.
2. Force the model to train for a maximum of 1 epoch (or a very small sweep — e.g., 2 hyperparameter combinations — for tree/sklearn models).
3. Pass `is_smoke_test=True` into the `ExperimentTracker` initialization.

**Why this flag exists:** As models grow more complex (DL, LoRA), running a full training pass to verify a script compiles and runs end-to-end is unreasonable. The smoke test flag lets the script author verify correctness — no import errors, no shape mismatches, no missing files — in seconds rather than hours. It is the mandatory verification step before handing a script over for a real training run. Smoke tests are never used to evaluate model quality; a smoke test result tells you nothing about the model's actual performance.

**Example invocations:**
```bash
python scripts/models/lightgbm_model.py --variant cleaned_raw --run-name "leaves128_lr001"
python scripts/models/fusion_mlp.py --variant normal_bc --image-size 336 --run-name "head_deep256"
python scripts/models/text_lora.py --variant normal_bc --run-name "lora_rank16"

# Smoke test — verify a new script runs end-to-end without errors, no meaningful output:
python scripts/models/fusion_lora.py --variant normal_bc --smoke-test
```

---

## 3. Experiment Tracking Contract

### 3.1 Directory Structure

```
outputs/
├── master_runs_log.csv              ← single source of truth; append only
└── runs/
    └── {run_id}_{ModelName}/
        ├── config.json              ← exact hyperparameters and environment
        ├── history.json             ← per-epoch train/val loss (DL only)
        ├── predictions.npz          ← y_true, y_pred_raw (all splits)
        ├── feature_importance.json  ← tree models only
        └── best_model.pth           ← PyTorch weights (DL only)
```

`run_id` format: `run_YYYYMMDD_HHMM`. The full folder name is `run_YYYYMMDD_HHMM_{ModelName}` where `ModelName` matches the `model_type` field in the CSV (e.g., `run_20260427_1430_LightGBM`).

### 3.2 `master_runs_log.csv` — Column Definitions

Every script appends exactly one row when it successfully completes. The CSV is created on first write if it does not exist. Columns:

| Column | Type | Description |
|---|---|---|
| `run_id` | string | e.g. `run_20260427_1430` |
| `model_type` | string | e.g. `DecisionTree`, `RandomForest`, `LightGBM`, `GradientBoosting`, `Ridge`, `PolynomialRidge`, `TabularMLP`, `TextMLP`, `ImageMLP`, `FusionMLP`, `TextLoRA`, `ImageLoRA`, `FusionLoRA` |
| `modalities` | string | `tabular`, `tab+text`, `tab+image`, `tab+text+image` |
| `dataset_variant` | string | one of the four canonical variant strings |
| `image_size` | int or null | `224`, `336`, or null for non-image models |
| `fusion_head` | string or null | architecture label, e.g. `shallow_64`, `deep_256`, null for tree/sklearn models |
| `lora_applied_to` | string or null | `text`, `image`, `text+image`, null for non-LoRA runs |
| `lora_rank` | int or null | e.g. `8`, `16`, null for non-LoRA runs |
| `trainable_parameters` | int | count of parameters updated during training (0 for frozen-backbone runs means fusion head only) |
| `train_rmse_raw` | float | RMSE on training set, raw dollars |
| `val_rmse_raw` | float | RMSE on validation set, raw dollars |
| `test_rmse_raw` | float | RMSE on test set, raw dollars — primary comparison metric |
| `test_mae_raw` | float | MAE on test set, raw dollars |
| `test_r2` | float | R² on test set |
| `batch_size` | int or null | batch size used during training, null for tree/sklearn models |
| `training_time_minutes` | float | wall-clock time from first `.fit()` call to final metric computation |
| `avg_time_per_epoch_sec` | float or null | average wall-clock seconds per epoch (DL models only, null for trees) |
| `peak_vram_gb` | float or null | peak VRAM via `torch.cuda.max_memory_allocated()`, null for CPU-only runs |
| `peak_ram_gb` | float or null | peak system RAM in GB via `psutil.Process().memory_info().rss`, measured at end of training |
| `best_hyperparams` | string | JSON-encoded dict of the best hyperparameters found (e.g. `{"max_depth": 15, "min_samples_leaf": 5}`) |
| `run_name` | string | value passed via `--run-name`, empty string if not provided |
| `artifact_folder` | string | relative path to the run folder, e.g. `outputs/runs/run_20260427_1430_LightGBM` |
| `notes` | string | free-text, empty by default; can be manually edited after the fact |

### 3.3 `config.json` — Required Fields

Saved inside the run folder at the start of training (not at the end, so partial runs still leave a trace):

```json
{
  "run_id": "run_20260427_1430",
  "model_type": "LightGBM",
  "modalities": "tabular",
  "dataset_variant": "cleaned_raw",
  "image_size": null,
  "fusion_head": null,
  "lora_applied_to": null,
  "lora_rank": null,
  "seed": 42,
  "target_column": "price",
  "target_transform": "none",
  "box_cox_lambda": null,
  "feature_cols": ["room_type", "..."],
  "run_name": "leaves128_lr001",
  "host_machine": "node1",
  "device_used": "cuda:0",
  "dataloader_workers": 4,
  "started_at": "2026-04-27T14:30:00"
}
```

For `*_bc` variants, `box_cox_lambda` must be populated by reading it from the loaded `price_transformer` object. This is critical: without it, predictions saved in `predictions.npz` cannot be inverse-transformed by a notebook cell that doesn't have the `.joblib` file available.

### 3.4 `history.json` — DL Models Only

```json
{
  "train_loss": [2.34, 1.87, 1.62, "..."],
  "val_loss": [2.51, 2.03, 1.78, "..."],
  "val_rmse_raw": [98.4, 85.2, 79.1, "..."]
}
```

One entry per epoch. `val_rmse_raw` is in raw dollars (inverse-transformed if `*_bc` variant) so learning curves are interpretable without further processing.

### 3.5 `predictions.npz`

Saved after final evaluation. Keys:

```python
np.savez(
    path,
    train_y_true=...,   # raw dollars, shape (N_train,)
    train_y_pred=...,   # raw dollars, shape (N_train,)
    val_y_true=...,
    val_y_pred=...,
    test_y_true=...,
    test_y_pred=...,
)
```

All values in raw Canadian dollars, regardless of training target. This enables scatter plots and residual plots in the notebook without any transformation code.

---

## 4. Per-Model-Family Rules

### 4.1 Tree and Ensemble Models (sklearn / LightGBM)

**Models:** `DecisionTree`, `RandomForest`, `GradientBoosting`, `LightGBM`

**Data:** tabular parquets only. No embeddings.

**Feature columns:** the 16-column set from Section 1.2.

**Target:** both `price` and `price_bc` are valid. For `*_bc` variants: fit on `price_bc`, inverse-transform predictions for metrics.

**`sample_weight`:** pass `sample_weight=train_df["sample_weight"].values` to `.fit()`. Required.

**Hyperparameter search:** grid search over val RMSE. Do not use cross-validation — use the provided val split. Select the best configuration, then refit on train only (not train+val) and evaluate on test.

**Loss function / training target note:** tree models do not have a "loss function" in the same sense as DL. They split on squared error by default. Using `price_bc` as the target for a Decision Tree is a valid experiment — it changes the distribution the tree splits on. The result is reported in raw dollars after inverse transform.

**No `history.json`** — tree models do not have epochs. Only `config.json`, `predictions.npz`, and `feature_importance.json`.

**`feature_importance.json` format:**
```json
{"room_type": 0.142, "neighbourhood_cleansed": 0.331, "...": "..."}
```
Normalized so values sum to 1.

### 4.2 Classical Regression Models (sklearn)

**Models:** `Ridge`, `PolynomialRidge`

**Data:** tabular parquets only.

**Feature columns:** same 16-column set. For `PolynomialRidge`, `PolynomialFeatures(degree=2)` is applied inside the script before fitting.

**Target:** both `price` and `price_bc` are valid.

**`sample_weight`:** pass to `.fit()`. Required.

**No `history.json`**, no `feature_importance.json`.

### 4.3 Tabular MLP (PyTorch)

**Model:** `TabularMLP`

**Data:** tabular parquets only. No embeddings.

**Feature columns:** same 16-column set, split into categorical (4 columns: `room_type`, `neighbourhood_cleansed`, `property_type`, `instant_bookable`) and numeric (12 columns). Categoricals go through `nn.Embedding` layers. Numerics are already scaled by the upstream processor — do not re-scale.

**Target:** `price` with Huber loss, or `price_bc` with MSE loss, per variant.

**`sample_weight`:** multiply per-sample loss by `sample_weight` before `.mean()`. Required.

**Early stopping:** monitor val loss. Stop if val loss does not improve for 10 consecutive epochs. Save the best checkpoint.

**`fusion_head` field:** for this model, refers to the MLP body itself, not a fusion component. Use the same label convention: `shallow_64` = one hidden layer of 64 units, `deep_256` = two hidden layers of 256 units.

**Saves:** `config.json`, `history.json`, `predictions.npz`, `best_model.pth`.

### 4.4 Text MLP — Frozen Backbone (PyTorch)

**Model:** `TextMLP`

**Data:** tabular parquet + pre-computed DistilBERT embeddings from `/data/embeddings/`. Training script loads the `.npy` files. It never instantiates DistilBERT.

**Input to MLP:** concatenation of text embedding (768D) + tabular features (16D) = 784D input vector.

**Target:** `price` with Huber loss, or `price_bc` with MSE loss, per variant.

**`sample_weight`:** same as TabularMLP — multiply per-sample loss by weight before mean.

**`fusion_head`:** describes the MLP head applied after concatenation. Use label convention: `shallow_64`, `medium_128`, `deep_256`, `deep_512`.

**Early stopping:** 10-epoch patience on val loss.

**Saves:** `config.json`, `history.json`, `predictions.npz`, `best_model.pth`.

### 4.5 Image MLP — Frozen Backbone (PyTorch)

**Model:** `ImageMLP`

**Data:** tabular parquet + pre-computed CLIP embeddings from `/data/embeddings/`. Training script loads `.npy` files. It never instantiates CLIP.

**Input to MLP:** concatenation of image embedding (512D for 224px / 768D for 336px) + tabular features (16D).

**Target:** `price` with Huber loss, or `price_bc` with MSE loss, per variant.

**`sample_weight`:** required, same pattern.

**`fusion_head`:** same label convention.

**`image_size`:** must match the resolution of the embedding files loaded. Recorded in CSV and `config.json`.

**Early stopping:** 10-epoch patience.

**Saves:** `config.json`, `history.json`, `predictions.npz`, `best_model.pth`.

### 4.6 Fusion MLP — Frozen Backbone (PyTorch)

**Model:** `FusionMLP`

**Data:** tabular parquet + text embeddings + image embeddings.

**Input to MLP:** concatenation of text embedding + image embedding + tabular features. Total input dim = 768 + 512 (or 768) + 16.

**Target, `sample_weight`, `fusion_head`, early stopping:** same rules as 4.4 and 4.5.

**Saves:** `config.json`, `history.json`, `predictions.npz`, `best_model.pth`.

### 4.7 Text LoRA (PyTorch + PEFT)

**Model:** `TextLoRA`

**Architecture:** DistilBERT with LoRA adapters on the transformer attention layers + tabular branch + fusion MLP head. All three components trained simultaneously.

**STRICT RULE:** Even in LoRA mode, the raw embedding input pipeline is bypassed. LoRA training works directly on raw tokenized text. This is the *only* case where a training script holds a live backbone in memory. The pre-computed `.npy` embeddings are NOT used for LoRA training — the backbone must be loaded because its weights are being updated.

**Data:** tokenized text (from `full_text` column in the tabular parquet) + tabular features. Raw parquets are used as the text source.

**Target:** `price_bc` with MSE loss preferred (LoRA on normal_bc is the primary run). Other variants are valid but secondary.

**`sample_weight`:** required.

**`lora_rank`:** hyperparameter, e.g. 8 or 16. Recorded in CSV.

**`lora_applied_to`:** `"text"` for this model.

**`trainable_parameters`:** count only the LoRA adapter parameters + tabular branch + fusion head. The frozen backbone parameter count is excluded.

**Early stopping:** 10-epoch patience.

**Saves:** `config.json`, `history.json`, `predictions.npz`, `best_model.pth` (LoRA adapter weights only, not the full backbone).

### 4.8 Image LoRA (PyTorch + PEFT)

**Model:** `ImageLoRA`

Same pattern as TextLoRA but applies LoRA to the CLIP vision encoder. Loads raw images from `/images/processed_224/` or `/images/processed_336/` at training time (this is the only other case where training touches raw images — necessary because the backbone weights are being updated).

**`lora_applied_to`:** `"image"`.

### 4.9 Full Fusion LoRA (PyTorch + PEFT)

**Model:** `FusionLoRA`

Applies LoRA to both DistilBERT and CLIP simultaneously. Everything else follows the same rules as 4.7 and 4.8 combined.

**`lora_applied_to`:** `"text+image"`.

---

## 5. Model Registry

This table is the definitive list of all planned runs. Status is updated manually as runs complete. A run can be skipped — mark it `skipped` with a note rather than leaving it blank.

**Status values:** `planned` | `running` | `done` | `skipped`

### 5.1 Tree & Classical Models

| # | Script | Model Type | Variant | Notes | Status |
|---|---|---|---|---|---|
| 1 | `decision_tree.py` | DecisionTree | normal_raw | baseline | planned |
| 2 | `decision_tree.py` | DecisionTree | cleaned_raw | — | planned |
| 3 | `decision_tree.py` | DecisionTree | normal_bc | ablation: does BC help trees? | planned |
| 4 | `decision_tree.py` | DecisionTree | cleaned_bc | — | planned |
| 5 | `random_forest.py` | RandomForest | normal_raw | — | planned |
| 6 | `random_forest.py` | RandomForest | cleaned_raw | — | planned |
| 7 | `gradient_boosting.py` | GradientBoosting | normal_raw | — | planned |
| 8 | `gradient_boosting.py` | GradientBoosting | cleaned_raw | — | planned |
| 9 | `lightgbm_model.py` | LightGBM | normal_raw | — | planned |
| 10 | `lightgbm_model.py` | LightGBM | cleaned_raw | — | planned |
| 11 | `ridge_model.py` | Ridge | normal_raw | — | planned |
| 12 | `ridge_model.py` | Ridge | cleaned_raw | — | planned |
| 13 | `polynomial_ridge.py` | PolynomialRidge | normal_raw | — | planned |
| 14 | `polynomial_ridge.py` | PolynomialRidge | cleaned_raw | — | planned |

### 5.2 Deep Learning — Tabular Only

| # | Script | Model Type | Variant | Fusion Head | Status |
|---|---|---|---|---|---|
| 15 | `tabular_mlp.py` | TabularMLP | normal_raw | shallow_64 | planned |
| 16 | `tabular_mlp.py` | TabularMLP | normal_bc | shallow_64 | planned |
| 17 | `tabular_mlp.py` | TabularMLP | cleaned_raw | shallow_64 | planned |
| 18 | `tabular_mlp.py` | TabularMLP | cleaned_bc | shallow_64 | planned |
| 19 | `tabular_mlp.py` | TabularMLP | normal_bc | deep_256 | head ablation | planned |

### 5.3 Deep Learning — Text Branch (Frozen)

| # | Script | Model Type | Variant | Fusion Head | Status |
|---|---|---|---|---|---|
| 20 | `text_mlp.py` | TextMLP | normal_bc | shallow_64 | planned |
| 21 | `text_mlp.py` | TextMLP | cleaned_bc | shallow_64 | planned |
| 22 | `text_mlp.py` | TextMLP | normal_bc | deep_256 | head ablation | planned |

### 5.4 Deep Learning — Image Branch (Frozen)

| # | Script | Model Type | Variant | Fusion Head | Image Size | Status |
|---|---|---|---|---|---|---|
| 23 | `image_mlp.py` | ImageMLP | normal_bc | shallow_64 | 224 | planned |
| 24 | `image_mlp.py` | ImageMLP | cleaned_bc | shallow_64 | 224 | planned |
| 25 | `image_mlp.py` | ImageMLP | normal_bc | deep_256 | 224 | head ablation | planned |
| 26 | `image_mlp.py` | ImageMLP | normal_bc | shallow_64 | 336 | res ablation, if VRAM allows | planned |

### 5.5 Deep Learning — Full Fusion (Frozen)

| # | Script | Model Type | Variant | Fusion Head | Image Size | Status |
|---|---|---|---|---|---|---|
| 27 | `fusion_mlp.py` | FusionMLP | normal_bc | shallow_64 | 224 | planned |
| 28 | `fusion_mlp.py` | FusionMLP | cleaned_bc | shallow_64 | 224 | planned |
| 29 | `fusion_mlp.py` | FusionMLP | normal_bc | deep_256 | 224 | head ablation | planned |
| 30 | `fusion_mlp.py` | FusionMLP | normal_bc | shallow_64 | 336 | res ablation, if VRAM allows | planned |

### 5.6 LoRA Fine-Tuning

All LoRA runs include tabular features as a third input branch. The `Modalities` column is explicit — there are no implied inputs.

| # | Script | Model Type | Variant | Fusion Head | Modalities | Image Size | Rank | Status |
|---|---|---|---|---|---|---|---|---|
| 31 | `fusion_lora.py` | FusionLoRA | normal_bc | deep_256 | tab+text+image | 224 | 16 | PRIORITY 1 |
| 32 | `fusion_lora.py` | FusionLoRA | normal_bc | deep_256 | tab+text+image | 336 | 16 | PRIORITY 2 |
| 33 | `image_lora.py` | ImageLoRA | normal_bc | deep_256 | tab+image | 336 | 16 | ablation |
| 34 | `text_lora.py` | TextLoRA | normal_bc | deep_256 | tab+text | n/a | 16 | ablation |
| 35 | `fusion_lora.py` | FusionLoRA | normal_bc | shallow_64 | tab+text+image | 336 | 16 | head ablation |
| 36 | `fusion_lora.py` | FusionLoRA | normal_bc | deep_256 | tab+text+image | 336 | 8 | rank ablation |

**Total planned runs: 36.** Execution order reversed from original plan — LoRA runs first (highest compute value), tree/sklearn baselines second (written while GPU trains). All runs are optional; skip any that exceed compute budget.

---

## 6. The `ExperimentTracker` Utility

Every training script imports and uses an `ExperimentTracker` from `scripts/experiment_tracker.py`. This utility is the only shared code across all training scripts. It handles:

1. Creating the run folder under `outputs/runs/`
2. Writing `config.json` at the start of the run
3. Appending the completed row to `outputs/master_runs_log.csv`
4. Saving `predictions.npz`
5. Saving `history.json` (DL scripts call `tracker.log_epoch(train_loss, val_loss, val_rmse_raw)`)

**Usage pattern every training script must follow:**

```python
# At the top of every training script:
tracker = ExperimentTracker(
    model_type="LightGBM",
    modalities="tabular",
    variant=args.variant,          # from CLI
    run_name=args.run_name,        # from CLI, may be empty string
    config={                       # model-specific hyperparameters
        "num_leaves": 128,
        "learning_rate": 0.01,
        # ...
    }
)

# After training and evaluation:
tracker.finish(
    train_metrics={"rmse": ..., "mae": ..., "r2": ...},
    val_metrics={"rmse": ..., "mae": ..., "r2": ...},
    test_metrics={"rmse": ..., "mae": ..., "r2": ...},
    predictions={
        "train_y_true": ..., "train_y_pred": ...,
        "val_y_true": ..., "val_y_pred": ...,
        "test_y_true": ..., "test_y_pred": ...,
    },
    trainable_parameters=5000,
    training_time_minutes=12.4,
    peak_vram_gb=None,             # or float if GPU used
    extra_artifacts={              # optional: any additional files to save
        "feature_importance.json": {...},
    }
)
```

All metrics passed to `tracker.finish()` must already be in raw Canadian dollars. The tracker performs no transformation — it trusts the training script to have done the inverse transform before calling `finish()`.

**Smoke Test Handling:** If the tracker is initialized with `is_smoke_test=True`, it must:
1. Prepend `SMOKE_` to the `run_id` and folder name (e.g., `SMOKE_run_20260427_1430_FusionLoRA`).
2. **Skip** appending the final row to `outputs/master_runs_log.csv`. The master ledger must never contain smoke test rows.
3. It may still save `config.json` and `predictions.npz` in the smoke test folder for debugging purposes — these are useful for inspecting shapes and values if something is wrong.

The smoke test folder is disposable. Do not commit it. Delete it once you have confirmed the script runs correctly.

---

## 7. Alignment Verification Protocol

Before any training script begins consuming embeddings, it must verify alignment:

```python
tab_ids = train_df["listing_id"].values
emb_ids = np.load("data/embeddings/train_text_normal_ids.npy")
assert np.array_equal(tab_ids, emb_ids), \
    f"ALIGNMENT FAILURE: tabular and embedding row order do not match"
```

This assertion must not be removed or weakened. A silent misalignment would train the model on the wrong text/image for each listing, producing subtly incorrect results with no visible error.

---

## 8. Embedding Pre-Computation Scripts

These scripts exist in `scripts/` and must be run before any DL model training. They are not described in detail in this document — see the extraction scripts themselves. This section records only the contract they must satisfy.

**Text extraction** (`scripts/extract_text_features.py`):
- Input: raw parquet files (`train.parquet`, etc.), column `full_text`
- Output: `data/embeddings/{split}_text_{universe}.npy` and `{split}_text_{universe}_ids.npy`
- Backbone: `distilbert-base-multilingual-cased`, frozen, no gradient computation

**Image extraction** (`scripts/extract_vision_features.py`):
- Input: `/images/processed_224/` or `/images/processed_336/` directories
- Output: `data/embeddings/{split}_image_{universe}_{size}.npy` and corresponding `_ids.npy`
- Backbone: CLIP (`openai/clip-vit-base-patch32` for 224px; `openai/clip-vit-large-patch14` for 336px), frozen
- Aggregation: all valid images for a listing are averaged into a single listing-level embedding. The ImageNet-mean placeholder images (for listings with no valid photo) produce a `0.0` embedding vector after normalization — this is the intended behavior per the Data Architecture Manual.
- Only the universe filter (normal vs. cleaned) determines which rows are in the output. All other filtering (e.g., `has_valid_image`) is handled at training time via the tabular column — the extraction script always produces embeddings for every row in the split.

---

**End of Document.** Future modeling agents: implement one script at a time, verify it appends a correct row to `master_runs_log.csv`, then proceed to the next.
