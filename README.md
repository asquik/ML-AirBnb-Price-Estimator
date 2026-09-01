# Montreal Airbnb Price Estimator

A web app that predicts nightly rental prices for Montreal Airbnb listings. Enter a listing's details and get an instant estimate, powered by a LightGBM model trained on 29,000+ real listings across three seasonal snapshots.

---

## Quick Start — Web App

```bash
pip install -r requirements-app.txt

# Point to a trained model (run the trainer first — see Model Training below)
export MODEL_PATH=outputs/runs/<run_id>_LightGBM/model.joblib
export DATA_DIR=data

uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

---

## API

### `POST /predict`

Accepts listing attributes, returns a price estimate.

**Request body** (JSON):

| Field | Type | Notes |
|---|---|---|
| `room_type` | `"Entire home/apt"` \| `"Private room"` \| `"Hotel room"` \| `"Shared room"` | required |
| `neighbourhood` | `str` | must be a Montreal neighbourhood |
| `season` | `"Winter"` \| `"Spring"` \| `"Summer"` | required |
| `accommodates` | `int` 1–16 | |
| `bedrooms` | `int` 0–10 | |
| `bathrooms` | `float` 0–10 | |
| `beds` | `int` 0–20 | |
| `latitude` | `float` 45.4–45.7 | Montreal bbox |
| `longitude` | `float` -73.95 to -73.47 | Montreal bbox |
| `description` | `str` max 2000 chars | optional — used for amenity keyword extraction |
| `has_valid_image` | `bool` | default `true` |
| `instant_bookable` | `bool` | default `false` |
| `minimum_nights` | `int` 1–365 | default `2` |
| `availability_365` | `int` 0–365 | default `180` |
| `number_of_reviews` | `int` 0–5000 | default `0` |
| `host_total_listings_count` | `int` 1–500 | default `1` |

**Response**:

```json
{
  "predicted_price_cad": 142.0,
  "price_range": { "low": 0.0, "high": 286.0 },
  "confidence_note": "Based on 3 amenity keywords detected. Typical error ±$144 CAD (test RMSE).",
  "keywords_detected": ["kw_wifi", "kw_metro", "kw_kitchen"],
  "warnings": [],
  "model_version": "lgbm_v1"
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "version": "lgbm_v1" }
```

### `GET /neighbourhoods`

Returns the list of valid neighbourhood strings for the dropdown.

---

## Architecture

```
POST /predict
    │
    ▼
ListingInput (Pydantic validation)
    │
    ▼
AirbnbPredictor.predict()
    ├── encode categoricals     (tabular_encoders.joblib)
    ├── impute + scale numerics (numeric_imputer.joblib, numeric_scaler.joblib)
    ├── extract 20 keyword      (keyword_features.joblib + regex on description)
    │   binary features
    └── 36-feature vector → LightGBM → Box-Cox inverse → CAD price
    │
    ▼
check_prediction() — range guards, unknown neighbourhood warning
    │
    ▼
PredictionOutput
```

**Artifacts loaded at startup** (all produced by `scripts/data_processor.py`):

| File | Contents |
|---|---|
| `data/tabular_encoders.joblib` | `Dict[col → {value: int}]` — categorical label mappings |
| `data/numeric_imputer.joblib` | `Dict[col → median]` — train-set medians for imputation |
| `data/numeric_scaler.joblib` | `StandardScaler` fitted on train split |
| `data/keyword_features.joblib` | `Dict[feature → [keyword_variants]]` — regex keyword map |
| `data/price_transformer.joblib` | Box-Cox transformer for target inverse transform |
| `outputs/runs/.../model.joblib` | Trained LightGBM model |

---

## Benchmark Results

Ablation study across 14+ configurations. All metrics on the held-out test set (10%, never touched during training or hyperparameter search).

| Model | Test RMSE | Test R² | Modalities |
|---|---|---|---|
| Decision Tree (baseline) | $202 | 0.31 | tabular |
| LightGBM — 7 features | $169 | 0.40 | tabular |
| **LightGBM — 36 features** | **$144** | **0.56** | **tabular ← served by API** |
| TabularMLP | $161 | 0.44 | tabular |
| Text MLP (DistilBERT, frozen) | $175 | 0.06 | text + tabular |
| Image MLP (CLIP, frozen) | $168 | 0.12 | image + tabular |
| Fusion MLP (CLIP + DistilBERT) | $124 | 0.10 | image + text + tabular |

LightGBM was chosen for the API because it has the best tabular R², requires no GPU, and loads in milliseconds. The fusion MLP achieves a lower RMSE but its R² is poor (0.10), suggesting it overfits to a small slice of the price distribution.

---

## R&D History

The project ran through a deliberate sequence of experiments:

1. **Decision Tree baseline** — established a performance floor and confirmed the feature pipeline works end-to-end.
2. **Tree ensemble sweep** — Random Forest and GradientBoosting, then LightGBM with a grid search over `num_leaves`, `learning_rate`, and `n_estimators`.
3. **Feature engineering** — expanded from 7 to 16 base tabular features (location coordinates, property type, host listing count, availability, reviews). Added 20 binary keyword features extracted from listing descriptions (wifi, metro, pool, etc.) via bilingual regex matching (EN/FR).
4. **Deep learning branch** — TabularMLP, then unimodal LoRA fine-tuning on DistilBERT (text) and CLIP (images), then multimodal fusion head. Run on a single A100 GPU via Docker.
5. **Model selection** — LightGBM (36 features) won on tabular metrics and was chosen for serving.

See [decision_log.md](decision_log.md) for the full reasoning behind every architecture and data decision.

---

## Model Training

> Training requires the lab environment (GPU for LoRA models, raw CSV data). The trained artifacts are not committed to this repo.

### 1. Environment

```bash
# Tabular models
pip install -r requirements.txt

# LoRA / deep learning models
pip install -r requirements-lora.txt
```

### 2. Data processing

```bash
python scripts/data_processor.py
```

Outputs parquet splits and all preprocessing artifacts to `data/`.

### 3. Download listing images (optional — for image/fusion models)

```bash
python scripts/download_images.py
python scripts/image_processor.py
```

### 4. Train LightGBM (produces the model artifact for the API)

```bash
python scripts/models/lightgbm_model.py --variant normal_bc
```

The best model is saved to `outputs/runs/<run_id>_LightGBM/model.joblib`.

### 5. Train LoRA models (GPU required)

```bash
# All LoRA runs in sequence (smoke tests + full training)
bash scripts/run_all_lora.sh

# Or individually
python scripts/models/fusion_lora.py --variant normal_bc --lora-rank 16 --fusion-head deep_256
python scripts/models/text_lora.py   --variant normal_bc --lora-rank 16 --fusion-head deep_256
python scripts/models/image_lora.py  --variant normal_bc --lora-rank 16 --fusion-head deep_256
```

### 6. Compare all runs

All training scripts append a row to `outputs/master_runs_log.csv` on completion.

---

## Project Structure

```
app/
  main.py          FastAPI app — routes, lifespan, model loading
  predictor.py     AirbnbPredictor — inference pipeline
  schemas.py       Pydantic input/output models
  checks.py        Post-prediction guards
  static/          Single-page UI (HTML/CSS/JS, no framework)
  Dockerfile       CPU serving container

scripts/
  data_processor.py          Data cleaning, splitting, artifact export
  training_utils.py          Shared constants and metric functions
  experiment_tracker.py      Run folder management, CSV logging
  models/
    decision_tree.py
    random_forest.py
    gradient_boosting.py
    ridge_model.py
    polynomial_ridge.py
    lightgbm_model.py        ← model served by the API
    tabular_mlp.py
    text_mlp.py / text_lora.py
    image_mlp.py / image_lora.py
    fusion_mlp.py / fusion_lora.py

data/                        Parquet splits + preprocessing artifacts (not committed)
outputs/                     Training run folders + master_runs_log.csv (not committed)
docker/                      GPU training containers
tests/                       Unit tests for data_processor and image_processor
```

---

## Deployment (Docker)

```bash
# Build
docker build -f app/Dockerfile -t airbnb-estimator .

# Run (mount the data and model artifacts)
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e MODEL_PATH=/app/outputs/runs/latest/model.joblib \
  airbnb-estimator
```
