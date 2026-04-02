# Multi‑Modal Airbnb Price Predictor — Montreal

A compact, reproducible project that predicts nightly Airbnb price using a late‑fusion multimodal model (images + listing text + tabular features). Implementation focuses on clarity, reproducibility and fast iteration.

## Current state
- EDA and report are available (`outputs/report.md`, `outputs/figures/`).
- A resumable image downloader is provided (`scripts/download_images.py`).
- Decision tracking log initialized at `decision_log.md`.
- Data processor established with comprehensive unit tests (`scripts/data_processor.py`, `tests/test_data_processor.py`).
- Experiment workflow and timeline documented in `WORKFLOW.md`.
- **Next step:** Build Decision Tree regression baseline as first model (Week 1-2 of timeline), using deterministic train/test split from parquets.

## Motivation
- Estimate the marginal value contributed by listing images and textual quality beyond standard tabular features (the "curb appeal" effect).
- Apply transfer learning with frozen vision+language backbones and a lightweight fusion head for regression.

## Quickstart
1. Create a Python environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
2. Reproduce EDA outputs:
   ```bash
   .venv\Scripts\python scripts/eda.py
   ```
3. Inspect the committed report at `outputs/report.md`.

4. Generate deterministic train/test split (parquet format):
   ```bash
   python3 scripts/data_processor.py
   ```
   This creates `data/train.parquet` (80%) and `data/test.parquet` (20%).

5. Train the simple tabular Decision Tree baseline:
  ```bash
  python3 scripts/train_tabular_baseline.py
  ```

## Decision log and model-run workflow
- Record every meaningful preprocessing/modeling choice in `decision_log.md`.
- Each baseline run appends metrics and configuration to `outputs/model_runs.csv`.
- Start simple (decision tree), then compare alternative model compositions using the same run log.

## Downloader (resume‑safe & paced)
- Basic run (resume-aware, applies EDA price filter by default):
  ```bash
  .venv\Scripts\python scripts/download_images.py
  ```
- Spread downloads evenly across a duration (e.g. 24 hours):
  ```bash
  .venv\Scripts\python scripts/download_images.py --duration-hours 24 --min-interval 10
  ```
- Resize downloaded images to 224×224 using `--resize 224`.

## Repository layout (key files)
- `WORKFLOW.md` — detailed experiment timeline and submission plan  
- `Context.md` — project context and experimental plan
- `decision_log.md` — log of all modeling decisions and results
- `scripts/data_processor.py` — unified data pipeline (loads CSVs → outputs clean master DataFrame)
- `scripts/download_images.py` — resumable image downloader with pacing
- `scripts/train_tabular_baseline.py` — Decision Tree baseline trainer with run logging
- `tests/test_data_processor.py` — unit tests for data processor
- `outputs/` — EDA report, figures, model runs CSV

## Design principles
- Minimal, readable PyTorch code and standard Dataset/Dataloader patterns.
- Freeze large pre-trained backbones for compute efficiency and reproducible baselines.
- Maintain separation between data processing, dataset, model, and training logic.

## Next deliverables
- **Week 1-2:** Decision Tree regression baseline with hyperparameter sweep (see `WORKFLOW.md` for timeline)  
- **Weeks 3-4:** Optional text branch (BERT embeddings)  
- **Weeks 5-6:** Optional image branch (CLIP embeddings)  
- **Week 8:** Compile all experiments into single submission notebook with Google Drive data source

