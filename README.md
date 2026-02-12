# Multi‑Modal Airbnb Price Predictor — Montreal

A compact, reproducible project that predicts nightly Airbnb price using a late‑fusion multimodal model (images + listing text + tabular features). Implementation focuses on clarity, reproducibility and fast iteration.

## Current state
- EDA and report are available (`outputs/report.md`, `outputs/figures/`).
- A resumable image downloader is provided (`scripts/download_images.py`).
- Planned next steps: implement `Dataset` + preprocessing and the fusion `nn.Module`, then run baseline training.

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
- `Context.md` — project context and experimental plan
- `listings.csv` — raw dataset (kept unchanged)
- `scripts/eda.py` — dataset diagnostics and report generator
- `scripts/download_images.py` — resumable image downloader with pacing
- `outputs/` — EDA report, figures, and summary CSV

## Design principles
- Minimal, readable PyTorch code and standard Dataset/Dataloader patterns.
- Freeze large pre-trained backbones for compute efficiency and reproducible baselines.
- Maintain separation between data processing, dataset, model, and training logic.

## Next deliverables
- `Dataset` + preprocessing implementation (PyTorch Dataset).  
- `models.py` with the fusion `nn.Module` and a minimal training loop.  
- A reproducible notebook with results and ablation studies.

