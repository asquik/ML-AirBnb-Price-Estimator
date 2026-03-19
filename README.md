# Multi‑Modal Airbnb Price Predictor — Montreal

A compact, reproducible project that predicts nightly Airbnb price using a late‑fusion multimodal model (images + listing text + tabular features). Implementation focuses on clarity, reproducibility and fast iteration.

## Current state
- EDA and report are available (`outputs/report.md`, `outputs/figures/`).
- A resumable image downloader is provided (`scripts/download_images.py`).
- Decision tracking log initialized at `logs/decision_log.md`.
- Simple tabular baseline trainer implemented at `scripts/train_tabular_baseline.py`.
- Planned next steps: iterate on tabular baselines, then add text/image branches and compare gains.

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

4. Train the simple tabular Decision Tree baseline:
  ```bash
  python3 scripts/train_tabular_baseline.py
  ```

## Decision log and model-run workflow
- Record every meaningful preprocessing/modeling choice in `logs/decision_log.md`.
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
- Compare multiple tabular baselines (Decision Tree configs first).  
- Add text/image branches and run ablations against tabular baseline.  
- Produce a reproducible notebook with final comparisons.

