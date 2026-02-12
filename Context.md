# Project Context — Multi‑Modal Airbnb Price Predictor (Montreal)

## Purpose
Build a compact, reproducible late‑fusion model that predicts nightly Airbnb price using images, listing text, and a small set of tabular features. The implementation prioritizes clarity, reproducibility and fast experimentation for academic evaluation and interviewer demos.

## Key idea
Quantify "curb appeal": the marginal value that listing images and description quality add beyond standard tabular features. Use transfer learning (frozen vision+language backbones) and a lightweight fusion head for regression.

## Architecture (high level)
- Backbone (frozen): CLIP or ViLT for image and text embeddings. Do not train from scratch.
- Tabular branch: small MLP producing a low‑dim embedding (e.g. 32D).
- Fusion head: concatenate modality embeddings → dense layers → single scalar (price). Loss: MSE or log‑MSE.

## Dataset & preprocessing (deterministic rules)
- Source: Inside Airbnb (Montreal). Keep raw `listings.csv` unchanged in repo.
- Target cleaning: strip `$`, convert `price` → float, filter 50 < price < 1000.
- Text: concatenate `description` + `amenities` (minimal cleaning).
- Images: download from `picture_url`; resize to 224×224 for model input (script supports resizing). Drop rows with unreachable images.
- Tabular features kept: `room_type`, `neighbourhood_cleansed` (top‑N or one‑hot), `accommodates`, `bathrooms`, `bedrooms` (min‑max normalize).
- Drop: reviews/host metadata to avoid leakage and scope creep.

## Evaluation & deliverables
- Metrics: RMSE / MAE on raw price; compare log‑MSE for stability. Report R² and ablation gains (image/text/tabular).
- Deliverables: training script, `models.py` (nn.Module with fusion head clearly marked), notebook with EDA and ablation tables, professor‑ready report (`outputs/report.md`).

## Current status
- EDA completed and committed (see `outputs/report.md`).
- Resumable image downloader implemented (`scripts/download_images.py`).
- Next: implement `Dataset` + preprocessing and the fusion model with a small training loop.

## Experimental plan (short)
1. Baseline: tabular MLP only.
2. Add text embedding (frozen backbone) — measure delta.
3. Add image embedding (frozen backbone) — measure delta.
4. Full fusion + simple regularization and hyperparameter sweep (LR, head width).

## Reproducibility & coding rules
- Framework: PyTorch + Hugging Face.
- Keep backbones frozen for baseline experiments.
- Use `torch.utils.data.Dataset` + deterministic transforms.
- Do not commit large binary artifacts (images are downloaded locally; EDA outputs are committed for reporting).

## Notes
- Minimal, well‑documented code demonstrates transfer learning and a clear ablation strategy.  
- Available artifacts: EDA report, downloader; planned work includes Dataset and model implementations.

## Next actions (by priority)
1. Implement `Dataset` + preprocessing and unit tests.  
2. Implement fusion `nn.Module` and a small training loop for sanity checks.  
3. Run ablation experiments and produce a final reproducible report.
