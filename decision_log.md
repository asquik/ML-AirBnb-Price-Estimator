# Decision Log

A lightweight journal for all data and modeling decisions.

## How to use
- Add a new entry whenever you change preprocessing, filtering, features, split strategy, or evaluation.
- Keep entries short and concrete.
- Link the change to scripts/files and (if applicable) experiment outputs.

## Entry template

```md
## YYYY-MM-DD — Short decision title
- **Context:**
- **Decision:**
- **Reasoning:**
- **Impacted files:**
- **Validation:**
- **Next action:**
```

## Entries

## 2026-04-02 — Persist Box–Cox transformed target in data processor exports
- **Context:** Price is heavily right-skewed with extreme high outliers. Modeling on raw price makes training unstable and allows outliers to dominate MSE. We want a minimal, reversible stabilization applied at data-export time so all downstream experiments use the same transformed label.
- **Decision:** Add a `price_bc` column to the master DataFrame exported by `scripts/data_processor.py` (computed via `sklearn.preprocessing.PowerTransformer(method='box-cox')`) and include it in `train.parquet` and `test.parquet`. Also persist the fitted transformer object alongside the exported parquets. Fit transformer only on training set to avoid data leakage.
- **Reasoning:** Box–Cox (fitted via `PowerTransformer`) finds a power transform that can better normalize skewed, strictly-positive data. Persisting the transformer ensures reproducible inverse transforms for reporting RMSE in dollars. Fitting only on train set avoids leakage.
- **Impacted files:** `scripts/data_processor.py` (computes and persists `price_bc` and the transformer), `data/train.parquet`, `data/test.parquet`, `reports/report.md` (noted change in inspection section).
- **Validation:** Run `scripts/data_processor.py` → exported `train.parquet` and `test.parquet` contain `price_bc`, and a transformer file (e.g., `price_transformer.joblib`) exists in the output folder. Confirm `price_bc` equals the transform produced by the persisted transformer for sample rows.
- **Next action:** Train models on `price_bc` (and report RMSE on dollars after inverse-transforming predictions). Document performance differences between raw- and Box–Cox-target training in experiment logs.
## 2026-03-21 — Refactor data processor: deterministic train/test split, parquet export
- **Context:** For fair model comparison and cloud submission, all models must evaluate on the same held-out test set. Previously, data processor only loaded/cleaned data; train/test splitting was deferred to individual model scripts (inconsistent).
- **Decision:** Refactor `data_processor.py` to include deterministic 80/20 train/test split (seed=42) and export both splits as compressed parquet files in `data/` directory.
- **Reasoning:** 
  - Ensures all models (Decision Tree, text, image, fusion) evaluate on identical test set → fair comparison.
  - Parquet format: compressed (gzip), mixed dtypes, cloud-friendly, fast I/O.
  - Single source of truth: parquets uploaded to Drive; notebooks load pre-split data → no re-splitting bugs.
  - Lightweight: train.parquet (20,973 rows) + test.parquet (5,243 rows) can be quickly downloaded for model training.
- **Impacted files:** `scripts/data_processor.py` (added `split_and_export()` method), `tests/test_data_processor.py` (added 3 new unit tests), `WORKFLOW.md` (updated Week 1 tasks + notebook sections).
- **Validation:** All 8 unit tests pass. Data processor generates train.parquet (20,973 rows, 80%) + test.parquet (5,243 rows, 20%) with seed=42. Split is deterministic and reproducible.
- **Next action:** Build `train_tabular_baseline.py` using parquet files as input.

## 2026-03-20 — Finalize experiment workflow & submission plan
- **Context:** Need a clear roadmap from local experiments to final notebook submission, emphasizing Decision Tree as first model.
- **Decision:** Created `WORKFLOW.md` with 8-week timeline: Decision Tree baseline (Weeks 1-2), text branch (Weeks 3-4), image branch (Weeks 5-6), fusion (Week 6-7), submission packaging (Week 8).
- **Reasoning:** Structured iteration keeps experiments organized and results logged. Decision Tree goes first (professor's suggestion, simplicity). Each branch is optional, allowing flexible scope. Final notebook is pure narrative: problem → Decision Tree → (text) → (images) → comparison → conclusions.
- **Impacted files:** `WORKFLOW.md` (new), `README.md` (updated), `decision_log.md`.
- **Validation:** Workflow includes clear deliverables, decision log integration points, and submission checklist.
- **Next action:** Implement Decision Tree regression baseline trainer (Week 1 task).

## 2026-03-20 — Remove raw month, keep only season_ordinal
- **Context:** Raw month values (03, 06, 09) can mislead models into false ordinal correlation (e.g., month 11-12 = highest price).
- **Decision:** Remove `snapshot_month` entirely. Keep only `season_ordinal` (1, 2, 3) representing Winter (Oct-Apr), Spring (Apr-Jun), Summer (Jun-Oct).
- **Reasoning:** Semantic seasonal encoding prevents spurious correlations and is transparent to downstream models.
- **Impacted files:** `scripts/data_processor.py`, `tests/test_data_processor.py`.
- **Validation:** All 5 unit tests pass. Data processor returns only REQUIRED_COLUMNS + season_ordinal.
- **Next action:** Update EDA report to document seasonal encoding decision.

## 2026-03-18 — Baseline modeling strategy reset
- **Context:** Priority shifted to fast iteration on simple models before multimodal fusion.
- **Decision:** Start with a tabular-only Decision Tree baseline and log every run outcome.
- **Reasoning:** Simplest interpretable model reduces setup friction and creates a clear performance anchor.
- **Impacted files:** `scripts/train_tabular_baseline.py`, `outputs/model_runs.csv`, `README.md`.
- **Validation:** Baseline script prints RMSE/MAE/R2 and appends each run to the run log CSV.
- **Next action:** Compare multiple tree configurations (`max_depth`, `min_samples_leaf`) and keep best baseline.

## 2026-03-18 — Start formal decision tracking
- **Context:** Iterative data work is expected (frequent tweaks and reversals).
- **Decision:** Introduce a dedicated decision log at `decision_log.md`.
- **Reasoning:** Keeps a durable record of why changes were made, not just what changed in code.
- **Impacted files:** `decision_log.md`, `README.md`.
- **Validation:** Decision log template and first workflow entry created.
- **Next action:** Add a new log entry for each preprocessing or feature-engineering change.
