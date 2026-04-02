# Experiment Workflow & Timeline — Status Update (April 2, 2026)

## **OVERARCHING GOAL**
Build a **single Jupyter notebook** (`.ipynb`) that:
1. Downloads cleaned train/test data + optional images from Google Drive
2. Runs all model experiments end-to-end with consistent train/test split
3. Compares Decision Tree baseline vs. text branch vs. image branch
4. Outputs a narrative report with visualizations
5. Includes all code, markdown explanations, and results inline
6. **No external scripts or files needed**

This notebook is the ONLY submission. Everything else (data processor, image downloader, baseline scripts) is preparation.

### **Data Distribution Strategy**
- **`train_tabular.parquet`** and **`test_tabular.parquet`** (preprocessed data)
  - Deterministic 80/20 split (seed=42)
  - All models use the same held-out test set for fair comparison
  - Compressed with gzip, ~4.5 MB total
  - Includes: encoded room_type, neighbourhood, numeric features scaled
  - Reproducibility: encoders/scalers persisted in `data/tabular_encoders.joblib`
- **`train.parquet`** and **`test.parquet`** (raw data for text/image branches)
  - Same 80/20 split, includes description, amenities, picture_url
  - Used by text/image models that apply their own embeddings
- **`images/all/`** downloaded locally or linked from source (optional)
  - Not strictly required for Decision Tree baseline
  - Will be uploaded to Google Drive if image branch is pursued

---

## **REVISED TIMELINE: ~3 Weeks Remaining (Weeks 3-5 as of April 2)**

**Current Status (as of April 2, 2026):** 
- ✅ EDA complete with detailed findings (report.md)
- ✅ Data processor finalized with dual parquet exports + preprocessing
- ✅ 80/20 train/test split deterministic (seed=42)
- ✅ Tabular encoders/scalers persisted for reproducibility
- ✅ Decision Tree baseline script exists (`train_tabular_baseline.py`)
- ⏳ **Next: Run baseline experiment, then iterate on text/image branches**

---

## **WEEK 1 (March 18-21): DATA PIPELINE & BASELINE SETUP** ✅ COMPLETE
**Objective:** Establish data foundations and model baseline.

**Completed Tasks:**
- [x] Deterministic train/test split (80/20, seed=42) with parquet export
- [x] Dual preprocessing pipeline: raw parquets (for text/image) + tabular parquets (for tree/MLP)
- [x] Tabular feature encoding: LabelEncoder for categoricals, StandardScaler for numerics
- [x] Persist encoders to `data/tabular_encoders.joblib` for reproducibility
- [x] Build `scripts/train_tabular_baseline.py` with hyperparameter sweep infrastructure
- [x] Set up experiment logging to `outputs/model_runs.csv`
- [x] Update decision log with preprocessing strategy and rationale

**Status:** ✅ Data pipeline ready. Baseline script exists but hasn't been run yet.

**Outstanding (defer to Week 2):**
- [ ] Run baseline experiment and log results
- [ ] Create `prepare_submission.py` for Drive bundling
- [ ] Upload to Google Drive

---

## **WEEK 2 (April 2-8): DECISION TREE BASELINE + GOOGLE DRIVE PREP** ⏳ IN PROGRESS
**Objective:** Run baseline experiment, finalize data for cloud submission.

**Tasks:**
- [ ] Run `scripts/train_tabular_baseline.py` (baseline already coded, just execute)
  - Hyperparameter sweep: max_depth=[8, 12, 15, 20, 25, 30], min_samples_leaf=[5, 10, 20, 30]
  - Log results with timestamp to `outputs/model_runs.csv`
  - Record best RMSE/MAE/R² and best hyperparameters
- [ ] Analyze baseline results and update `decision_log.md` with findings
- [ ] Create `prepare_submission.py`: bundle train/test parquets + image metadata
  - Output: package for Google Drive (parquets + images directory export)
- [ ] Upload `train_tabular.parquet`, `test_tabular.parquet`, `images/` to Google Drive
  - Get shareable link for notebook download step
  - Store link in README or a separate `GOOGLE_DRIVE_LINKS.md`
- [ ] Commit: `"feat: baseline experiments logged, data ready for cloud"`

**Deliverable:** Baseline results logged; Google Drive ready with data + images

**Decision point:** Does baseline RMSE indicate good predictive signal? (proceed if RMSE < ±$200; else revisit feature engineering)

---

## **WEEK 3 (April 9-15): TEXT & IMAGE BRANCH EXPERIMENTS** ⏳ NOT STARTED
**Objective:** Measure marginal contribution of text and image modalities.

### **Option A: Text Branch (if pursuing)**
- [ ] Create `scripts/text_model.py`: embed description+amenities with DistilBERT (frozen)
  - MLP fusion head: BERT output (768D) + tabular features (7D) → hidden layer → price
  - Train with same train/test split for fair comparison
- [ ] Log results to `outputs/model_runs.csv`
- [ ] Compare vs. baseline: if RMSE improves >2%, keep for notebook

### **Option B: Image Branch (if pursuing)**
- [ ] Create `scripts/image_model.py`: extract image features with CLIP (frozen)
  - Handle multi-image per listing (avg or attention pooling)
  - MLP fusion head: image embedding (512D) + tabular features → price
  - Train with local or cloud-cached images
- [ ] Log results to `outputs/model_runs.csv`
- [ ] Compare vs. baseline: if RMSE improves >2%, keep for notebook

**Decision point for each branch:** >2% RMSE improvement? Yes → include in notebook; No → skip for scope/time

---

## **WEEK 4 (April 16-22): BUILD SUBMISSION NOTEBOOK** 🔴 CRITICAL PATH
**Objective:** Convert local experiments into single reproducible `.ipynb`.

**Notebook architecture:**

1. **Cells 1-3: Setup & Data Loading**
   - Markdown: Introduction to problem (Airbnb price prediction, Montreal)
   - Code: Install/import libraries (pandas, scikit-learn, torch, transformers, etc.)
   - Code: Authenticate Google Drive and download `train_tabular.parquet` + `test_tabular.parquet`
   - Display: train/test shapes and quick data preview

2. **Cells 4-6: EDA & Feature Overview**
   - Markdown: Explain dataset composition, temporal structure, outliers
   - Code: Load parquets, show distributions (price, min_nights, room_type, etc.)
   - Visualizations: price histogram, neighbourhood breakdown, seasonality

3. **Cells 7-11: Decision Tree Baseline**
   - Markdown: Why Decision Tree? (interpretable, fast, fair baseline)
   - Code: Hyperparameter sweep (same as local script)
   - Results table: RMSE, MAE, R² for best model
   - Feature importance plot
   - Sample predictions vs. actuals (top/bottom 5)

4. **Cells 12-16: (CONDITIONAL) Text Branch**
   - **Only include if local experiments showed >2% RMSE improvement**
   - Markdown: Why text signal matters (description quality, amenities)
   - Code: Embed description+amenities with DistilBERT, train MLP fusion
   - Results: Comparison vs. baseline, RMSE delta
   - Analysis: What aspects of text drive price?

5. **Cells 17-21: (CONDITIONAL) Image Branch**
   - **Only include if local experiments showed >2% RMSE improvement**
   - Markdown: Why images matter (curb appeal, visual quality)
   - Code: Download images from Drive, extract CLIP features, train MLP fusion
   - Results: Comparison vs. baseline, RMSE delta
   - Visualizations: sample images + predicted price impact

6. **Cells 22-26: Summary & Comparison**
   - Markdown: Recap of findings
   - Code: Side-by-side comparison table (all models on same test set)
   - Key insights: which modality contributes most? seasonal patterns? neighborhood effects?
   - Limitations: data gaps (no true peak summer), single city, frozen backbones

7. **Final Cell: Reproducibility Notes**
   - Markdown: Hyperlinks to GITHUB/Google Drive
   - Instructions for running notebook on fresh environment

**Critical rules:**
- ✅ All code inline (no `from scripts import ...`, copy relevant functions)
- ✅ Data loaded from Google Drive parquets (train/test already split)
- ✅ No hardcoded paths (use `pathlib.Path` + relative references)
- ✅ Fair comparison (all models evaluate on same `test_tabular.parquet`)
- ✅ Markdown + code alternating (narrative-driven)
- ✅ Runtime <30 min total (frozen backbones key for speed)

**Tasks:**
- [ ] Create fresh Jupyter notebook (`submission_notebook.ipynb`)
- [ ] Copy baseline code from `train_tabular_baseline.py` (inline)
- [ ] Add text branch code if keeping (inline)
- [ ] Add image branch code if keeping (inline)
- [ ] Test end-to-end: runs without errors, produces expected results
- [ ] Time notebook execution
- [ ] Polish markdown (headers, explanations, conclusions)
- [ ] Export as `.ipynb`, verify file structure
- [ ] Commit: `"feat: final submission notebook (complete, tested)"`

**Deliverable:** Polished, tested `.ipynb` ready for grading

---

## **WEEK 5 (April 23-29): QA & SUBMISSION** 🔴 FINAL PUSH
**Objective:** Final validation and submission.

**Tasks:**
- [ ] Test notebook on **fresh Python environment** (simulate grader setup)
  - Create new venv, install only requirements.txt, run notebook
  - Verify: all cells execute, no FileNotFoundError, reproducible results
- [ ] Check: visualizations render cleanly, no truncated output
- [ ] Add final polish to markdown (grammar, clarity, narrative flow)
- [ ] Verify Google Drive link is shareable and accessible
- [ ] Final commit: `"docs: final submission notebook (QA complete)"`
- [ ] **Submit notebook + Google Drive link**

---

## **FILES SUMMARY**

| Week | File | Status | Purpose |
|------|------|--------|---------|
| 1 | `scripts/data_processor.py` | ✅ | Preprocessing + train/test split |
| 1 | `data/train_tabular.parquet` | ✅ | Training data (80%, 20,973 rows) |
| 1 | `data/test_tabular.parquet` | ✅ | Test data (20%, 5,243 rows) |
| 1 | `data/tabular_encoders.joblib` | ✅ | Encoders/scalers for reproducibility |
| 1 | `scripts/train_tabular_baseline.py` | ✅ | Decision Tree trainer |
| 2 | `prepare_submission.py` | ⏳ | Bundle data for Google Drive |
| 2 | Google Drive (external) | ⏳ | Cloud storage for parquets + images |
| 3 | `scripts/text_model.py` | ⏳ Conditional | Text branch (if >2% improvement) |
| 3 | `scripts/image_model.py` | ⏳ Conditional | Image branch (if >2% improvement) |
| 4 | `submission_notebook.ipynb` | ⏳ | **FINAL DELIVERABLE** |
| Decision Log | `decision_log.md` | ✅ | Ongoing decision tracking |

---

## **SUBMISSION CHECKLIST** 

Before submitting:
- [ ] Notebook downloads data from Google Drive (no local files needed by grader)
- [ ] Notebook runs end-to-end without errors on fresh Python environment
- [ ] All models trained and compared fairly (same test set)
- [ ] Visualizations are clear and informative
- [ ] Markdown explains journey: hypothesis → experiment → findings
- [ ] Limitations acknowledged (data gaps, single city, frozen backbones)
- [ ] No external .py imports (all code inline in notebook)
- [ ] No hardcoded paths (uses `pathlib`)
- [ ] Google Drive links verified and shareable
- [ ] File is .ipynb format, well-structured
- [ ] Reproducibility guaranteed (seed=42, documented splits, encoder persistence)  
