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
- **Train/Validation/Test Parquets** (preprocessed tabular data)
  - Deterministic 80/10/10 split (seed=42): 80% train, 10% validation, 10% test
  - **`train_tabular.parquet`** (80%) — used to fit models and hyperparameters
  - **`val_tabular.parquet`** (10%) — used for hyperparameter tuning and early stopping during training
  - **`test_tabular.parquet`** (10%) — held-out final evaluation set (never touched during development)
  - All models evaluated fairly on the same identical test set
  - Compressed with gzip, ~5.5 MB total
  - Includes: encoded room_type, neighbourhood, numeric features scaled
  - Reproducibility: encoders/scalers persisted in `data/tabular_encoders.joblib`
- **`train.parquet`, `val.parquet`, `test.parquet`** (raw data for text/image branches)
  - Same 80/10/10 split, includes description, amenities, picture_url
  - Used by text/image models that apply their own embeddings
- **`images/all/`** downloaded locally or linked from source (optional)
  - Not strictly required for Decision Tree baseline
  - Will be uploaded to Google Drive if image branch is pursued

---

## **REVISED TIMELINE: ~3 Weeks Remaining (Weeks 3-5 as of April 2)**

**Current Status (as of April 5, 2026):** 
- ✅ EDA complete with detailed findings (report.md)
- ✅ Data processor finalized with dual parquet exports + preprocessing
- ⏳ **ACTIVE:** Refactoring to 80/10/10 train-validation-test split (seed=42)
- ✅ Tabular encoders/scalers persisted for reproducibility
- ✅ Decision Tree baseline script exists (`train_tabular_baseline.py`)
- ⏳ **Next: Update data processor for tri-split, write validation tests, then run baseline**

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

## **WEEK 2 (April 5-11): TRAIN-VAL-TEST SPLIT + DECISION TREE BASELINE** ⏳ IN PROGRESS
**Objective:** Implement proper train-validation-test split, run baseline with validation-based tuning.

**Tasks:**
- [ ] **Refactor `data_processor.py` to 80/10/10 split** (PRIORITY)
  - Update `split_and_export()` to generate 3 splits instead of 2
  - Export: `train.parquet`, `val.parquet`, `test.parquet` (raw)
  - Export: `train_tabular.parquet`, `val_tabular.parquet`, `test_tabular.parquet` (preprocessed)
  - Fit encoders/scalers on train only; apply to val and test (no data leakage)
  - Persist encoders/scalers as before
  - Write comprehensive tests for tripled data splits before implementing
- [ ] Update `scripts/train_tabular_baseline.py` to use validation set
  - Hyperparameter sweep on validation set (max_depth=[8, 12, 15, 20, 25, 30], min_samples_leaf=[5, 10, 20, 30])
  - Select best model based on validation RMSE
  - Report final metrics on held-out test set only
  - Log results (train/val/test RMSE/MAE/R²) to `outputs/model_runs.csv`
- [ ] Analyze baseline results and update `decision_log.md` with findings
- [ ] Create `prepare_submission.py`: bundle train/val/test parquets + image metadata
  - Output: package for Google Drive (parquets + images directory export)
- [ ] Upload `train_tabular.parquet`, `val_tabular.parquet`, `test_tabular.parquet`, `images/` to Google Drive
  - Get shareable link for notebook download step
  - Store link in README or a separate `GOOGLE_DRIVE_LINKS.md`
- [ ] Commit: `"feat: train-val-test split implemented, baseline experiments logged"`

**Deliverable:** Baseline results logged (train/val/test metrics); Google Drive ready with data + images

**Decision point:** Does baseline RMSE on test set indicate good predictive signal? (proceed if RMSE < ±$200; else revisit feature engineering)

---

## **WEEK 3 (April 12-18): TEXT & IMAGE BRANCH EXPERIMENTS** ⏳ NOT STARTED
**Objective:** Measure marginal contribution of text and image modalities using validation set for tuning.

### **Option A: Text Branch (if pursuing)**
- [ ] Create `scripts/text_model.py`: embed description+amenities with DistilBERT (frozen)
  - MLP fusion head: BERT output (768D) + tabular features (7D) → hidden layer → price
  - Train on train_tabular.parquet; tune hyperparameters on val_tabular.parquet
  - Report final metrics on held-out test_tabular.parquet
- [ ] Log results (train/val/test) to `outputs/model_runs.csv`
- [ ] Compare vs. baseline on test set: if RMSE improves >2%, keep for notebook

### **Option B: Image Branch (if pursuing)**
- [ ] Create `scripts/image_model.py`: extract image features with CLIP (frozen)
  - Handle multi-image per listing (avg or attention pooling)
  - MLP fusion head: image embedding (512D) + tabular features → price
  - Train on train_tabular.parquet; tune hyperparameters on val_tabular.parquet
  - Report final metrics on held-out test_tabular.parquet
- [ ] Log results (train/val/test) to `outputs/model_runs.csv`
- [ ] Compare vs. baseline on test set: if RMSE improves >2%, keep for notebook

**Decision point for each branch:** >2% test RMSE improvement over baseline? Yes → include in notebook; No → skip for scope/time

---

## **WEEK 4 (April 19-25): BUILD SUBMISSION NOTEBOOK** 🔴 CRITICAL PATH
**Objective:** Convert local experiments into single reproducible `.ipynb`.

**Notebook architecture:**

1. **Cells 1-3: Setup & Data Loading**
   - Markdown: Introduction to problem (Airbnb price prediction, Montreal)
   - Code: Install/import libraries (pandas, scikit-learn, torch, transformers, etc.)
   - Code: Authenticate Google Drive and download `train_tabular.parquet`, `val_tabular.parquet`, `test_tabular.parquet`
   - Display: train/val/test shapes and quick data preview

2. **Cells 4-6: EDA & Feature Overview**
   - Markdown: Explain dataset composition, temporal structure, outliers, train-val-test split rationale
   - Code: Load parquets, show distributions (price, min_nights, room_type, etc.)
   - Visualizations: price histogram, neighbourhood breakdown, seasonality

3. **Cells 7-11: Decision Tree Baseline**
   - Markdown: Why Decision Tree? (interpretable, fast, fair baseline)
   - Code: Hyperparameter sweep on validation set; final evaluation on test set
   - Results table: train/val/test RMSE, MAE, R² for best model
   - Feature importance plot
   - Sample predictions vs. actuals (top/bottom 5 on test set)

4. **Cells 12-16: (CONDITIONAL) Text Branch**
   - **Only include if local experiments showed >2% RMSE improvement on test set**
   - Markdown: Why text signal matters (description quality, amenities)
   - Code: Embed description+amenities with DistilBERT, train MLP fusion (tune on val, eval on test)
   - Results: train/val/test comparison vs. baseline, RMSE delta on test set
   - Analysis: What aspects of text drive price?

5. **Cells 17-21: (CONDITIONAL) Image Branch**
   - **Only include if local experiments showed >2% RMSE improvement on test set**
   - Markdown: Why images matter (curb appeal, visual quality)
   - Code: Download images from Drive, extract CLIP features, train MLP fusion (tune on val, eval on test)
   - Results: train/val/test comparison vs. baseline, RMSE delta on test set
   - Visualizations: sample images + predicted price impact

6. **Cells 22-26: Summary & Comparison**
   - Markdown: Recap of findings
   - Code: Side-by-side comparison table (all models evaluated on same test set)
   - Key insights: which modality contributes most? seasonal patterns? neighborhood effects?
   - Limitations: data gaps (no true peak summer), single city, frozen backbones

7. **Final Cell: Reproducibility Notes**
   - Markdown: Hyperlinks to GITHUB/Google Drive; explain train-val-test split strategy
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

## **WEEK 5 (April 26-May 2): QA & SUBMISSION** 🔴 FINAL PUSH
**Objective:** Final validation and submission with proper train-val-test workflow.

**Tasks:**
- [ ] Test notebook on **fresh Python environment** (simulate grader setup)
  - Create new venv, install only requirements.txt, run notebook
  - Verify: all cells execute, no FileNotFoundError, reproducible results
  - Confirm: validation set used only for tuning, test set never touched during development
- [ ] Check: visualizations render cleanly, no truncated output
- [ ] Add final polish to markdown (grammar, clarity, narrative flow)
- [ ] Verify Google Drive link is shareable and accessible (contains train/val/test splits)
- [ ] Final commit: `"docs: final submission notebook (QA complete, train-val-test workflow)"`
- [ ] **Submit notebook + Google Drive link**

---

## **FILES SUMMARY**

| Week | File | Status | Purpose |
|------|------|--------|---------|
| 1 | `scripts/data_processor.py` | ✅→⏳ | Preprocessing + train/val/test split |
| 1 | `data/train_tabular.parquet` | ⏳ | Training data (80%, ~16,973 rows) |
| 1 | `data/val_tabular.parquet` | ⏳ | Validation data (10%, ~2,621 rows) |
| 1 | `data/test_tabular.parquet` | ⏳ | Test data (10%, ~2,622 rows) |
| 1 | `data/tabular_encoders.joblib` | ✅ | Encoders/scalers for reproducibility |
| 1 | `scripts/train_tabular_baseline.py` | ✅→⏳ | Decision Tree trainer (with val tuning) |
| 2 | `prepare_submission.py` | ⏳ | Bundle data for Google Drive |
| 2 | Google Drive (external) | ⏳ | Cloud storage for parquets + images |
| 3 | `scripts/text_model.py` | ⏳ Conditional | Text branch (if >2% improvement) |
| 3 | `scripts/image_model.py` | ⏳ Conditional | Image branch (if >2% improvement) |
| 4 | `submission_notebook.ipynb` | ⏳ | **FINAL DELIVERABLE** |
| Decision Log | `decision_log.md` | ✅→⏳ | Ongoing decision tracking |

---

## **SUBMISSION CHECKLIST** 

Before submitting:
- [ ] Notebook downloads train/val/test data from Google Drive (no local files needed by grader)
- [ ] Notebook runs end-to-end without errors on fresh Python environment
- [ ] All models trained and compared fairly (identical 80/10/10 splits)
- [ ] Hyperparameters tuned on validation set; final metrics reported on held-out test set only
- [ ] Visualizations are clear and informative
- [ ] Markdown explains journey: train-val-test strategy → experiment → findings
- [ ] Limitations acknowledged (data gaps, single city, frozen backbones)
- [ ] No external .py imports (all code inline in notebook)
- [ ] No hardcoded paths (uses `pathlib`)
- [ ] Google Drive links verified and shareable (exposes all 3 splits)
- [ ] File is .ipynb format, well-structured
- [ ] **Reproducibility guaranteed: seed=42, documented 80/10/10 split, encoder persistence, test set never touched during development**  

## REVISED TIMELINE: PHASE 2 (The Multi-Modal Ablation Study)

### **WEEK 3 (Current): Feature Extraction, Language, & Baselines**
**Hardware target:** 4GB GPU Server (Xeon W3680)

* [ ] **Data Prep:** Resize all images to 224x224 to reduce storage footprint.
* [ ] **Imbalance Hybrid Script:** Implement the mild physical oversampling (2x minority classes) and compute class weights for the PyTorch loss function.
* [ ] **Feature Extraction (Pass 1):** Run `distilbert-base-multilingual-cased` and CLIP over the dataset. Save output vectors as `.npy` or `.pt`.
* [ ] **Language Engineering:** Generate the `is_french` tabular column using a language detection script. Run standard English BERT over the text to save the English-only embeddings.
* [ ] **Fusion Training:** Train the Late Fusion MLP head on the extracted vectors. Compare: (Tabular) vs. (Tab + Text) vs. (Tab + Image) vs. (All).

### **WEEK 4: Parameter-Efficient Fine Tuning (PEFT) & Escalation**
**Hardware target:** 6GB GPU Laptop (Primary) & Google Colab (Fallback)

* [ ] **LoRA Implementation:** Use Hugging Face `peft` to inject LoRA adapters into the Text branch. Train on the 6GB laptop GPU. Compare RMSE against the frozen Feature Extraction baseline.
* [ ] **LoRA Implementation (Image):** Apply LoRA to the CLIP vision encoder. Compare against the frozen baseline.
* [ ] **The Full Fine-Tune Test:** Attempt unfreezing all layers on a smaller text model (e.g., `distilbert`) on the 6GB GPU to establish the "maximum compute" accuracy ceiling.
* [ ] **(Optional/Stretch) Colab Escalation:** If time permits, push the notebook to Google Colab and attempt simultaneous LoRA fine-tuning on both image and text branches.

### **WEEK 5: Final Submission Notebook Compilation**
* [ ] Consolidate the findings into the final Jupyter Notebook.
* [ ] Structure the narrative: Explain the hybrid imbalance strategy, compare the language handling techniques, and present the Feature Extraction vs. LoRA vs. Fine-Tuning matrix.
* [ ] QA the notebook in a fresh virtual environment to guarantee reproducibility for the grader.
