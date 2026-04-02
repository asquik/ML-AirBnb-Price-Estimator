# Experiment Workflow & Timeline (5 Weeks → 1 Final Notebook)

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
- **`train.parquet`** and **`test.parquet`** uploaded to Google Drive
  - Deterministic 80/20 split (seed=42)
  - All models use the same held-out test set for fair comparison
  - Compressed with gzip (lightweight)
  - Includes: tabular features, text (description + amenities), picture_url for image branch
- **`images/all/`** stays on Google Drive (optional, only downloaded for image branch)
  - Not downloaded unless image model is used
  - Referenced via listing `id` in the DataFrames

---

## **COMPRESSED TIMELINE: 5 Weeks**

### **Week 1: Decision Tree Regression Baseline (LOCAL)**
**Objective:** Establish solid baseline with fair test set for all models.

**Tasks:**
- [x] Run `scripts/data_processor.py` to generate deterministic train/test split
  - Outputs: `data/train.parquet` (20,973 rows) + `data/test.parquet` (5,243 rows)
  - Seed=42 for reproducibility
- [ ] Build `scripts/train_tabular_baseline.py` using `train.parquet` + `test.parquet`
  - Decision Tree regressor (tabular only: room_type, neighbourhood, accommodates, bathrooms, bedrooms, minimum_nights, season_ordinal)
  - Quick hyperparameter sweep: max_depth=[8, 12, 15], min_samples_leaf=[10, 20]
  - Log best RMSE/MAE/R² to `outputs/model_runs.csv`
- [ ] Create `prepare_submission.py`: bundle `train.parquet` + `test.parquet` + image metadata
  - Output: parquet files + `images/metadata.csv` (listing_id → image directory mapping)
- [ ] Upload `train.parquet`, `test.parquet`, and `images/` to Google Drive, get shareable link
- [ ] Update decision log with baseline RMSE and split details
- [ ] Commit: `"feat: deterministic train/test split + Decision Tree baseline"`

**Deliverable:** Google Drive link with data + images ready; local baseline results logged

---

### **Week 2: Text Branch (LOCAL → LOG RESULTS)**
**Objective:** Measure text contribution, prepare code for notebook.

**Tasks:**
- [ ] Create `scripts/text_model.py`: concatenate description+amenities, embed with DistilBERT, train MLP
- [ ] Train model: tabular + text embeddings
- [ ] Log results to `outputs/model_runs.csv`
- [ ] Compare vs. Decision Tree: if RMSE improves >2%, keep for notebook; else skip
- [ ] Commit: `"feat: text branch (DistilBERT), improvement=$X RMSE"`

**Decision point:** Include text in notebook or skip? (If modest improvement, you can skip.)

---

### **Week 3: Image Branch (LOCAL → LOG RESULTS)**
**Objective:** Measure image contribution, prepare code for notebook.

**Tasks:**
- [ ] Create `scripts/image_model.py`: extract image features with CLIP, train MLP
- [ ] Train model: tabular + image embeddings
- [ ] Log results to `outputs/model_runs.csv`
- [ ] Compare vs. Decision Tree: if RMSE improves >2%, keep for notebook; else skip
- [ ] Commit: `"feat: image branch (CLIP), improvement=$X RMSE"`

**Decision point:** Include images in notebook or skip? (If modest improvement, you can skip.)

---

### **Week 4: Build Submission Notebook (CRITICAL)**
**Objective:** Convert local experiments into single reproducible `.ipynb`.

**Notebook structure:**
1. **Cell 1-2:** Setup (imports, Google Drive authentication, download parquets)
   - Load `train.parquet` and `test.parquet` from Drive
   - Load image metadata if image branch is included
2. **Cells 3-5:** Problem & EDA (explain Montreal Airbnb, show data, visualizations)
3. **Cells 6-10:** Decision Tree Baseline
   - Markdown: explain why Decision Tree
   - Code: train with best hyperparameters from Week 1
   - Results: RMSE/MAE/R², feature importance plot
4. **Cells 11-15:** (If kept) Text Branch
   - Markdown: explain text signal
   - Code: train with DistilBERT (frozen backbone)
   - Results: comparison vs. baseline
5. **Cells 16-20:** (If kept) Image Branch
   - Markdown: explain curb appeal
   - Code: download images from Drive, extract CLIP features, train MLP
   - Results: comparison vs. baseline
6. **Cells 21-25:** Summary & Comparison
   - Comparison table: all models side-by-side (using same test set)
   - Key insights: which modality matters most?
   - Limitations: data gaps, future work
7. **Final cells:** Conclusions and reproducibility notes

**Critical rules:**
- ✅ All code is inline (NO imports from external .py files)
- ✅ Data loaded from Google Drive parquets (train + test already split)
- ✅ **No hardcoded paths** (all relative, uses `pathlib.Path`)
- ✅ **Fair comparison:** all models evaluate on same `test.parquet`
- ✅ Markdown + code alternating (narrative-driven, not just code dump)

**Tasks:**
- [ ] Create `submission_notebook.ipynb` from scratch OR convert existing code to notebook cells
- [ ] Copy best-performing model code from Weeks 1-3 into notebook
- [ ] Test end-to-end: runs, downloads data, trains, outputs results
- [ ] Time notebook: target <30 min total runtime
- [ ] Commit: `"feat: final submission notebook (complete, tested)"`

---

### **Week 5: Polish & Submit (FINAL PUSH)**
**Objective:** Final QA and submission.

**Tasks:**
- [ ] Test notebook on fresh Python environment (simulate grader's setup)
- [ ] Check: all visualizations render, no errors, reproducible results
- [ ] Add final markdown polish (headers, explanations, conclusions)
- [ ] Export as `.ipynb`, verify file is clean
- [ ] Create README comment with Google Drive link inside notebook (for grader)
- [ ] Final commit: `"docs: final submission notebook ready"`
- [ ] **Submit**

---

## **FILES CREATED/MODIFIED**

| Week | File | Purpose |
|------|------|---------|
| 1 | `scripts/data_processor.py` | UPDATED: now exports train/test parquets |
| 1 | `data/train.parquet` | Deterministic train set (80%, 20,973 rows) |
| 1 | `data/test.parquet` | Deterministic test set (20%, 5,243 rows) |
| 1 | `scripts/train_tabular_baseline.py` | Decision Tree trainer |
| 1 | `prepare_submission.py` | Bundle data for cloud upload |
| 1 | Google Drive link (external) | Cloud storage (parquets + images) |
| 2 | `scripts/text_model.py` (kept/discarded) | Text branch trainer |
| 3 | `scripts/image_model.py` (kept/discarded) | Image branch trainer |
| 4 | `submission_notebook.ipynb` | **FINAL DELIVERABLE** |
| 5 | (polish only) | No new files |

---

## **DECISION LOG ENTRIES (ONE PER WEEK)**

Each Friday:
1. What was built/tested
2. Key result (RMSE, improvement %, keep/skip decision)
3. Next week's goal
4. One line in decision log

---

## **SUBMISSION CHECKLIST**

Before submitting:
- [ ] Notebook downloads data from Google Drive (no local data needed)
- [ ] Notebook runs end-to-end without errors
- [ ] All models trained and compared
- [ ] Visualizations are clear
- [ ] Markdown explains the journey: hypothesis → experiment → result
- [ ] Limitations and future work discussed
- [ ] No hardcoded paths or dependencies on .py files
- [ ] README or comment in notebook has Google Drive link
- [ ] File is `.ipynb` format
- [ ] Tested on fresh Python environment  
