# Technical Specification Document
## Multi-Modal Airbnb Price Pipeline
### Functional, Pure-Function Architecture Only

This document defines the final processing contract for the project before any Python implementation. It is written as a strict functional specification: no classes, no `self`, no hidden mutable state, and no `fit_transform` shortcuts. Every learned transformation is split into a `fit_*` function and an `apply_*` function.

---

## 1. Design Principles

1. The source of truth for every row is the InsideAirbnb `listing_id` or `id`.
2. Data processing and image processing are separate phases.
3. Data processing handles tabular + text metadata only.
4. Image processing handles image files only.
5. Feature extraction is a third, separate phase that consumes outputs from both processors.
6. All learned transforms must be fit on training data only.
7. All downstream artifacts must preserve row identity through `listing_id`.
8. No combined fit/transform functions are allowed.
9. The normal dataset is the default dataset.
10. The cleaned dataset is an explicit comparison variant.

---

## 2. Output Contract

## 2.1 Normal Variant
Default split/export behavior.

### Raw parquet outputs
- `train.parquet`
- `val.parquet`
- `test.parquet`

### Tabular parquet outputs
- `train_tabular.parquet`
- `val_tabular.parquet`
- `test_tabular.parquet`

## 2.2 Cleaned Variant
Comparison-only variant used for ablation studies.

### Cleaning rule
- Drop rows where `price < 50`
- Drop rows where `price > 1000`

### Outputs
- `train_cleaned.parquet`
- `val_cleaned.parquet`
- `test_cleaned.parquet`
- `train_cleaned_tabular.parquet`
- `val_cleaned_tabular.parquet`
- `test_cleaned_tabular.parquet`

## 2.3 Shared Required Columns in Tabular Outputs
Every tabular parquet must include:

- `listing_id`
- `price`
- `price_bc`
- `sample_weight`
- `has_valid_image`
- `is_french`
- `full_text`
- all model input features
- all encoded categorical columns
- all scaled numeric columns
- any metadata required for downstream joining

---

## 3. Data Processor Specification

## 3.1 Scope
The data processor:
- loads raw CSV snapshots
- cleans and standardizes tabular fields
- creates text features
- creates language flag
- checks image availability by file existence only
- creates train/val/test splits
- fits and applies tabular transforms
- writes raw and tabular outputs

The data processor must not:
- load image pixel data
- resize image files
- create embeddings
- run CLIP
- run BERT
- use object-oriented classes

---

## 3.2 Required Feature Logic

### Price handling
- Keep `price` as the raw numeric target.
- Also create `price_bc` as the Box-Cox transformed target.
- Keep both columns in every tabular output.

### Text handling
- Create `full_text = description + " [SEP] " + amenities`
- Create `is_french` from raw description language detection.

### Image availability
- Create `has_valid_image` by checking whether a raw downloaded image exists for the listing.
- This check must be file existence only.
- No image decoding or preprocessing is allowed in the data processor.

### Sample weighting
- Create `sample_weight` from the training-set frequency of `room_type` only.
- Use inverse-frequency weighting.
- The weight for each row is determined by its `room_type`.
- The class frequencies used for weights must come only from the training split.

A clear default formula is:

$$
sample\_weight_i = \frac{N_{train}}{K \cdot count(room\_type_i)}
$$

where:
- $N_{train}$ is the number of training rows
- $K$ is the number of unique room types in training
- $count(room\_type_i)$ is the frequency of that room type in the training set

This keeps the average weight near 1 and penalizes minority room types more heavily.

---

## 3.3 PyTorch-Safe Categorical Encoding
Categorical encoders must satisfy:

1. Unknown or unseen categories map to `0`.
2. Known categories from training start at `1`.
3. Encoded values must be integers suitable for `nn.Embedding`.
4. Encoding vocabulary must be learned on train only.
5. The same mapping must be applied to val and test.

---

## 3.4 Learned Transformation Rules
The processor must separate learning and application for every transform.

### Box-Cox
- `fit_box_cox_transformer` learns the transform from training `price`
- `apply_box_cox_transformer` applies the fitted transform to any split

### Numeric scaling
- `fit_numeric_scaler` learns numeric statistics from training data only
- `apply_numeric_scaler` applies the fitted scaler to train/val/test

### Categorical encoding
- `fit_categorical_encoder` learns category-to-index mappings from training data only
- `apply_categorical_encoder` maps values to integer indices, with unknowns as `0`

### Sample weights
- `fit_room_type_weights` computes room-type frequencies on training data only
- `apply_room_type_weights` assigns a weight to each row using the learned frequencies

---

## 3.5 Function Specification: Data Processor

### `load_raw_csvs`
- **Inputs:** `file_paths: list[pathlib.Path]`
- **Transformation Logic:** Load and concatenate raw CSV snapshots into one dataframe.
- **Outputs:** `raw_df: pd.DataFrame`

### `normalize_listing_id`
- **Inputs:** `df: pd.DataFrame, id_column: str`
- **Transformation Logic:** Standardize the primary key column name to `listing_id`.
- **Outputs:** `df: pd.DataFrame`

### `clean_price_column`
- **Inputs:** `df: pd.DataFrame`
- **Transformation Logic:** Strip currency symbols and punctuation from `price`, then convert it to numeric.
- **Outputs:** `df: pd.DataFrame`

### `drop_missing_price_rows`
- **Inputs:** `df: pd.DataFrame`
- **Transformation Logic:** Remove rows where `price` is missing or not parseable.
- **Outputs:** `df: pd.DataFrame`

### `add_season_ordinal`
- **Inputs:** `df: pd.DataFrame, month_column: str`
- **Transformation Logic:** Map snapshot month to a fixed ordinal season label.
- **Outputs:** `df: pd.DataFrame`

### `create_full_text_column`
- **Inputs:** `df: pd.DataFrame, description_column: str, amenities_column: str`
- **Transformation Logic:** Concatenate description and amenities with a fixed separator token.
- **Outputs:** `df: pd.DataFrame`

### `fit_language_detector`
- **Inputs:** `description_series: pd.Series`
- **Transformation Logic:** Learn any optional language-detection resources needed by the chosen detector wrapper.
- **Outputs:** `language_detector_artifact: object`

### `apply_language_detector`
- **Inputs:** `df: pd.DataFrame, language_detector_artifact: object, description_column: str`
- **Transformation Logic:** Detect whether each listing description is French and store the result as a boolean flag.
- **Outputs:** `df: pd.DataFrame`

### `mark_image_availability`
- **Inputs:** `df: pd.DataFrame, raw_image_dir: pathlib.Path, listing_id_column: str`
- **Transformation Logic:** Set a boolean flag indicating whether a valid raw image file exists for each listing.
- **Outputs:** `df: pd.DataFrame`

### `split_data_80_10_10`
- **Inputs:** `df: pd.DataFrame, seed: int`
- **Transformation Logic:** Deterministically split the dataframe into train, validation, and test partitions.
- **Outputs:** `train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame`

### `fit_box_cox_transformer`
- **Inputs:** `target_series: pd.Series`
- **Transformation Logic:** Learn the Box-Cox parameters from the training target only.
- **Outputs:** `box_cox_transformer: object`

### `apply_box_cox_transformer`
- **Inputs:** `target_series: pd.Series, box_cox_transformer: object`
- **Transformation Logic:** Transform the target into Box-Cox space using the fitted transformer.
- **Outputs:** `transformed_target: pd.Series`

### `fit_numeric_imputer`
- **Inputs:** `train_df: pd.DataFrame, numeric_columns: list[str]`
- **Transformation Logic:** Learn train-set medians for numeric missing-value imputation.
- **Outputs:** `numeric_imputer_artifact: dict[str, float]`

### `apply_numeric_imputer`
- **Inputs:** `df: pd.DataFrame, numeric_imputer_artifact: dict[str, float], numeric_columns: list[str]`
- **Transformation Logic:** Fill missing numeric values with train medians.
- **Outputs:** `df: pd.DataFrame`

### `fit_categorical_encoder`
- **Inputs:** `train_df: pd.DataFrame, categorical_columns: list[str]`
- **Transformation Logic:** Build train-only category vocabularies with `0` reserved for unknown.
- **Outputs:** `categorical_encoder_artifact: dict[str, dict[str, int]]`

### `apply_categorical_encoder`
- **Inputs:** `df: pd.DataFrame, categorical_encoder_artifact: dict[str, dict[str, int]], categorical_columns: list[str]`
- **Transformation Logic:** Convert categorical values to integer ids using train vocabularies.
- **Outputs:** `df: pd.DataFrame`

### `fit_numeric_scaler`
- **Inputs:** `train_df: pd.DataFrame, numeric_columns: list[str]`
- **Transformation Logic:** Learn train-only numeric scaling parameters.
- **Outputs:** `numeric_scaler_artifact: object`

### `apply_numeric_scaler`
- **Inputs:** `df: pd.DataFrame, numeric_scaler_artifact: object, numeric_columns: list[str]`
- **Transformation Logic:** Scale numeric features using the train-fitted scaler.
- **Outputs:** `df: pd.DataFrame`

### `fit_room_type_weights`
- **Inputs:** `train_df: pd.DataFrame, room_type_column: str`
- **Transformation Logic:** Compute inverse-frequency weights for each room type from training data only.
- **Outputs:** `room_type_weight_map: dict[str, float]`

### `apply_room_type_weights`
- **Inputs:** `df: pd.DataFrame, room_type_weight_map: dict[str, float], room_type_column: str`
- **Transformation Logic:** Assign a per-row sample weight using the learned room-type frequency map.
- **Outputs:** `df: pd.DataFrame`

### `assemble_tabular_output`
- **Inputs:** `df: pd.DataFrame, target_columns: list[str], feature_columns: list[str]`
- **Transformation Logic:** Select and order the final columns for tabular export.
- **Outputs:** `tabular_df: pd.DataFrame`

### `save_parquet`
- **Inputs:** `df: pd.DataFrame, output_path: pathlib.Path`
- **Transformation Logic:** Persist a dataframe to parquet without changing row order.
- **Outputs:** `None`

### `save_artifact`
- **Inputs:** `artifact: object, output_path: pathlib.Path`
- **Transformation Logic:** Persist a fitted preprocessing artifact for reuse.
- **Outputs:** `None`

---

## 4. Cleaned Variant Specification

The cleaned variant is a parallel output branch with explicit, fixed filtering.

### `filter_cleaned_variant`
- **Inputs:** `df: pd.DataFrame`
- **Transformation Logic:** Remove rows where `price < 50` or `price > 1000`.
- **Outputs:** `df: pd.DataFrame`

The cleaned variant must be processed through the same function chain as the normal variant after filtering.

---

## 5. Image Processor Specification

## 5.1 Scope
The image processor:
- reads raw downloaded images
- creates two standardized resolutions
- writes processed images keyed by `listing_id`
- writes blank black placeholder images when needed

The image processor must not:
- load tabular CSVs for model logic
- build language features
- build tabular features
- create embeddings
- depend on the data processor’s internal state

---

## 5.2 Image Rules

### Dual resolution outputs
For every listing, generate:
- `224x224`
- `336x336`

### Naming rule
- `images/processed_224/{listing_id}.jpg`
- `images/processed_336/{listing_id}.jpg`

### Missing/corrupt image rule
- If the raw image is absent or corrupted, write an all-black JPEG placeholder.
- The placeholder must be created for both resolutions.
- The filename still uses the same `listing_id`.

### Zero misalignment rule
- Images are always loaded by `listing_id`, never by sorted position.
- No array-only representation may be used as the authoritative mapping.

---

## 5.3 Function Specification: Image Processor

### `find_raw_image_path`
- **Inputs:** `listing_id: str, raw_image_dir: pathlib.Path`
- **Transformation Logic:** Resolve the expected raw image path for a listing.
- **Outputs:** `image_path: pathlib.Path | None`

### `is_image_valid`
- **Inputs:** `image_path: pathlib.Path | None`
- **Transformation Logic:** Check whether a raw image exists and is readable.
- **Outputs:** `valid: bool`

### `load_image`
- **Inputs:** `image_path: pathlib.Path`
- **Transformation Logic:** Decode the image file into an in-memory image object.
- **Outputs:** `image: object`

### `create_blank_image`
- **Inputs:** `size: tuple[int, int]`
- **Transformation Logic:** Create an all-black RGB image of the requested size.
- **Outputs:** `blank_image: object`

### `resize_and_center_crop_image`
- **Inputs:** `image: object, size: tuple[int, int]`
- **Transformation Logic:** Resize and center-crop the image to the target square resolution.
- **Outputs:** `processed_image: object`

### `save_image`
- **Inputs:** `image: object, output_path: pathlib.Path`
- **Transformation Logic:** Write the image to disk as a JPEG.
- **Outputs:** `None`

### `process_single_listing_image`
- **Inputs:** `listing_id: str, raw_image_dir: pathlib.Path, output_dir_224: pathlib.Path, output_dir_336: pathlib.Path`
- **Transformation Logic:** Produce both resolution outputs for one listing, using a blank image if the raw image is invalid.
- **Outputs:** `has_valid_image: bool`

### `process_all_listing_images`
- **Inputs:** `listing_ids: pd.Series, raw_image_dir: pathlib.Path, output_dir_224: pathlib.Path, output_dir_336: pathlib.Path`
- **Transformation Logic:** Process every listing independently by `listing_id`.
- **Outputs:** `validity_map: dict[str, bool]`

---

## 6. Feature Extraction Boundary

Feature extraction is a third separate script that runs after both processors complete.

It must:
- load exported parquets
- load processed images by `listing_id`
- run frozen backbones
- save embeddings keyed by `listing_id`

It must not:
- change raw tabular preprocessing
- alter image processing outputs
- redefine splits
- invent new target handling

---

## 7. Feature Extraction Function Specification

### `load_tabular_split`
- **Inputs:** `parquet_path: pathlib.Path`
- **Transformation Logic:** Read a split parquet into memory.
- **Outputs:** `df: pd.DataFrame`

### `load_processed_image_by_listing_id`
- **Inputs:** `listing_id: str, processed_image_dir: pathlib.Path`
- **Transformation Logic:** Load the already processed image for a listing by filename lookup.
- **Outputs:** `image: object`

### `extract_text_embedding`
- **Inputs:** `full_text: str, text_model: object, tokenizer: object`
- **Transformation Logic:** Convert listing text into a frozen text embedding.
- **Outputs:** `embedding: np.ndarray`

### `extract_image_embedding`
- **Inputs:** `image: object, vision_model: object, processor: object`
- **Transformation Logic:** Convert a processed image into a frozen vision embedding.
- **Outputs:** `embedding: np.ndarray`

### `save_embeddings_by_listing_id`
- **Inputs:** `embeddings: dict[str, np.ndarray], output_path: pathlib.Path`
- **Transformation Logic:** Persist embeddings with `listing_id` as the key.
- **Outputs:** `None`

---

## 8. Required Data Ordering

The strict execution order for the data processor is:

1. Load raw CSVs.
2. Normalize `listing_id`.
3. Clean price.
4. Drop rows with missing price.
5. Create `season_ordinal`.
6. Create `full_text`.
7. Detect language and create `is_french`.
8. Check image availability and create `has_valid_image`.
9. Split into train/val/test.
10. Fit all train-only transformers.
11. Apply transformers to train/val/test.
12. Compute `price_bc`.
13. Compute `sample_weight`.
14. Write normal outputs.
15. Optionally repeat for cleaned variant.
16. Save artifacts.

The image processor runs independently after raw images exist.

---

## 9. Main Execution Flow

Below is the intended top-level flow for the final `__main__` block of the data processor.

```python
file_paths = [pathlib.Path("listings-03-25.csv"), pathlib.Path("listings-06-25.csv"), pathlib.Path("listings-09-25.csv")]
raw_image_dir = pathlib.Path("images/raw")
output_dir = pathlib.Path("data")
artifacts_dir = pathlib.Path("data")

raw_df = load_raw_csvs(file_paths)
raw_df = normalize_listing_id(raw_df, id_column="id")
raw_df = clean_price_column(raw_df)
raw_df = drop_missing_price_rows(raw_df)
raw_df = add_season_ordinal(raw_df, month_column="snapshot_month")
raw_df = create_full_text_column(raw_df, description_column="description", amenities_column="amenities")

language_detector_artifact = fit_language_detector(raw_df["description"])
raw_df = apply_language_detector(raw_df, language_detector_artifact, description_column="description")

raw_df = mark_image_availability(raw_df, raw_image_dir, listing_id_column="listing_id")

normal_train_df, normal_val_df, normal_test_df = split_data_80_10_10(raw_df, seed=42)

box_cox_transformer = fit_box_cox_transformer(normal_train_df["price"])
numeric_imputer_artifact = fit_numeric_imputer(normal_train_df, numeric_columns=[
    "accommodates", "bathrooms", "bedrooms", "beds",
    "host_total_listings_count", "latitude", "longitude",
    "minimum_nights", "availability_365", "number_of_reviews"
])
categorical_encoder_artifact = fit_categorical_encoder(normal_train_df, categorical_columns=[
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable"
])
numeric_scaler_artifact = fit_numeric_scaler(normal_train_df, numeric_columns=[
    "accommodates", "bathrooms", "bedrooms", "beds",
    "host_total_listings_count", "latitude", "longitude",
    "minimum_nights", "availability_365", "number_of_reviews"
])
room_type_weight_map = fit_room_type_weights(normal_train_df, room_type_column="room_type")

normal_train_df = apply_numeric_imputer(normal_train_df, numeric_imputer_artifact, numeric_columns=[...])
normal_val_df = apply_numeric_imputer(normal_val_df, numeric_imputer_artifact, numeric_columns=[...])
normal_test_df = apply_numeric_imputer(normal_test_df, numeric_imputer_artifact, numeric_columns=[...])

normal_train_df = apply_categorical_encoder(normal_train_df, categorical_encoder_artifact, categorical_columns=[...])
normal_val_df = apply_categorical_encoder(normal_val_df, categorical_encoder_artifact, categorical_columns=[...])
normal_test_df = apply_categorical_encoder(normal_test_df, categorical_encoder_artifact, categorical_columns=[...])

normal_train_df = apply_numeric_scaler(normal_train_df, numeric_scaler_artifact, numeric_columns=[...])
normal_val_df = apply_numeric_scaler(normal_val_df, numeric_scaler_artifact, numeric_columns=[...])
normal_test_df = apply_numeric_scaler(normal_test_df, numeric_scaler_artifact, numeric_columns=[...])

normal_train_df["price_bc"] = apply_box_cox_transformer(normal_train_df["price"], box_cox_transformer)
normal_val_df["price_bc"] = apply_box_cox_transformer(normal_val_df["price"], box_cox_transformer)
normal_test_df["price_bc"] = apply_box_cox_transformer(normal_test_df["price"], box_cox_transformer)

normal_train_df = apply_room_type_weights(normal_train_df, room_type_weight_map, room_type_column="room_type")
normal_val_df = apply_room_type_weights(normal_val_df, room_type_weight_map, room_type_column="room_type")
normal_test_df = apply_room_type_weights(normal_test_df, room_type_weight_map, room_type_column="room_type")

normal_train_tabular = assemble_tabular_output(normal_train_df, target_columns=["price", "price_bc", "sample_weight"], feature_columns=[...])
normal_val_tabular = assemble_tabular_output(normal_val_df, target_columns=["price", "price_bc", "sample_weight"], feature_columns=[...])
normal_test_tabular = assemble_tabular_output(normal_test_df, target_columns=["price", "price_bc", "sample_weight"], feature_columns=[...])

save_parquet(normal_train_df, output_dir / "train.parquet")
save_parquet(normal_val_df, output_dir / "val.parquet")
save_parquet(normal_test_df, output_dir / "test.parquet")

save_parquet(normal_train_tabular, output_dir / "train_tabular.parquet")
save_parquet(normal_val_tabular, output_dir / "val_tabular.parquet")
save_parquet(normal_test_tabular, output_dir / "test_tabular.parquet")

save_artifact(box_cox_transformer, artifacts_dir / "price_transformer.joblib")
save_artifact(numeric_imputer_artifact, artifacts_dir / "numeric_imputer.joblib")
save_artifact(categorical_encoder_artifact, artifacts_dir / "categorical_encoder.joblib")
save_artifact(numeric_scaler_artifact, artifacts_dir / "numeric_scaler.joblib")
save_artifact(room_type_weight_map, artifacts_dir / "room_type_weights.joblib")
save_artifact(language_detector_artifact, artifacts_dir / "language_detector.joblib")
```

---

## 10. Cleaned Variant Execution Flow

```python
cleaned_df = filter_cleaned_variant(raw_df)

cleaned_train_df, cleaned_val_df, cleaned_test_df = split_data_80_10_10(cleaned_df, seed=42)

cleaned_box_cox_transformer = fit_box_cox_transformer(cleaned_train_df["price"])
cleaned_numeric_imputer_artifact = fit_numeric_imputer(cleaned_train_df, numeric_columns=[...])
cleaned_categorical_encoder_artifact = fit_categorical_encoder(cleaned_train_df, categorical_columns=[...])
cleaned_numeric_scaler_artifact = fit_numeric_scaler(cleaned_train_df, numeric_columns=[...])
cleaned_room_type_weight_map = fit_room_type_weights(cleaned_train_df, room_type_column="room_type")

cleaned_train_df = apply_numeric_imputer(cleaned_train_df, cleaned_numeric_imputer_artifact, numeric_columns=[...])
cleaned_val_df = apply_numeric_imputer(cleaned_val_df, cleaned_numeric_imputer_artifact, numeric_columns=[...])
cleaned_test_df = apply_numeric_imputer(cleaned_test_df, cleaned_numeric_imputer_artifact, numeric_columns=[...])

cleaned_train_df = apply_categorical_encoder(cleaned_train_df, cleaned_categorical_encoder_artifact, categorical_columns=[...])
cleaned_val_df = apply_categorical_encoder(cleaned_val_df, cleaned_categorical_encoder_artifact, categorical_columns=[...])
cleaned_test_df = apply_categorical_encoder(cleaned_test_df, cleaned_categorical_encoder_artifact, categorical_columns=[...])

cleaned_train_df = apply_numeric_scaler(cleaned_train_df, cleaned_numeric_scaler_artifact, numeric_columns=[...])
cleaned_val_df = apply_numeric_scaler(cleaned_val_df, cleaned_numeric_scaler_artifact, numeric_columns=[...])
cleaned_test_df = apply_numeric_scaler(cleaned_test_df, cleaned_numeric_scaler_artifact, numeric_columns=[...])

cleaned_train_df["price_bc"] = apply_box_cox_transformer(cleaned_train_df["price"], cleaned_box_cox_transformer)
cleaned_val_df["price_bc"] = apply_box_cox_transformer(cleaned_val_df["price"], cleaned_box_cox_transformer)
cleaned_test_df["price_bc"] = apply_box_cox_transformer(cleaned_test_df["price"], cleaned_box_cox_transformer)

cleaned_train_df = apply_room_type_weights(cleaned_train_df, cleaned_room_type_weight_map, room_type_column="room_type")
cleaned_val_df = apply_room_type_weights(cleaned_val_df, cleaned_room_type_weight_map, room_type_column="room_type")
cleaned_test_df = apply_room_type_weights(cleaned_test_df, cleaned_room_type_weight_map, room_type_column="room_type")

save_parquet(cleaned_train_df, output_dir / "train_cleaned.parquet")
save_parquet(cleaned_val_df, output_dir / "val_cleaned.parquet")
save_parquet(cleaned_test_df, output_dir / "test_cleaned.parquet")

save_artifact(cleaned_box_cox_transformer, artifacts_dir / "cleaned_price_transformer.joblib")
save_artifact(cleaned_numeric_imputer_artifact, artifacts_dir / "cleaned_numeric_imputer.joblib")
save_artifact(cleaned_categorical_encoder_artifact, artifacts_dir / "cleaned_categorical_encoder.joblib")
save_artifact(cleaned_numeric_scaler_artifact, artifacts_dir / "cleaned_numeric_scaler.joblib")
save_artifact(cleaned_room_type_weight_map, artifacts_dir / "cleaned_room_type_weights.joblib")
```

---

## 11. Image Processor Main Flow

```python
listing_ids = normal_df["listing_id"]

validity_map = process_all_listing_images(
    listing_ids=listing_ids,
    raw_image_dir=pathlib.Path("images/raw"),
    output_dir_224=pathlib.Path("images/processed_224"),
    output_dir_336=pathlib.Path("images/processed_336"),
)
```

---

## 12. Final Notes

1. The data processor and image processor are intentionally independent.
2. The `listing_id` key must survive every phase.
3. The tabular outputs must contain both `price` and `price_bc`.
4. The cleaned variant is strict: only `50 <= price <= 1000` survives.
5. The cleaned variant must fit and apply its own preprocessing artifacts, independent of the normal variant.
6. Unknown categories must map to `0` for embedding safety.
7. `sample_weight` must come from training room-type frequency only.
8. No `fit_transform` convenience functions should exist.
9. No class-based pipeline wrapper should be used.

If you approve this spec, the next step is to convert it into the actual functional scripts in data_processor.py and `scripts/image_processor.py`, then wire the downstream extraction scripts to the same `listing_id` contract.