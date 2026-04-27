# Image Processing Technical Specification
## Multi-Modal Airbnb Price Pipeline (Functional Version)

This document defines the image-processing phase only. It is intentionally simpler than the data processor spec and follows the same architecture principles: pure functions, no classes, no hidden state, deterministic behavior, and ID-safe outputs.

---

## 1. Purpose

Standardize listing images for vision-model experiments while preserving perfect row-to-image alignment through `listing_id`.

Primary goals:
1. Produce two model-ready resolutions per listing: `224x224` and `336x336`.
2. Guarantee every listing has a corresponding image file by writing neutral placeholders for missing/corrupt inputs.
3. Preserve ID-based lookup contract so downstream datasets load by `listing_id`, never by index order.

---

## 2. Scope And Boundaries

### In Scope
1. Read raw image files from a source directory.
2. Resolve image paths by `listing_id`.
3. Validate/decode images.
4. Resize + center-crop to 224 and 336.
5. Write processed JPEG outputs using strict naming.
6. Write neutral JPEG placeholders when source is unavailable or unreadable.
7. Return/run summary counts (valid, missing, corrupted, placeholders written).

### Out of Scope
1. No CSV/tabular/text preprocessing.
2. No train/val/test splitting.
3. No embedding extraction (CLIP/ViT/DINO etc.).
4. No batching into `.npy` arrays as source-of-truth assets.
5. No model training logic.

---

## 3. Source Of Truth Contract

1. `listing_id` is the only join key across phases.
2. Processed filenames must equal `listing_id` plus `.jpg`.
3. Processed outputs must be saved as:
   - `images/processed_224/{listing_id}.jpg`
   - `images/processed_336/{listing_id}.jpg`
4. Downstream loaders must request image by ID-based path construction.
5. Array position must never be used as identity.

---

## 4. Inputs

### Required Inputs
1. `listing_ids`: sequence of listing IDs to process.
2. `raw_image_dir`: directory where original downloaded files exist.
3. `output_dir_224`: destination directory for 224x224 outputs.
4. `output_dir_336`: destination directory for 336x336 outputs.

### Optional Inputs
1. `jpeg_quality`: integer (default 95).
2. `allowed_extensions`: ordered tuple, default `(.jpg, .jpeg, .png, .webp)`.
3. `overwrite`: boolean (default `True`).

---

## 5. Outputs

### File Outputs
1. One image file per listing in each output resolution directory.
2. Missing/corrupt source images produce neutral placeholders at both resolutions.

### Runtime Return Object
A pure dictionary summary:
1. `total_listing_ids`
2. `valid_source_images`
3. `missing_source_images`
4. `corrupt_source_images`
5. `placeholder_images_written_224`
6. `placeholder_images_written_336`
7. `real_images_written_224`
8. `real_images_written_336`

---

## 6. Image Processing Rules

1. Open source image as RGB.
2. Resize preserving aspect ratio so the shorter side reaches target size.
3. Center-crop to exact square target (`224x224` or `336x336`).
4. Save as JPEG.
5. For missing/corrupt source image, generate an ImageNet-mean neutral RGB placeholder of target size and save with the same ID filename.
6. Neutral placeholder pixel values must be fixed to 8-bit RGB `[122, 116, 104]`.

---

## 7. Functional API Specification

### `find_raw_image_path`
- Inputs: `listing_id: str`, `raw_image_dir: Path`, `allowed_extensions: tuple[str, ...]`
- Transformation Logic: Resolve candidate source file path for a listing ID by extension priority.
- Outputs: `image_path: Path | None`

### `is_image_readable`
- Inputs: `image_path: Path | None`
- Transformation Logic: Check if file exists and can be decoded as an image.
- Outputs: `is_readable: bool`, `error_type: str | None` where `error_type in {"missing", "corrupt", None}`

### `load_image_rgb`
- Inputs: `image_path: Path`
- Transformation Logic: Decode image and convert to RGB mode.
- Outputs: `image_rgb: PIL.Image.Image`

### `create_neutral_placeholder`
- Inputs: `size: tuple[int, int]`
- Transformation Logic: Create an RGB image filled with ImageNet 8-bit mean pixel values `[122, 116, 104]`.
- Outputs: `placeholder_image: PIL.Image.Image`

### `resize_and_center_crop`
- Inputs: `image_rgb: PIL.Image.Image`, `target_size: int`
- Transformation Logic: Aspect-preserving resize then center-crop to `target_size x target_size`.
- Outputs: `processed_image: PIL.Image.Image`

### `save_jpeg`
- Inputs: `image_rgb: PIL.Image.Image`, `output_path: Path`, `jpeg_quality: int`
- Transformation Logic: Persist image to JPEG, creating parent directories if needed.
- Outputs: `None`

### `process_single_listing_image`
- Inputs: `listing_id: str`, `raw_image_dir: Path`, `output_dir_224: Path`, `output_dir_336: Path`, `allowed_extensions: tuple[str, ...]`, `jpeg_quality: int`, `overwrite: bool`
- Transformation Logic: Produce both resolution outputs for one listing using source image when valid, else neutral placeholders.
- Outputs: `result: dict` with keys `listing_id`, `has_valid_source`, `error_type`, `wrote_224`, `wrote_336`, `used_placeholder`

### `process_all_listing_images`
- Inputs: `listing_ids: Sequence[str]`, `raw_image_dir: Path`, `output_dir_224: Path`, `output_dir_336: Path`, `allowed_extensions: tuple[str, ...] = (.jpg, .jpeg, .png, .webp)`, `jpeg_quality: int = 95`, `overwrite: bool = True`
- Transformation Logic: Iterate deterministically over listing IDs and process each listing independently.
- Outputs: `summary: dict`, `per_listing_results: list[dict]`

---

## 8. Failure Behavior

1. Missing image file: do not fail pipeline; write placeholder outputs.
2. Corrupt/unreadable image: do not fail pipeline; write placeholder outputs.
3. Output write error (permissions/disk full): raise exception (hard failure).
4. Invalid target size in internal calls: raise `ValueError`.

---

## 9. Determinism Rules

1. Same input IDs + same raw files + same params must produce byte-stable or behavior-stable outputs.
2. Iteration order must be deterministic (input order preserved, or explicitly sorted and documented).
3. Filename mapping is deterministic by `listing_id`.

---

## 10. Main Execution Flow (Linear)

```python
listing_ids = load_listing_ids_from_parquet(parquet_path)

summary, per_listing_results = process_all_listing_images(
    listing_ids=listing_ids,
    raw_image_dir=Path("images/raw"),
    output_dir_224=Path("images/processed_224"),
    output_dir_336=Path("images/processed_336"),
    allowed_extensions=(".jpg", ".jpeg", ".png", ".webp"),
    jpeg_quality=95,
    overwrite=True,
)

save_json(summary, Path("images/image_processing_summary.json"))
save_json(per_listing_results, Path("images/image_processing_results.json"))
```

---

## 11. Acceptance Criteria

1. Every `listing_id` in input has both output files (`224` and `336`).
2. Missing/corrupt raw images still produce both outputs (neutral placeholders with `[122, 116, 104]`).
3. Output filenames exactly match `listing_id`.
4. No data processor behavior is embedded in image processor.
5. Summary counts reconcile with input size.

---

## 12. Non-Goals For This Phase

1. Speed optimization (multiprocessing/GPU decode) is optional and deferred.
2. Perceptual quality scoring is deferred.
3. Augmentation policy for training is deferred to model dataset/dataloader stage.
