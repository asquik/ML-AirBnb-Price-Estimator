from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError


NEUTRAL_PLACEHOLDER_RGB = (122, 116, 104)
DEFAULT_ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_SAFE_PIXELS = 89_478_485


def _as_path(path_like: Path | str) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def find_raw_image_path(listing_id: str, raw_image_dir: Path, allowed_extensions: tuple[str, ...]) -> Path | None:
    """Resolve candidate source file path for a listing ID by extension priority."""
    raw_image_dir = _as_path(raw_image_dir)
    for extension in allowed_extensions:
        candidate = raw_image_dir / f"{listing_id}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate

    listing_dir = raw_image_dir / str(listing_id)
    if listing_dir.exists() and listing_dir.is_dir():
        nested_candidates = sorted(
            candidate
            for candidate in listing_dir.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in allowed_extensions
        )
        if nested_candidates:
            return nested_candidates[0]
    return None


def is_image_readable(image_path: Path | None) -> tuple[bool, str | None]:
    """Check if file exists and can be decoded as an image."""
    if image_path is None:
        return False, "missing"

    image_path = _as_path(image_path)
    if not image_path.exists() or not image_path.is_file():
        return False, "missing"

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width * height > MAX_SAFE_PIXELS:
                return False, "corrupt"
            image.verify()
        return True, None
    except Exception:
        return False, "corrupt"


def load_image_rgb(image_path: Path) -> Image.Image:
    """Decode image and convert to RGB mode."""
    image_path = _as_path(image_path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return rgb.copy()


def create_neutral_placeholder(size: tuple[int, int]) -> Image.Image:
    """Create an ImageNet-mean neutral RGB placeholder image."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Placeholder size must be positive")
    return Image.new("RGB", size, color=NEUTRAL_PLACEHOLDER_RGB)


def resize_and_center_crop(image_rgb: Image.Image, target_size: int) -> Image.Image:
    """Aspect-preserving resize then center-crop to target_size x target_size."""
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    return ImageOps.fit(
        image_rgb,
        (target_size, target_size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def save_jpeg(image_rgb: Image.Image, output_path: Path, jpeg_quality: int) -> None:
    """Persist image to JPEG, creating parent directories if needed."""
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb.save(output_path, "JPEG", quality=jpeg_quality)


def process_single_listing_image(
    listing_id: str,
    raw_image_dir: Path,
    output_dir_224: Path,
    output_dir_336: Path,
    allowed_extensions: tuple[str, ...],
    jpeg_quality: int,
    overwrite: bool,
) -> dict:
    """Produce both resolution outputs for one listing using source image when valid, else neutral placeholders."""
    raw_image_dir = _as_path(raw_image_dir)
    output_dir_224 = _as_path(output_dir_224)
    output_dir_336 = _as_path(output_dir_336)

    out_224_path = output_dir_224 / f"{listing_id}.jpg"
    out_336_path = output_dir_336 / f"{listing_id}.jpg"

    should_write_224 = overwrite or not out_224_path.exists()
    should_write_336 = overwrite or not out_336_path.exists()

    source_path = find_raw_image_path(listing_id, raw_image_dir, allowed_extensions)
    is_valid_source, error_type = is_image_readable(source_path)

    used_placeholder = not is_valid_source

    if is_valid_source:
        try:
            source_rgb = load_image_rgb(source_path)  # type: ignore[arg-type]
            image_224 = resize_and_center_crop(source_rgb, 224)
            image_336 = resize_and_center_crop(source_rgb, 336)
        except Exception:
            is_valid_source = False
            error_type = "corrupt"
            used_placeholder = True
            image_224 = create_neutral_placeholder((224, 224))
            image_336 = create_neutral_placeholder((336, 336))
    else:
        image_224 = create_neutral_placeholder((224, 224))
        image_336 = create_neutral_placeholder((336, 336))

    if should_write_224:
        save_jpeg(image_224, out_224_path, jpeg_quality=jpeg_quality)
    if should_write_336:
        save_jpeg(image_336, out_336_path, jpeg_quality=jpeg_quality)

    return {
        "listing_id": listing_id,
        "has_valid_source": is_valid_source,
        "error_type": error_type,
        "wrote_224": should_write_224,
        "wrote_336": should_write_336,
        "used_placeholder": used_placeholder,
    }


def process_all_listing_images(
    listing_ids: Sequence[str],
    raw_image_dir: Path,
    output_dir_224: Path,
    output_dir_336: Path,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    jpeg_quality: int = 95,
    overwrite: bool = True,
) -> tuple[dict, list[dict]]:
    """Iterate deterministically over listing IDs and process each listing independently."""
    per_listing_results: list[dict] = []

    for listing_id in listing_ids:
        result = process_single_listing_image(
            listing_id=str(listing_id),
            raw_image_dir=raw_image_dir,
            output_dir_224=output_dir_224,
            output_dir_336=output_dir_336,
            allowed_extensions=allowed_extensions,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )
        per_listing_results.append(result)

    summary = {
        "total_listing_ids": len(listing_ids),
        "valid_source_images": sum(1 for r in per_listing_results if r["has_valid_source"]),
        "missing_source_images": sum(1 for r in per_listing_results if r["error_type"] == "missing"),
        "corrupt_source_images": sum(1 for r in per_listing_results if r["error_type"] == "corrupt"),
        "placeholder_images_written_224": sum(
            1 for r in per_listing_results if r["used_placeholder"] and r["wrote_224"]
        ),
        "placeholder_images_written_336": sum(
            1 for r in per_listing_results if r["used_placeholder"] and r["wrote_336"]
        ),
        "real_images_written_224": sum(
            1 for r in per_listing_results if (not r["used_placeholder"]) and r["wrote_224"]
        ),
        "real_images_written_336": sum(
            1 for r in per_listing_results if (not r["used_placeholder"]) and r["wrote_336"]
        ),
    }

    return summary, per_listing_results
