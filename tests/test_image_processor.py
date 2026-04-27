import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.image_processor import (
    create_neutral_placeholder,
    find_raw_image_path,
    is_image_readable,
    load_image_rgb,
    process_all_listing_images,
    process_single_listing_image,
    resize_and_center_crop,
    save_jpeg,
)


IMAGENET_MEAN_RGB = (122, 116, 104)


def _make_image(path: Path, size=(320, 240), mode="RGB", color=(200, 10, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, size, color=color)
    image.save(path)


@pytest.fixture
def image_dirs(tmp_path):
    raw = tmp_path / "raw"
    out224 = tmp_path / "processed_224"
    out336 = tmp_path / "processed_336"
    raw.mkdir(parents=True)
    return raw, out224, out336


# find_raw_image_path (3 tests)
def test_find_raw_image_path_returns_preferred_extension(image_dirs):
    raw, _, _ = image_dirs
    _make_image(raw / "123.png")
    _make_image(raw / "123.jpg")

    resolved = find_raw_image_path("123", raw, (".jpg", ".png"))

    assert resolved is not None
    assert resolved.name == "123.jpg"


def test_find_raw_image_path_returns_none_when_not_found(image_dirs):
    raw, _, _ = image_dirs

    resolved = find_raw_image_path("999", raw, (".jpg", ".png"))

    assert resolved is None


def test_find_raw_image_path_finds_existing_webp(image_dirs):
    raw, _, _ = image_dirs
    _make_image(raw / "42.webp")

    resolved = find_raw_image_path("42", raw, (".jpg", ".jpeg", ".png", ".webp"))

    assert resolved is not None
    assert resolved.suffix == ".webp"


def test_find_raw_image_path_searches_nested_listing_directory(image_dirs):
    raw, _, _ = image_dirs
    nested_image = raw / "123" / "nested" / "photo.jpeg"
    _make_image(nested_image)

    resolved = find_raw_image_path("123", raw, (".jpg", ".jpeg", ".png", ".webp"))

    assert resolved == nested_image


# is_image_readable (4 tests)
def test_is_image_readable_valid_image(image_dirs):
    raw, _, _ = image_dirs
    image_path = raw / "1.jpg"
    _make_image(image_path)

    readable, error_type = is_image_readable(image_path)

    assert readable is True
    assert error_type is None


def test_is_image_readable_none_path_returns_missing():
    readable, error_type = is_image_readable(None)

    assert readable is False
    assert error_type == "missing"


def test_is_image_readable_missing_file_returns_missing(image_dirs):
    raw, _, _ = image_dirs

    readable, error_type = is_image_readable(raw / "missing.jpg")

    assert readable is False
    assert error_type == "missing"


def test_is_image_readable_corrupt_bytes_returns_corrupt(image_dirs):
    raw, _, _ = image_dirs
    corrupt = raw / "bad.jpg"
    corrupt.write_bytes(b"not-an-image")

    readable, error_type = is_image_readable(corrupt)

    assert readable is False
    assert error_type == "corrupt"


# load_image_rgb (3 tests)
def test_load_image_rgb_happy_path_rgb(image_dirs):
    raw, _, _ = image_dirs
    image_path = raw / "2.jpg"
    _make_image(image_path, mode="RGB", color=(10, 20, 30))

    image = load_image_rgb(image_path)

    assert image.mode == "RGB"
    assert image.size == (320, 240)


def test_load_image_rgb_converts_rgba_to_rgb(image_dirs):
    raw, _, _ = image_dirs
    image_path = raw / "3.png"
    _make_image(image_path, mode="RGBA", color=(1, 2, 3, 255))

    image = load_image_rgb(image_path)

    assert image.mode == "RGB"


def test_load_image_rgb_raises_on_missing_path(image_dirs):
    raw, _, _ = image_dirs

    with pytest.raises(Exception):
        load_image_rgb(raw / "does_not_exist.jpg")


# create_neutral_placeholder (3 tests)
def test_create_neutral_placeholder_happy_path_size_and_mode():
    image = create_neutral_placeholder((224, 224))

    assert image.size == (224, 224)
    assert image.mode == "RGB"


def test_create_neutral_placeholder_uses_imagenet_mean_rgb_values():
    image = create_neutral_placeholder((4, 4))
    arr = np.array(image)

    assert tuple(arr[0, 0].tolist()) == IMAGENET_MEAN_RGB
    assert np.all(arr[:, :, 0] == IMAGENET_MEAN_RGB[0])
    assert np.all(arr[:, :, 1] == IMAGENET_MEAN_RGB[1])
    assert np.all(arr[:, :, 2] == IMAGENET_MEAN_RGB[2])


def test_create_neutral_placeholder_raises_on_invalid_size():
    with pytest.raises(ValueError):
        create_neutral_placeholder((0, 224))


# resize_and_center_crop (4 tests)
def test_resize_and_center_crop_happy_path_square_output():
    image = Image.new("RGB", (320, 240), color=(255, 0, 0))

    out = resize_and_center_crop(image, 224)

    assert out.size == (224, 224)


def test_resize_and_center_crop_landscape_input(image_dirs):
    image = Image.new("RGB", (640, 240), color=(0, 255, 0))

    out = resize_and_center_crop(image, 224)

    assert out.size == (224, 224)


def test_resize_and_center_crop_portrait_input(image_dirs):
    image = Image.new("RGB", (240, 640), color=(0, 0, 255))

    out = resize_and_center_crop(image, 336)

    assert out.size == (336, 336)


def test_resize_and_center_crop_raises_on_invalid_target_size():
    image = Image.new("RGB", (100, 100), color=(0, 0, 0))
    with pytest.raises(ValueError):
        resize_and_center_crop(image, 0)


# save_jpeg (3 tests)
def test_save_jpeg_happy_path_writes_file(tmp_path):
    out_path = tmp_path / "a" / "b" / "img.jpg"
    image = Image.new("RGB", (100, 100), color=(11, 22, 33))

    save_jpeg(image, out_path, jpeg_quality=95)

    assert out_path.exists()


def test_save_jpeg_creates_parent_dirs(tmp_path):
    out_path = tmp_path / "nested" / "dir" / "img.jpg"
    image = Image.new("RGB", (64, 64), color=(1, 1, 1))

    save_jpeg(image, out_path, jpeg_quality=90)

    assert out_path.parent.exists()


def test_save_jpeg_raises_on_invalid_quality(tmp_path):
    out_path = tmp_path / "img.jpg"
    image = Image.new("RGB", (32, 32), color=(1, 2, 3))

    with pytest.raises(ValueError):
        save_jpeg(image, out_path, jpeg_quality=200)


# process_single_listing_image (4 tests)
def test_process_single_listing_image_happy_path_real_image(image_dirs):
    raw, out224, out336 = image_dirs
    _make_image(raw / "100.jpg", size=(500, 300), color=(33, 44, 55))

    result = process_single_listing_image(
        listing_id="100",
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert result["has_valid_source"] is True
    assert result["used_placeholder"] is False
    assert (out224 / "100.jpg").exists()
    assert (out336 / "100.jpg").exists()


def test_process_single_listing_image_missing_source_uses_neutral_placeholder(image_dirs):
    raw, out224, out336 = image_dirs

    result = process_single_listing_image(
        listing_id="101",
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert result["has_valid_source"] is False
    assert result["error_type"] == "missing"
    assert result["used_placeholder"] is True

    img224 = Image.open(out224 / "101.jpg").convert("RGB")
    pixel = img224.getpixel((0, 0))
    assert pixel == IMAGENET_MEAN_RGB


def test_process_single_listing_image_corrupt_source_uses_placeholder(image_dirs):
    raw, out224, out336 = image_dirs
    (raw / "102.jpg").write_bytes(b"corrupt")

    result = process_single_listing_image(
        listing_id="102",
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert result["has_valid_source"] is False
    assert result["error_type"] == "corrupt"
    assert result["used_placeholder"] is True
    assert (out224 / "102.jpg").exists()
    assert (out336 / "102.jpg").exists()


def test_process_single_listing_image_overwrite_false_skips_existing_outputs(image_dirs):
    raw, out224, out336 = image_dirs
    _make_image(raw / "103.jpg", size=(400, 300), color=(200, 10, 10))
    _make_image(out224 / "103.jpg", size=(224, 224), color=(1, 1, 1))
    _make_image(out336 / "103.jpg", size=(336, 336), color=(2, 2, 2))

    result = process_single_listing_image(
        listing_id="103",
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=False,
    )

    assert result["wrote_224"] is False
    assert result["wrote_336"] is False


# process_all_listing_images (3 tests)
def test_process_all_listing_images_happy_path_summary_and_outputs(image_dirs):
    raw, out224, out336 = image_dirs
    _make_image(raw / "1.jpg")
    _make_image(raw / "2.jpg")

    summary, per_listing = process_all_listing_images(
        listing_ids=["1", "2"],
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert summary["total_listing_ids"] == 2
    assert summary["valid_source_images"] == 2
    assert len(per_listing) == 2
    assert (out224 / "1.jpg").exists()
    assert (out336 / "2.jpg").exists()


def test_process_all_listing_images_handles_mixed_valid_missing_corrupt(image_dirs):
    raw, out224, out336 = image_dirs
    _make_image(raw / "10.jpg")
    (raw / "12.jpg").write_bytes(b"corrupt")

    summary, per_listing = process_all_listing_images(
        listing_ids=["10", "11", "12"],
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert summary["total_listing_ids"] == 3
    assert summary["valid_source_images"] == 1
    assert summary["missing_source_images"] == 1
    assert summary["corrupt_source_images"] == 1
    assert summary["placeholder_images_written_224"] == 2
    assert summary["placeholder_images_written_336"] == 2


def test_process_all_listing_images_preserves_input_order_in_results(image_dirs):
    raw, out224, out336 = image_dirs
    _make_image(raw / "a.jpg")
    _make_image(raw / "b.jpg")

    _, per_listing = process_all_listing_images(
        listing_ids=["b", "a"],
        raw_image_dir=raw,
        output_dir_224=out224,
        output_dir_336=out336,
        allowed_extensions=(".jpg", ".png"),
        jpeg_quality=95,
        overwrite=True,
    )

    assert [item["listing_id"] for item in per_listing] == ["b", "a"]
