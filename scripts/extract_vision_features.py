"""Pre-compute CLIP image embeddings for all dataset splits.

This script reads listing IDs from raw parquets in ``data/`` and images from
``images/processed_224/`` and ``images/processed_336/``. It extracts CLIP
embeddings and saves listing-level vectors by averaging all image embeddings
for each listing.

Outputs are written to ``data/embeddings/`` with the naming contract required
by the training specification:

- ``train_image_normal_224.npy`` and ``train_image_normal_224_ids.npy``
- ``val_image_normal_224.npy`` and ``val_image_normal_224_ids.npy``
- ``test_image_normal_224.npy`` and ``test_image_normal_224_ids.npy``
- ``train_image_cleaned_224.npy`` and ``train_image_cleaned_224_ids.npy``
- ``val_image_cleaned_224.npy`` and ``val_image_cleaned_224_ids.npy``
- ``test_image_cleaned_224.npy`` and ``test_image_cleaned_224_ids.npy``
- ``train_image_normal_336.npy`` and ``train_image_normal_336_ids.npy``
- ``val_image_normal_336.npy`` and ``val_image_normal_336_ids.npy``
- ``test_image_normal_336.npy`` and ``test_image_normal_336_ids.npy``
- ``train_image_cleaned_336.npy`` and ``train_image_cleaned_336_ids.npy``
- ``val_image_cleaned_336.npy`` and ``val_image_cleaned_336_ids.npy``
- ``test_image_cleaned_336.npy`` and ``test_image_cleaned_336_ids.npy``
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel


DATA_DIR = Path("data")
EMBED_DIR = DATA_DIR / "embeddings"
RESOLUTION_CONFIG = {
    224: {
        "image_root": Path("images/processed_224"),
        "model_name": "openai/clip-vit-base-patch32",
        "batch_size": None,
    },
    336: {
        "image_root": Path("images/processed_336"),
        "model_name": "openai/clip-vit-large-patch14-336",
        "batch_size": 4,
    },
}

SPLIT_SPECS = {
    "normal": {
        "train": "train.parquet",
        "val": "val.parquet",
        "test": "test.parquet",
    },
    "cleaned": {
        "train": "train_cleaned.parquet",
        "val": "val_cleaned.parquet",
        "test": "test_cleaned.parquet",
    },
}

# Placeholder chosen per project design for missing images.
PLACEHOLDER_RGB = (122, 116, 104)


class ImageRecordDataset(Dataset):
    """Dataset over expanded (listing_index, image_path) records."""

    def __init__(self, records: list[tuple[int, Path | None]], placeholder_size: tuple[int, int]) -> None:
        self.records = records
        self.placeholder_size = placeholder_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, Image.Image]:
        listing_index, image_path = self.records[index]
        if image_path is None:
            image = Image.new("RGB", self.placeholder_size, color=PLACEHOLDER_RGB)
        else:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        return listing_index, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP listing-level image embeddings for normal and cleaned splits."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for CLIP image inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        choices=sorted(RESOLUTION_CONFIG.keys()),
        default=[224, 336],
        help="Image resolutions to extract (default: both 224 and 336).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional custom image root. Only valid when a single resolution is selected "
            "via --resolutions."
        ),
    )
    return parser.parse_args()


def load_split_listing_ids(parquet_path: Path) -> np.ndarray:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=["listing_id"])
    return df["listing_id"].astype(str).to_numpy(copy=True)


def infer_listing_id(image_path: Path, image_root: Path) -> str | None:
    # Case 1: nested layout .../<listing_id>/<image_name>.jpg
    if image_path.parent != image_root and image_path.parent.name.isdigit():
        return image_path.parent.name

    # Case 2: flat layout with names like "<listing_id>.jpg" or
    # "<listing_id>_1.jpg".
    stem = image_path.stem
    if stem.isdigit():
        return stem

    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else None


def build_image_index(image_root: Path) -> dict[str, list[Path]]:
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    image_index: dict[str, list[Path]] = defaultdict(list)
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}

    for image_path in image_root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in valid_suffixes:
            continue

        listing_id = infer_listing_id(image_path=image_path, image_root=image_root)
        if listing_id is None:
            continue
        image_index[listing_id].append(image_path)

    for listing_id in image_index:
        image_index[listing_id].sort()

    return dict(image_index)


def build_records(
    listing_ids: np.ndarray,
    image_index: dict[str, list[Path]],
) -> tuple[list[tuple[int, Path | None]], int]:
    records: list[tuple[int, Path | None]] = []
    missing_count = 0

    for idx, listing_id in enumerate(listing_ids):
        paths = image_index.get(str(listing_id))
        if not paths:
            # Keep alignment by generating a synthetic placeholder input.
            records.append((idx, None))
            missing_count += 1
            continue

        for path in paths:
            records.append((idx, path))

    return records, missing_count


def extract_listing_embeddings(
    model: CLIPModel,
    processor: CLIPImageProcessor,
    records: list[tuple[int, Path | None]],
    listing_count: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    placeholder_size: tuple[int, int],
) -> np.ndarray:
    dataset = ImageRecordDataset(records=records, placeholder_size=placeholder_size)

    def collate_fn(batch: list[tuple[int, Image.Image]]) -> tuple[np.ndarray, torch.Tensor]:
        listing_indices = np.array([item[0] for item in batch], dtype=np.int64)
        images = [item[1] for item in batch]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return listing_indices, pixel_values

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    embedding_dim = int(model.config.projection_dim)
    sums = np.zeros((listing_count, embedding_dim), dtype=np.float32)
    counts = np.zeros(listing_count, dtype=np.int32)

    def format_cuda_memory(device_obj: torch.device) -> str:
        if device_obj.type != "cuda":
            return "vram=cpu"

        allocated = torch.cuda.memory_allocated(device_obj)
        reserved = torch.cuda.memory_reserved(device_obj)
        peak_allocated = torch.cuda.max_memory_allocated(device_obj)
        peak_reserved = torch.cuda.max_memory_reserved(device_obj)
        return (
            f"vram_allocated={allocated / 1024**2:.1f}MiB "
            f"vram_reserved={reserved / 1024**2:.1f}MiB "
            f"vram_peak_allocated={peak_allocated / 1024**2:.1f}MiB "
            f"vram_peak_reserved={peak_reserved / 1024**2:.1f}MiB"
        )

    model.eval()
    with torch.no_grad():
        for batch_index, (listing_indices, pixel_values) in enumerate(loader, start=1):
            batch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            pixel_values = pixel_values.to(device, non_blocking=True)
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            image_features = model.visual_projection(vision_outputs.pooler_output)
            batch_np = image_features.detach().cpu().numpy().astype(np.float32, copy=False)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_start

            for row_index, feature in zip(listing_indices, batch_np):
                sums[row_index] += feature
                counts[row_index] += 1

            print(
                f"batch={batch_index} size={len(listing_indices)} "
                f"time={batch_elapsed:.3f}s {format_cuda_memory(device)}"
            )

    if np.any(counts == 0):
        raise RuntimeError("Found listings with zero image count after record expansion.")

    return sums / counts[:, None]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.image_root is not None and len(args.resolutions) != 1:
        raise ValueError("--image-root can only be used with a single resolution in --resolutions.")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for resolution in args.resolutions:
        config = RESOLUTION_CONFIG[resolution]
        image_root = args.image_root if args.image_root is not None else config["image_root"]
        model_name = str(config["model_name"])
        resolution_batch_size = int(config["batch_size"]) if config["batch_size"] is not None else int(args.batch_size)
        placeholder_size = (resolution, resolution)

        print(f"\n=== Resolution {resolution}: model={model_name}, image_root={image_root} ===")
        print(f"Using batch size {resolution_batch_size} for resolution {resolution}.")

        print(f"Indexing image files under {image_root} ...")
        image_index = build_image_index(image_root)
        total_images = sum(len(v) for v in image_index.values())
        print(f"Indexed {total_images} images mapped to {len(image_index)} listing IDs.")

        processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)

        for universe, split_map in SPLIT_SPECS.items():
            for split, parquet_name in split_map.items():
                listing_ids = load_split_listing_ids(DATA_DIR / parquet_name)
                records, missing_count = build_records(listing_ids=listing_ids, image_index=image_index)

                print(
                    f"Extracting {split}/{universe}/{resolution}: listings={len(listing_ids)}, "
                    f"expanded_images={len(records)}, synthetic_placeholders={missing_count}"
                )

                embeddings = extract_listing_embeddings(
                    model=model,
                    processor=processor,
                    records=records,
                    listing_count=len(listing_ids),
                    batch_size=resolution_batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    placeholder_size=placeholder_size,
                )

                out_embed_path = EMBED_DIR / f"{split}_image_{universe}_{resolution}.npy"
                out_ids_path = EMBED_DIR / f"{split}_image_{universe}_{resolution}_ids.npy"
                np.save(out_embed_path, embeddings.astype(np.float32, copy=False))
                np.save(out_ids_path, listing_ids.astype(str))

    print("Vision embedding extraction complete.")


if __name__ == "__main__":
    main()
