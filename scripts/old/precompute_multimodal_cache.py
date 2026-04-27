"""
Precompute multimodal training cache to reduce CPU bottlenecks during training.

Outputs per split (train/val/test):
- <split>_input_ids.npy           (int32)
- <split>_attention_mask.npy      (int8)
- <split>_images_uint8.npy        (uint8, N x 3 x 224 x 224)
- <split>_tabular.npy             (float32)
- <split>_price.npy               (float32)

Text construction includes:
- description
- amenities
- selected listing attributes
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm


def build_image_index(image_dir: Path):
    """Build lookup by listing_id from flat processed_224 image folder."""
    index = {}
    jpg_files = list(image_dir.glob("*.jpg"))
    for p in jpg_files:
        key = p.stem.split("_")[0]
        if key not in index:
            index[key] = p
    return index


def build_text(row: pd.Series) -> str:
    attr_parts = [
        f"room_type={row.get('room_type', '')}",
        f"property_type={row.get('property_type', '')}",
        f"neighbourhood={row.get('neighbourhood_cleansed', '')}",
        f"accommodates={row.get('accommodates', '')}",
        f"bathrooms={row.get('bathrooms', '')}",
        f"bedrooms={row.get('bedrooms', '')}",
        f"beds={row.get('beds', '')}",
        f"instant_bookable={row.get('instant_bookable', '')}",
        f"minimum_nights={row.get('minimum_nights', '')}",
        f"availability_365={row.get('availability_365', '')}",
        f"number_of_reviews={row.get('number_of_reviews', '')}",
        f"season_ordinal={row.get('season_ordinal', '')}",
    ]
    return (
        f"Description: {str(row.get('description', '')).strip()} "
        f"Amenities: {str(row.get('amenities', '')).strip()} "
        f"Listing attributes: {'; '.join(attr_parts)}"
    ).strip()[:1200]


def load_image_uint8(listing_id: str, image_index: dict, image_size: int = 224):
    path = image_index.get(listing_id)
    if path is None:
        arr = np.full((image_size, image_size, 3), 128, dtype=np.uint8)
    else:
        try:
            img = Image.open(path).convert("RGB").resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.uint8)
        except Exception:
            arr = np.full((image_size, image_size, 3), 128, dtype=np.uint8)
    # HWC -> CHW
    return np.transpose(arr, (2, 0, 1))


def precompute_split(split_name: str,
                     df: pd.DataFrame,
                     tokenizer,
                     image_index: dict,
                     tabular_cols,
                     max_text_length: int,
                     output_dir: Path):
    n = len(df)
    input_ids = np.zeros((n, max_text_length), dtype=np.int32)
    attention_mask = np.zeros((n, max_text_length), dtype=np.int8)
    images_uint8 = np.zeros((n, 3, 224, 224), dtype=np.uint8)
    tabular = np.zeros((n, len(tabular_cols)), dtype=np.float32)
    price = np.zeros((n,), dtype=np.float32)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n, desc=f"Precompute {split_name}")):
        text = build_text(row)
        tok = tokenizer(
            text,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids[i] = tok["input_ids"][0].astype(np.int32)
        attention_mask[i] = tok["attention_mask"][0].astype(np.int8)

        listing_id = str(row.get("id", ""))
        images_uint8[i] = load_image_uint8(listing_id, image_index, image_size=224)

        tabular[i] = np.array([float(row.get(c, 0.0)) for c in tabular_cols], dtype=np.float32)
        price[i] = float(row.get("price", 0.0))

    np.save(output_dir / f"{split_name}_input_ids.npy", input_ids)
    np.save(output_dir / f"{split_name}_attention_mask.npy", attention_mask)
    np.save(output_dir / f"{split_name}_images_uint8.npy", images_uint8)
    np.save(output_dir / f"{split_name}_tabular.npy", tabular)
    np.save(output_dir / f"{split_name}_price.npy", price)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-dir", type=str, default="/mnt/nvme_data/linux_sys/ml_images/processed_224")
    parser.add_argument("--output-dir", type=str, default="data/cache_multimodal")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-multilingual-cased")
    parser.add_argument("--max-text-length", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "train_tabular.parquet")
    val_df = pd.read_parquet(data_dir / "val_tabular.parquet")
    test_df = pd.read_parquet(data_dir / "test_tabular.parquet")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    image_index = build_image_index(Path(args.image_dir))

    tabular_cols = [
        "room_type",
        "neighbourhood_cleansed",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "minimum_nights",
        "season_ordinal",
        "beds",
        "host_total_listings_count",
        "latitude",
        "longitude",
        "property_type",
        "instant_bookable",
        "availability_365",
        "number_of_reviews",
    ]
    tabular_cols = [c for c in tabular_cols if c in train_df.columns]

    precompute_split("train", train_df, tokenizer, image_index, tabular_cols, args.max_text_length, output_dir)
    precompute_split("val", val_df, tokenizer, image_index, tabular_cols, args.max_text_length, output_dir)
    precompute_split("test", test_df, tokenizer, image_index, tabular_cols, args.max_text_length, output_dir)

    meta = {
        "tokenizer": args.tokenizer,
        "max_text_length": args.max_text_length,
        "tabular_cols": tabular_cols,
        "image_dir": args.image_dir,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Cache precompute complete")
    print(f"Output dir: {output_dir}")
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
