"""Pre-compute DistilBERT text embeddings for all dataset splits.

This script reads raw parquet files from ``data/`` and extracts one 768D
embedding per listing from the ``full_text`` column using
``distilbert-base-multilingual-cased``.

Outputs are written to ``data/embeddings/`` with the naming contract required
by the training specification:

- ``train_text_normal.npy`` and ``train_text_normal_ids.npy``
- ``val_text_normal.npy`` and ``val_text_normal_ids.npy``
- ``test_text_normal.npy`` and ``test_text_normal_ids.npy``
- ``train_text_cleaned.npy`` and ``train_text_cleaned_ids.npy``
- ``val_text_cleaned.npy`` and ``val_text_cleaned_ids.npy``
- ``test_text_cleaned.npy`` and ``test_text_cleaned_ids.npy``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DATA_DIR = Path("data")
EMBED_DIR = DATA_DIR / "embeddings"
MODEL_NAME = "distilbert-base-multilingual-cased"

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


class TextDataset(Dataset):
    """Memory-efficient dataset that stores IDs and raw text only."""

    def __init__(self, listing_ids: np.ndarray, texts: np.ndarray) -> None:
        self.listing_ids = listing_ids
        self.texts = texts

    def __len__(self) -> int:
        return len(self.listing_ids)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return str(self.listing_ids[index]), str(self.texts[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DistilBERT text embeddings for normal and cleaned splits."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DistilBERT inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (e.g., cuda, cuda:0, cpu).",
    )
    return parser.parse_args()


def load_raw_split(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=["listing_id", "full_text"])
    listing_ids = df["listing_id"].astype(str).to_numpy(copy=True)
    texts = df["full_text"].fillna("").astype(str).to_numpy(copy=True)
    return listing_ids, texts


def extract_one_split(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    listing_ids: np.ndarray,
    texts: np.ndarray,
    out_embed_path: Path,
    out_ids_path: Path,
    batch_size: int,
    max_length: int,
    num_workers: int,
    device: torch.device,
) -> None:
    dataset = TextDataset(listing_ids=listing_ids, texts=texts)

    def collate_fn(batch: list[tuple[str, str]]) -> tuple[list[str], dict[str, torch.Tensor]]:
        ids, batch_texts = zip(*batch)
        tokens = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return list(ids), tokens

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    hidden_size = int(model.config.hidden_size)
    out_matrix = open_memmap(
        out_embed_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), hidden_size),
    )
    out_ids = np.empty(len(dataset), dtype=object)

    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch_ids, tokens in loader:
            tokens = {
                key: value.to(device, non_blocking=True)
                for key, value in tokens.items()
            }
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_np = cls_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)

            batch_n = batch_np.shape[0]
            out_matrix[cursor : cursor + batch_n] = batch_np
            out_ids[cursor : cursor + batch_n] = batch_ids
            cursor += batch_n

    out_matrix.flush()
    np.save(out_ids_path, out_ids.astype(str))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    for universe, split_map in SPLIT_SPECS.items():
        for split, parquet_name in split_map.items():
            parquet_path = DATA_DIR / parquet_name
            listing_ids, texts = load_raw_split(parquet_path)

            out_embed_path = EMBED_DIR / f"{split}_text_{universe}.npy"
            out_ids_path = EMBED_DIR / f"{split}_text_{universe}_ids.npy"

            print(
                f"Extracting {split}/{universe}: rows={len(listing_ids)} -> {out_embed_path.name}"
            )
            extract_one_split(
                model=model,
                tokenizer=tokenizer,
                listing_ids=listing_ids,
                texts=texts,
                out_embed_path=out_embed_path,
                out_ids_path=out_ids_path,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_workers=args.num_workers,
                device=device,
            )

    print("Text embedding extraction complete.")


if __name__ == "__main__":
    main()
