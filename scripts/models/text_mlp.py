"""
TextMLP training script for the Multi-Modal Airbnb Price Predictor.

Model registry coverage:
    #20  text_mlp.py  TextMLP  normal_bc   shallow_64
    #21  text_mlp.py  TextMLP  cleaned_bc  shallow_64
    #22  text_mlp.py  TextMLP  normal_bc   deep_256   (head ablation)

The script consumes pre-computed DistilBERT embeddings from disk and concatenates
them with the 16-column tabular feature set. No backbone encoder is instantiated.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracker import ExperimentTracker


DATA_DIR = Path("data")
EMBED_DIR = DATA_DIR / "embeddings"

FEATURE_COLS = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "season_ordinal",
    "has_valid_image",
]

CATEGORICAL_COLS = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
]

NUMERIC_COLS = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "season_ordinal",
    "has_valid_image",
]

DEFAULT_HEADS = {
    "shallow_64": [64],
    "deep_256": [256, 256],
}

DEFAULT_DROPOUT = {
    "shallow_64": 0.20,
    "deep_256": 0.30,
}


@dataclass(slots=True)
class SplitBundle:
    """In-memory split bundle for one data split."""

    listing_ids: np.ndarray
    text_embeddings: np.ndarray
    categorical_features: np.ndarray
    numeric_features: np.ndarray
    y_raw: np.ndarray
    y_target: np.ndarray
    sample_weight: np.ndarray


class TextFusionDataset(Dataset):
    """Fully in-memory dataset for tabular + text fusion training."""

    def __init__(self, bundle: SplitBundle) -> None:
        self.listing_ids = bundle.listing_ids
        self.text_embeddings = bundle.text_embeddings.astype(np.float32, copy=False)
        self.categorical_features = bundle.categorical_features.astype(np.int64, copy=False)
        self.numeric_features = bundle.numeric_features.astype(np.float32, copy=False)
        self.y_raw = bundle.y_raw.astype(np.float32, copy=False)
        self.y_target = bundle.y_target.astype(np.float32, copy=False)
        self.sample_weight = bundle.sample_weight.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.y_raw)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.text_embeddings[idx]),
            torch.from_numpy(self.categorical_features[idx]),
            torch.from_numpy(self.numeric_features[idx]),
            torch.tensor(self.y_target[idx], dtype=torch.float32),
            torch.tensor(self.y_raw[idx], dtype=torch.float32),
            torch.tensor(self.sample_weight[idx], dtype=torch.float32),
        )


class TextMLP(nn.Module):
    """Late-fusion MLP over precomputed text embeddings and tabular features."""

    def __init__(
        self,
        text_dim: int,
        hidden_dims: list[int],
        dropout: float,
        num_room_types: int,
        num_neighbourhoods: int,
        num_property_types: int,
        num_instant_bookable: int,
        embedding_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if embedding_dims is None:
            embedding_dims = [4, 16, 8, 2]

        self.room_type_embedding = nn.Embedding(num_room_types + 2, embedding_dims[0], padding_idx=0)
        self.neighbourhood_embedding = nn.Embedding(num_neighbourhoods + 2, embedding_dims[1], padding_idx=0)
        self.property_type_embedding = nn.Embedding(num_property_types + 2, embedding_dims[2], padding_idx=0)
        self.instant_bookable_embedding = nn.Embedding(num_instant_bookable + 2, embedding_dims[3], padding_idx=0)

        layers: list[nn.Module] = []
        prev_dim = text_dim + len(NUMERIC_COLS) + sum(embedding_dims)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _normalize_embedding_idx(values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.long)
        return torch.clamp(values, min=-1) + 1

    @staticmethod
    def _clip_oob_to_unknown(indices: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        in_range = indices < num_embeddings
        return torch.where(in_range, indices, torch.zeros_like(indices))

    def forward(
        self,
        text_embeddings: torch.Tensor,
        categorical_features: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        cat = self._normalize_embedding_idx(categorical_features)
        room_idx = self._clip_oob_to_unknown(cat[:, 0], self.room_type_embedding.num_embeddings)
        neighbourhood_idx = self._clip_oob_to_unknown(cat[:, 1], self.neighbourhood_embedding.num_embeddings)
        property_type_idx = self._clip_oob_to_unknown(cat[:, 2], self.property_type_embedding.num_embeddings)
        instant_bookable_idx = self._clip_oob_to_unknown(cat[:, 3], self.instant_bookable_embedding.num_embeddings)

        room_emb = self.room_type_embedding(room_idx)
        neighbourhood_emb = self.neighbourhood_embedding(neighbourhood_idx)
        property_type_emb = self.property_type_embedding(property_type_idx)
        instant_bookable_emb = self.instant_bookable_embedding(instant_bookable_idx)

        x = torch.cat(
            [
                text_embeddings,
                room_emb,
                neighbourhood_emb,
                property_type_emb,
                instant_bookable_emb,
                numeric_features,
            ],
            dim=1,
        )
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TextMLP on precomputed DistilBERT embeddings")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"],
        help="Dataset variant to train on.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        dest="run_name",
        help="Optional human-readable suffix for the run folder.",
    )
    parser.add_argument(
        "--fusion-head",
        choices=sorted(DEFAULT_HEADS),
        default="shallow_64",
        help="Fusion head label. Use deep_256 for the registry head ablation run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional dropout override; defaults to the registry-appropriate value.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on validation loss.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count tracked in config.json.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Rapid debug mode: truncates splits to 100 rows and trains for 1 epoch.",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_price_transformer(variant: str):
    suffix = "_cleaned" if variant.startswith("cleaned") else ""
    transformer_path = DATA_DIR / f"price_transformer{suffix}.joblib"
    if not transformer_path.exists():
        raise FileNotFoundError(f"Missing price transformer: {transformer_path}")
    return joblib.load(transformer_path)


def load_tabular_split(split: str, variant: str) -> pd.DataFrame:
    suffix = "_cleaned" if variant.startswith("cleaned") else ""
    parquet_path = DATA_DIR / f"{split}{suffix}_tabular.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing tabular parquet: {parquet_path}")

    columns = ["listing_id", "price", "price_bc", "sample_weight", *FEATURE_COLS]
    df = pd.read_parquet(parquet_path, columns=columns)
    df = df.reset_index(drop=True)
    df["listing_id"] = df["listing_id"].astype(str)
    return df


def _load_ids_file(path_base: Path) -> np.ndarray:
    npy_path = path_base.with_name(f"{path_base.stem}_ids.npy")
    csv_path = path_base.with_name(f"{path_base.stem}_ids.csv")

    if npy_path.exists():
        ids = np.load(npy_path, allow_pickle=False)
    elif csv_path.exists():
        ids_df = pd.read_csv(csv_path)
        if "listing_id" in ids_df.columns:
            ids = ids_df["listing_id"].to_numpy()
        elif "id" in ids_df.columns:
            ids = ids_df["id"].to_numpy()
        else:
            ids = ids_df.iloc[:, 0].to_numpy()
    else:
        raise FileNotFoundError(
            f"Could not find IDs file for {path_base.name}. Expected {npy_path.name} or {csv_path.name}."
        )

    return ids.astype(str)


def _find_embedding_matrix(split: str, variant: str) -> Path:
    suffix = "_cleaned" if variant.startswith("cleaned") else "_normal"
    candidates = [
        EMBED_DIR / f"{split}_text{suffix}.npy",
        EMBED_DIR / f"{split}_text_{suffix.lstrip('_')}.npy",
        EMBED_DIR / f"{split}_distilbert-base-multilingual-cased.npy",
        EMBED_DIR / f"{split}_distilbert-base-multilingual-cased{suffix}.npy",
        EMBED_DIR / f"{split}_distilbert-base-multilingual-cased_{suffix.lstrip('_')}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find an embedding matrix for split='{split}' variant='{variant}'. "
        f"Looked for: {[path.name for path in candidates]}"
    )


def load_embedding_split(split: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    matrix_path = _find_embedding_matrix(split, variant)
    embeddings = np.load(matrix_path, mmap_mode=None, allow_pickle=False).astype(np.float32, copy=False)
    ids = _load_ids_file(matrix_path)
    return embeddings, ids


def maybe_smoke_slice(bundle: SplitBundle, max_rows: int = 100) -> SplitBundle:
    return SplitBundle(
        listing_ids=bundle.listing_ids[:max_rows].copy(),
        text_embeddings=bundle.text_embeddings[:max_rows].copy(),
        categorical_features=bundle.categorical_features[:max_rows].copy(),
        numeric_features=bundle.numeric_features[:max_rows].copy(),
        y_raw=bundle.y_raw[:max_rows].copy(),
        y_target=bundle.y_target[:max_rows].copy(),
        sample_weight=bundle.sample_weight[:max_rows].copy(),
    )


def load_split_bundle(split: str, variant: str, target_col: str) -> SplitBundle:
    df = load_tabular_split(split, variant)
    text_embeddings, emb_ids = load_embedding_split(split, variant)

    tab_ids = df["listing_id"].to_numpy(dtype=str)
    assert np.array_equal(
        tab_ids, emb_ids
    ), "ALIGNMENT FAILURE: tabular and embedding row order do not match"

    categorical_features = df[CATEGORICAL_COLS].to_numpy(dtype=np.int64, copy=True)
    numeric_features = df[NUMERIC_COLS].to_numpy(dtype=np.float32, copy=True)
    return SplitBundle(
        listing_ids=tab_ids,
        text_embeddings=text_embeddings,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        y_raw=df["price"].to_numpy(dtype=np.float32, copy=True),
        y_target=df[target_col].to_numpy(dtype=np.float32, copy=True),
        sample_weight=df["sample_weight"].to_numpy(dtype=np.float32, copy=True),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def to_raw_dollars(preds: np.ndarray, price_transformer) -> np.ndarray:
    if price_transformer is None:
        return preds.astype(np.float32, copy=False)

    # Box-Cox inverse requires lambda * y + 1 > 0; clip into the valid domain
    # to avoid NaN/Inf when model outputs drift outside transform support.
    preds_safe = preds.astype(np.float64, copy=True)
    lam = float(price_transformer.lambdas_[0])
    eps = 1e-6
    if lam < 0:
        upper = (-1.0 / lam) - eps
        preds_safe = np.minimum(preds_safe, upper)
    elif lam > 0:
        lower = (-1.0 / lam) + eps
        preds_safe = np.maximum(preds_safe, lower)

    raw = price_transformer.inverse_transform(preds_safe.reshape(-1, 1)).ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0)
    return raw.astype(np.float32, copy=False)


def weighted_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
    target_transform: str,
) -> torch.Tensor:
    if target_transform == "box_cox":
        per_sample = F.mse_loss(preds, targets, reduction="none")
    else:
        per_sample = F.smooth_l1_loss(preds, targets, reduction="none", beta=1.0)
    return (per_sample * sample_weight).mean()


def infer_cardinalities(*bundles: SplitBundle) -> dict[str, int]:
    room_type_max = max(int(bundle.categorical_features[:, 0].max()) for bundle in bundles)
    neighbourhood_max = max(int(bundle.categorical_features[:, 1].max()) for bundle in bundles)
    property_type_max = max(int(bundle.categorical_features[:, 2].max()) for bundle in bundles)
    instant_bookable_max = max(int(bundle.categorical_features[:, 3].max()) for bundle in bundles)
    return {
        "room_type": room_type_max,
        "neighbourhood_cleansed": neighbourhood_max,
        "property_type": property_type_max,
        "instant_bookable": instant_bookable_max,
    }


def build_model_from_cardinalities(
    text_dim: int,
    cardinalities: dict[str, int],
    fusion_head: str,
    dropout: float,
) -> TextMLP:
    hidden_dims = DEFAULT_HEADS[fusion_head]
    return TextMLP(
        text_dim=text_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_room_types=cardinalities["room_type"],
        num_neighbourhoods=cardinalities["neighbourhood_cleansed"],
        num_property_types=cardinalities["property_type"],
        num_instant_bookable=cardinalities["instant_bookable"],
    )


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    price_transformer,
    target_transform: str,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true_raw_parts: list[np.ndarray] = []
    y_pred_raw_parts: list[np.ndarray] = []

    with torch.no_grad():
        for text_embeddings, categorical_features, numeric_features, _, y_raw, _ in loader:
            text_embeddings = text_embeddings.to(device, non_blocking=True)
            categorical_features = categorical_features.to(device, non_blocking=True)
            numeric_features = numeric_features.to(device, non_blocking=True)
            preds = model(text_embeddings, categorical_features, numeric_features).detach().cpu().numpy()
            y_pred_raw = to_raw_dollars(preds, price_transformer) if target_transform == "box_cox" else preds
            y_true_raw_parts.append(y_raw.numpy())
            y_pred_raw_parts.append(y_pred_raw)

    y_true_raw = np.concatenate(y_true_raw_parts).astype(np.float32, copy=False)
    y_pred_raw = np.concatenate(y_pred_raw_parts).astype(np.float32, copy=False)
    return compute_metrics(y_true_raw, y_pred_raw), y_true_raw, y_pred_raw


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tracker: ExperimentTracker,
    price_transformer,
    target_transform: str,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        tracker.start_epoch()
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for text_embeddings, categorical_features, numeric_features, targets, _, sample_weight in train_loader:
            text_embeddings = text_embeddings.to(device, non_blocking=True)
            categorical_features = categorical_features.to(device, non_blocking=True)
            numeric_features = numeric_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(text_embeddings, categorical_features, numeric_features)
            loss = weighted_loss(preds, targets, sample_weight, target_transform)
            loss.backward()
            optimizer.step()

            batch_size = text_embeddings.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for text_embeddings, categorical_features, numeric_features, targets, _, sample_weight in val_loader:
                text_embeddings = text_embeddings.to(device, non_blocking=True)
                categorical_features = categorical_features.to(device, non_blocking=True)
                numeric_features = numeric_features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                sample_weight = sample_weight.to(device, non_blocking=True)
                preds = model(text_embeddings, categorical_features, numeric_features)
                loss = weighted_loss(preds, targets, sample_weight, target_transform)
                batch_size = text_embeddings.shape[0]
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)
        val_metrics, _, _ = evaluate_split(
            model=model,
            loader=val_loader,
            price_transformer=price_transformer,
            target_transform=target_transform,
            device=device,
        )

        tracker.log_epoch(
            train_loss=train_loss,
            val_loss=val_loss,
            val_rmse_raw=val_metrics["rmse"],
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            tracker.save_best_model(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | val_rmse=${val_metrics['rmse']:.2f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def main() -> None:
    args = parse_args()
    set_seed(42)

    variant = args.variant
    target_col = "price_bc" if variant.endswith("_bc") else "price"
    target_transform = "box_cox" if variant.endswith("_bc") else "none"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dropout = args.dropout if args.dropout is not None else DEFAULT_DROPOUT[args.fusion_head]
    epochs = 1 if args.smoke_test else args.epochs

    print("=" * 80)
    print(f"TextMLP | variant={variant} | target={target_col} | head={args.fusion_head}")
    print(f"device={device} | smoke_test={args.smoke_test}")
    print("=" * 80)

    price_transformer = load_price_transformer(variant) if variant.endswith("_bc") else None

    print("Loading train/val data and embeddings into RAM...")
    train_bundle = load_split_bundle("train", variant, target_col)
    val_bundle = load_split_bundle("val", variant, target_col)

    if args.smoke_test:
        print("[SMOKE TEST] Truncating train/val splits to 100 rows immediately after load.")
        train_bundle = maybe_smoke_slice(train_bundle, 100)
        val_bundle = maybe_smoke_slice(val_bundle, 100)

    cardinalities = infer_cardinalities(train_bundle, val_bundle)

    if price_transformer is not None:
        print(f"Loaded price transformer for {variant}.")
        box_cox_lambda = float(price_transformer.lambdas_[0])
    else:
        box_cox_lambda = None

    tracker = ExperimentTracker(
        model_type="TextMLP",
        modalities="tab+text",
        variant=variant,
        run_name=args.run_name,
        fusion_head=args.fusion_head,
        batch_size=args.batch_size,
        dataloader_workers=args.num_workers,
        device_used=str(device),
        is_smoke_test=args.smoke_test,
        config={
            "feature_cols": FEATURE_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols": NUMERIC_COLS,
            "text_embedding_source": "precomputed_distilbert",
            "text_embedding_dim": int(train_bundle.text_embeddings.shape[1]),
            "categorical_cardinalities": {
                "room_type": cardinalities["room_type"],
                "neighbourhood_cleansed": cardinalities["neighbourhood_cleansed"],
                "property_type": cardinalities["property_type"],
                "instant_bookable": cardinalities["instant_bookable"],
            },
            "tabular_numeric_dim": len(NUMERIC_COLS),
            "hidden_dims": DEFAULT_HEADS[args.fusion_head],
            "embedding_dims": [4, 16, 8, 2],
            "dropout": float(dropout),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "epochs": int(epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "target_column": target_col,
            "target_transform": target_transform,
            "box_cox_lambda": box_cox_lambda,
        },
    )

    if price_transformer is not None:
        tracker.set_box_cox_lambda(box_cox_lambda)

    train_dataset = TextFusionDataset(train_bundle)
    val_dataset = TextFusionDataset(val_bundle)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model_from_cardinalities(
        text_dim=int(train_bundle.text_embeddings.shape[1]),
        cardinalities=cardinalities,
        fusion_head=args.fusion_head,
        dropout=dropout,
    ).to(device)

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_parameters:,}")

    t0 = time.time()
    model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
        price_transformer=price_transformer,
        target_transform=target_transform,
        device=device,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    training_time_minutes = (time.time() - t0) / 60.0

    train_metrics, train_y_true_raw, train_y_pred_raw = evaluate_split(
        model=model,
        loader=train_loader,
        price_transformer=price_transformer,
        target_transform=target_transform,
        device=device,
    )
    val_metrics, val_y_true_raw, val_y_pred_raw = evaluate_split(
        model=model,
        loader=val_loader,
        price_transformer=price_transformer,
        target_transform=target_transform,
        device=device,
    )

    # Respect split isolation: load test only for final evaluation.
    print("Loading test split into RAM for final evaluation...")
    test_bundle = load_split_bundle("test", variant, target_col)
    if args.smoke_test:
        print("[SMOKE TEST] Truncating test split to 100 rows immediately after load.")
        test_bundle = maybe_smoke_slice(test_bundle, 100)

    test_dataset = TextFusionDataset(test_bundle)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_metrics, test_y_true_raw, test_y_pred_raw = evaluate_split(
        model=model,
        loader=test_loader,
        price_transformer=price_transformer,
        target_transform=target_transform,
        device=device,
    )

    print("Final metrics (raw Canadian dollars):")
    print(
        f"  Train | RMSE ${train_metrics['rmse']:.2f} | MAE ${train_metrics['mae']:.2f} | R² {train_metrics['r2']:.4f}"
    )
    print(
        f"  Val   | RMSE ${val_metrics['rmse']:.2f} | MAE ${val_metrics['mae']:.2f} | R² {val_metrics['r2']:.4f}"
    )
    print(
        f"  Test  | RMSE ${test_metrics['rmse']:.2f} | MAE ${test_metrics['mae']:.2f} | R² {test_metrics['r2']:.4f}"
    )

    tracker.finish(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions={
            "train_y_true": train_y_true_raw,
            "train_y_pred": train_y_pred_raw,
            "val_y_true": val_y_true_raw,
            "val_y_pred": val_y_pred_raw,
            "test_y_true": test_y_true_raw,
            "test_y_pred": test_y_pred_raw,
        },
        trainable_parameters=trainable_parameters,
        training_time_minutes=training_time_minutes,
        best_hyperparams={
            "fusion_head": args.fusion_head,
            "hidden_dims": DEFAULT_HEADS[args.fusion_head],
            "dropout": float(dropout),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "best_val_loss": float(best_val_loss),
        },
        extra_artifacts=None,
    )


if __name__ == "__main__":
    main()