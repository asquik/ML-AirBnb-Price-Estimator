"""
Inference script: Load trained models and run predictions on test data.

This script demonstrates:
1. Loading saved models from outputs/models/
2. Running predictions on test sets
3. Comparing predictions vs actual prices
4. Showing example listings with predictions
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Recreate model class definitions from training script
NUMERIC_COLS_DL = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "minimum_nights",
    "season_ordinal",
    "beds",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "availability_365",
    "number_of_reviews",
]

CATEGORICAL_COLS_DL = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
]

FEATURE_COLS = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "minimum_nights",
    "season_ordinal",
    "beds",
    "host_total_listings_count",
    "latitude",
    "longitude",
    "availability_365",
    "number_of_reviews",
]


def _cat_to_embedding_idx(values: pd.Series) -> np.ndarray:
    """Map encoded categories to embedding indices."""
    arr = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int).to_numpy()
    arr = np.where(arr < 0, 0, arr + 1)
    return arr.reshape(-1, 1).astype(np.int64)


class AirbnbTabularDataset(Dataset):
    """PyTorch Dataset for tabular Airbnb data."""

    def __init__(self, df: pd.DataFrame):
        self.numeric_data = df[NUMERIC_COLS_DL].to_numpy(dtype=np.float32)
        self.room_type = _cat_to_embedding_idx(df["room_type"])
        self.neighbourhood = _cat_to_embedding_idx(df["neighbourhood_cleansed"])
        self.property_type = _cat_to_embedding_idx(df["property_type"])
        self.instant_bookable = _cat_to_embedding_idx(df["instant_bookable"])
        self.targets = df["price_bc"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.numeric_data)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.numeric_data[idx]),
            torch.from_numpy(self.room_type[idx]),
            torch.from_numpy(self.neighbourhood[idx]),
            torch.from_numpy(self.property_type[idx]),
            torch.from_numpy(self.instant_bookable[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class TabularMLP(nn.Module):
    """MLP for tabular data with embeddings for categorical features."""

    def __init__(
        self,
        num_numeric: int,
        num_room_types: int,
        num_neighbourhoods: int,
        num_property_types: int,
        num_instant_bookable: int,
        hidden_dims: list[int],
        embedding_dims: list[int],
        dropout_rate: float,
    ):
        super().__init__()

        self.room_type_emb = nn.Embedding(num_room_types, embedding_dims[0])
        self.neighbourhood_emb = nn.Embedding(num_neighbourhoods, embedding_dims[1])
        self.property_type_emb = nn.Embedding(num_property_types, embedding_dims[2])
        self.instant_bookable_emb = nn.Embedding(num_instant_bookable, embedding_dims[3])

        total_input = num_numeric + sum(embedding_dims)

        layers: list[nn.Module] = []
        input_size = total_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim

        layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        numeric: torch.Tensor,
        room_type: torch.Tensor,
        neighbourhood: torch.Tensor,
        property_type: torch.Tensor,
        instant_bookable: torch.Tensor,
    ) -> torch.Tensor:
        room_emb = self.room_type_emb(room_type).squeeze(1)
        neigh_emb = self.neighbourhood_emb(neighbourhood).squeeze(1)
        prop_emb = self.property_type_emb(property_type).squeeze(1)
        inst_emb = self.instant_bookable_emb(instant_bookable).squeeze(1)
        x = torch.cat([numeric, room_emb, neigh_emb, prop_emb, inst_emb], dim=1)
        return self.mlp(x)


def list_available_models(models_dir: Path) -> list:
    """List all available saved models."""
    models = []
    if models_dir.exists():
        models = sorted(models_dir.glob("*.joblib")) + sorted(models_dir.glob("*.pt"))
    return [(m.name, m) for m in models]


def load_sklearn_model(model_path: Path):
    """Load a joblib-saved sklearn model."""
    return joblib.load(model_path)


def load_torch_model(model_path: Path, model_architecture_hint: str, train_df: pd.DataFrame):
    """Load a PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer embedding sizes from training data
    room_max = int(pd.to_numeric(train_df["room_type"], errors="coerce").fillna(-1).max())
    neigh_max = int(pd.to_numeric(train_df["neighbourhood_cleansed"], errors="coerce").fillna(-1).max())
    prop_max = int(pd.to_numeric(train_df["property_type"], errors="coerce").fillna(-1).max())
    inst_max = int(pd.to_numeric(train_df["instant_bookable"], errors="coerce").fillna(-1).max())

    num_room_types = max(room_max, 0) + 2
    num_neighbourhoods = max(neigh_max, 0) + 2
    num_property_types = max(prop_max, 0) + 2
    num_instant_bookable = max(inst_max, 0) + 2

    # Determine architecture from filename hint
    if "simple" in model_architecture_hint.lower():
        model = TabularMLP(
            num_numeric=len(NUMERIC_COLS_DL),
            num_room_types=num_room_types,
            num_neighbourhoods=num_neighbourhoods,
            num_property_types=num_property_types,
            num_instant_bookable=num_instant_bookable,
            hidden_dims=[64, 32],
            embedding_dims=[4, 8, 8, 2],
            dropout_rate=0.2,
        ).to(device)
    else:  # sophisticated
        model = TabularMLP(
            num_numeric=len(NUMERIC_COLS_DL),
            num_room_types=num_room_types,
            num_neighbourhoods=num_neighbourhoods,
            num_property_types=num_property_types,
            num_instant_bookable=num_instant_bookable,
            hidden_dims=[256, 128, 64, 32],
            embedding_dims=[8, 16, 16, 4],
            dropout_rate=0.4,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_sklearn_inference(model, test_df: pd.DataFrame, test_raw: np.ndarray, price_transformer) -> dict:
    """Run inference with sklearn model."""
    X_test = test_df[FEATURE_COLS].copy()
    y_pred_bc = model.predict(X_test)
    y_pred_raw = price_transformer.inverse_transform(y_pred_bc.reshape(-1, 1)).ravel()

    return {"predictions": y_pred_raw, "actual": test_raw}


def run_torch_inference(model, test_df: pd.DataFrame, test_raw: np.ndarray, price_transformer) -> dict:
    """Run inference with PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AirbnbTabularDataset(test_df)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    predictions_bc = []
    with torch.no_grad():
        for numeric, room_type, neighbourhood, property_type, instant_bookable, _ in loader:
            numeric = numeric.to(device)
            room_type = room_type.to(device)
            neighbourhood = neighbourhood.to(device)
            property_type = property_type.to(device)
            instant_bookable = instant_bookable.to(device)
            pred = model(numeric, room_type, neighbourhood, property_type, instant_bookable)
            predictions_bc.append(pred.cpu().numpy().ravel())

    y_pred_bc = np.concatenate(predictions_bc)
    y_pred_raw = price_transformer.inverse_transform(y_pred_bc.reshape(-1, 1)).ravel()

    return {"predictions": y_pred_raw, "actual": test_raw}


def main():
    models_dir = Path("outputs/models")
    data_dir = Path("data")

    # List available models
    available_models = list_available_models(models_dir)

    if not available_models:
        print("❌ No models found in outputs/models/")
        print("   Run scripts/train_tabular_models.py first to train and save models.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    for i, (name, path) in enumerate(available_models, 1):
        model_type = "PyTorch" if name.endswith(".pt") else "sklearn"
        print(f"({i:2d}) {name:<60} [{model_type}]")

    # Ask user to select a model
    while True:
        try:
            selection = int(input("\nSelect model (enter number): ").strip())
            if 1 <= selection <= len(available_models):
                model_name, model_path = available_models[selection - 1]
                break
            else:
                print(f"Invalid selection. Choose 1-{len(available_models)}.")
        except ValueError:
            print("Invalid input. Enter a number.")

    # Determine variant and feature set from filename
    if "normal" in model_name:
        variant_tag = None
        variant_name = "normal"
    else:
        variant_tag = "cleaned"
        variant_name = "cleaned"

    # Load data
    print(f"\n📂 Loading {variant_name} dataset...")
    suffix = "" if variant_tag is None else f"_{variant_tag}"

    test_df = pd.read_parquet(data_dir / f"test_tabular{suffix}.parquet")
    price_transformer = joblib.load(data_dir / f"price_transformer{suffix}.joblib")

    y_test_raw = test_df["price"].values

    print(f"✅ Loaded {len(test_df)} test records")

    # Load model
    print(f"\n🤖 Loading model: {model_name}")
    if model_name.endswith(".pt"):
        model = load_torch_model(model_path, model_name, test_df)
        print("   Using PyTorch inference...")
        result = run_torch_inference(model, test_df, y_test_raw, price_transformer)
    else:
        model = load_sklearn_model(model_path)
        print("   Using sklearn inference...")
        result = run_sklearn_inference(model, test_df, y_test_raw, price_transformer)

    y_pred = result["predictions"]
    y_actual = result["actual"]

    # Compute metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print(f"\n" + "=" * 80)
    print("TEST SET METRICS")
    print("=" * 80)
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # Show example predictions
    print(f"\n" + "=" * 80)
    print("PREDICTION EXAMPLES (First 15 records)")
    print("=" * 80)

    examples_df = pd.DataFrame({
        "Actual ($)": y_actual[:15].astype(int),
        "Predicted ($)": y_pred[:15].astype(int),
        "Error ($)": (y_actual[:15] - y_pred[:15]).astype(int),
        "Error (%)": ((y_actual[:15] - y_pred[:15]) / y_actual[:15] * 100).astype(int),
    })
    examples_df.index = range(1, len(examples_df) + 1)

    print(examples_df.to_string())

    # Prediction distribution
    print(f"\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)
    errors = np.abs(y_actual - y_pred)
    print(f"Mean absolute error: ${errors.mean():.2f}")
    print(f"Median absolute error: ${np.median(errors):.2f}")
    print(f"Std dev of error: ${errors.std():.2f}")
    print(f"Max error: ${errors.max():.2f}")
    print(f"% predictions within $100: {(errors <= 100).sum() / len(errors) * 100:.1f}%")
    print(f"% predictions within $50: {(errors <= 50).sum() / len(errors) * 100:.1f}%")


if __name__ == "__main__":
    main()
