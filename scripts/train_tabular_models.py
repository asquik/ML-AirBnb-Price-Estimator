"""
Train and compare tabular models on Airbnb price prediction.

Models:
1. Decision Tree (baseline)
2. LightGBM (if installed) OR sklearn GradientBoostingRegressor
3. Ridge Regression
4. Polynomial Features + Ridge
5. MLP (Simple) - shallow neural network (PyTorch)
6. MLP (Sophisticated) - deeper neural network with regularization (PyTorch)

This script is designed to run the full suite twice:
- default dataset artifacts (no suffix)
- cleaned dataset artifacts (suffix "_cleaned"), produced by the data processor

Key guarantees:
- Uses deterministic train/val/test split (80/10/10) produced upstream.
- Uses the per-variant Box-Cox transformer (fit on that variant's TRAIN only).
- No leakage: models tune on val; final metrics reported on test.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import Dataset, DataLoader

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

# =========================================================================
# FEATURE FLAG FOR DEEP LEARNING
# =========================================================================
ENABLE_DEEP_LEARNING = True  # requires PyTorch

# Feature columns (preprocessed: encoded categoricals, scaled numerics)
FEATURE_COLS = [
    # categoricals (encoded)
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
    # numerics (scaled)
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

# Deep learning expects these columns (already present in tabular parquets)
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


# =========================================================================
# MODEL PERSISTENCE HELPERS
# =========================================================================
def _sanitize_model_name(model_name: str) -> str:
    """Convert model name to safe filename."""
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("=", "_")
        .replace(".", "")
        .replace("[", "")
        .replace("]", "")
    )


def _save_sklearn_model(model, model_name: str, variant_name: str, models_dir: Path) -> str:
    """Save sklearn model and return filename."""
    sanitized = _sanitize_model_name(model_name)
    filename = f"{sanitized}_{variant_name}.joblib"
    filepath = models_dir / filename
    joblib.dump(model, filepath)
    return filename


def _save_torch_model(model, model_name: str, variant_name: str, models_dir: Path) -> str:
    """Save PyTorch model and return filename."""
    sanitized = _sanitize_model_name(model_name)
    filename = f"{sanitized}_{variant_name}.pt"
    filepath = models_dir / filename
    torch.save(model.state_dict(), filepath)
    return filename


def _suffix(tag: Optional[str]) -> str:
    return "" if not tag else f"_{tag}"


def _load_variant(data_dir: Path, tag: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    sfx = _suffix(tag)

    train_df = pd.read_parquet(data_dir / f"train_tabular{sfx}.parquet")
    val_df = pd.read_parquet(data_dir / f"val_tabular{sfx}.parquet")
    test_df = pd.read_parquet(data_dir / f"test_tabular{sfx}.parquet")

    price_transformer = joblib.load(data_dir / f"price_transformer{sfx}.joblib")
    return train_df, val_df, test_df, price_transformer


def _cat_to_embedding_idx(values: pd.Series) -> np.ndarray:
    """Map encoded categories to embedding indices.

    Upstream preprocessing maps unseen categories in val/test to -1.
    Embeddings cannot index -1, so we map:
      - any negative value -> 0 (unknown)
      - non-negative value k -> k+1

    Result indices are int64 in shape (N, 1).
    """
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
        hidden_dims: List[int],
        embedding_dims: List[int],
        dropout_rate: float,
    ):
        super().__init__()

        self.room_type_emb = nn.Embedding(num_room_types, embedding_dims[0])
        self.neighbourhood_emb = nn.Embedding(num_neighbourhoods, embedding_dims[1])
        self.property_type_emb = nn.Embedding(num_property_types, embedding_dims[2])
        self.instant_bookable_emb = nn.Embedding(num_instant_bookable, embedding_dims[3])

        total_input = num_numeric + sum(embedding_dims)

        layers: List[nn.Module] = []
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


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15,
) -> nn.Module:
    """Train PyTorch model with early stopping."""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<12} {'Patience':<10}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for numeric, room_type, neighbourhood, property_type, instant_bookable, target in train_loader:
            numeric = numeric.to(device)
            room_type = room_type.to(device)
            neighbourhood = neighbourhood.to(device)
            property_type = property_type.to(device)
            instant_bookable = instant_bookable.to(device)
            target = target.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(numeric, room_type, neighbourhood, property_type, instant_bookable)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(numeric)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for numeric, room_type, neighbourhood, property_type, instant_bookable, target in val_loader:
                numeric = numeric.to(device)
                room_type = room_type.to(device)
                neighbourhood = neighbourhood.to(device)
                property_type = property_type.to(device)
                instant_bookable = instant_bookable.to(device)
                target = target.to(device).unsqueeze(1)

                pred = model(numeric, room_type, neighbourhood, property_type, instant_bookable)
                loss = criterion(pred, target)
                val_loss += loss.item() * len(numeric)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch + 1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<12.6f} {patience_counter:<10}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_torch_model(
    model: nn.Module,
    dataset: Dataset,
    y_raw: np.ndarray,
    model_name: str,
    price_transformer,
    device: torch.device,
) -> Dict:
    """Evaluate PyTorch model and return metrics in raw dollars."""

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

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

    rmse = np.sqrt(mean_squared_error(y_raw, y_pred_raw))
    mae = mean_absolute_error(y_raw, y_pred_raw)
    r2 = r2_score(y_raw, y_pred_raw)

    print(f"\n{model_name}")
    print("=" * 80)
    print(f"Test (raw $):        RMSE=${rmse:.2f}  MAE=${mae:.2f}  R²={r2:.4f}")

    return {"test_rmse_raw": rmse, "test_mae_raw": mae, "test_r2_raw": r2}


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_val_raw: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_test_raw: np.ndarray,
    model_name: str,
    price_transformer,
) -> Dict:
    """Evaluate sklearn model on validation and test sets."""

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Metrics on Box-Cox transformed scale
    val_r2_bc = r2_score(y_val, y_val_pred)
    test_r2_bc = r2_score(y_test, y_test_pred)

    # Inverse-transform predictions back to raw dollars
    y_val_pred_raw = price_transformer.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
    y_test_pred_raw = price_transformer.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

    val_rmse_raw = np.sqrt(mean_squared_error(y_val_raw, y_val_pred_raw))
    val_mae_raw = mean_absolute_error(y_val_raw, y_val_pred_raw)
    val_r2_raw = r2_score(y_val_raw, y_val_pred_raw)

    test_rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_test_pred_raw))
    test_mae_raw = mean_absolute_error(y_test_raw, y_test_pred_raw)
    test_r2_raw = r2_score(y_test_raw, y_test_pred_raw)

    print(f"\n{model_name}")
    print("=" * 80)
    print(f"Validation (raw $):  RMSE=${val_rmse_raw:.2f}  MAE=${val_mae_raw:.2f}  R²={val_r2_raw:.4f}")
    print(f"Test (raw $):        RMSE=${test_rmse_raw:.2f}  MAE=${test_mae_raw:.2f}  R²={test_r2_raw:.4f}")
    print(f"[Box-Cox R²: val={val_r2_bc:.4f}, test={test_r2_bc:.4f}]")

    return {
        "model_name": model_name,
        "val_rmse_raw": val_rmse_raw,
        "val_mae_raw": val_mae_raw,
        "val_r2_raw": val_r2_raw,
        "test_rmse_raw": test_rmse_raw,
        "test_mae_raw": test_mae_raw,
        "test_r2_raw": test_r2_raw,
        "test_r2_bc": test_r2_bc,
        "best_params": str(model.get_params()) if hasattr(model, "get_params") else "N/A",
    }


def run_variant(data_dir: Path, tag: Optional[str], variant_name: str, models_dir: Path) -> pd.DataFrame:
    print("\n" + "#" * 90)
    print(f"DATASET VARIANT: {variant_name} (suffix='{_suffix(tag)}')")
    print("#" * 90)

    print("Loading train/val/test tabular data...")
    train_df, val_df, test_df, price_transformer = _load_variant(data_dir, tag)

    # Extract X,y for each split
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["price_bc"].values
    y_train_raw = train_df["price"].values

    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df["price_bc"].values
    y_val_raw = val_df["price"].values

    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["price_bc"].values
    y_test_raw = test_df["price"].values

    print(f"✅ Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"   Features: {FEATURE_COLS}")
    print(f"   Target: price_bc (Box-Cox transformed)")
    print(
        f"   Raw price stats (train): mean=${y_train_raw.mean():.2f}, std=${y_train_raw.std():.2f}, max=${y_train_raw.max():.0f}"
    )
    print(f"   Transformed price stats (train): mean={y_train.mean():.3f}, std={y_train.std():.3f}\n")

    all_model_results: List[Dict] = []

    # =========================================================================
    # 1. DECISION TREE BASELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. DECISION TREE BASELINE")
    print("=" * 80)

    best_dt_model = None
    best_dt_params = {}
    best_dt_val_rmse = float("inf")

    max_depths = [3, 5, 8, 12, 15, 20, 25, 30]
    min_samples_leafs = [2, 5, 10, 20, 30]

    print(f"\nHyperparameter sweep: max_depth={max_depths}, min_samples_leaf={min_samples_leafs}")
    print(f"{'max_depth':<12} {'min_samples_leaf':<18} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
    print("-" * 70)

    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leafs:
            dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
            dt.fit(X_train, y_train)
            y_val_pred = dt.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            print(f"{max_depth:<12} {min_samples_leaf:<18} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")

            if val_rmse < best_dt_val_rmse:
                best_dt_val_rmse = val_rmse
                best_dt_model = dt
                best_dt_params = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}

    print("-" * 70)
    print(f"✅ Best Decision Tree: max_depth={best_dt_params['max_depth']}, min_samples_leaf={best_dt_params['min_samples_leaf']}")

    dt_result = evaluate_model(
        best_dt_model,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
        f"Decision Tree (max_depth={best_dt_params['max_depth']}, min_samples_leaf={best_dt_params['min_samples_leaf']})",
        price_transformer,
    )
    dt_result["model_filename"] = _save_sklearn_model(best_dt_model, dt_result["model_name"], variant_name, models_dir)
    all_model_results.append(dt_result)

    # =========================================================================
    # 2. GRADIENT BOOSTING (sklearn or LightGBM)
    # =========================================================================
    if HAS_LIGHTGBM:
        print("\n" + "=" * 80)
        print("2. LIGHTGBM (GRADIENT BOOSTING)")
        print("=" * 80)

        best_lgb_model = None
        best_lgb_params = {}
        best_lgb_val_rmse = float("inf")

        num_leaves_list = [15, 31, 50, 100]
        learning_rates = [0.01, 0.05, 0.1]

        print(
            f"\nHyperparameter sweep: num_leaves={num_leaves_list}, learning_rate={learning_rates}, n_estimators={[100, 200]}"
        )
        print(f"{'num_leaves':<12} {'learning_rate':<16} {'n_estimators':<14} {'Val RMSE':<12} {'Val R²':<10}")
        print("-" * 70)

        for num_leaves in num_leaves_list:
            for learning_rate in learning_rates:
                for n_estimators in [100, 200]:
                    lgb_model = lgb.LGBMRegressor(
                        num_leaves=num_leaves,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        random_state=42,
                        verbose=-1,
                        force_col_wise=True,
                    )
                    lgb_model.fit(X_train, y_train)
                    y_val_pred = lgb_model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    val_r2 = r2_score(y_val, y_val_pred)

                    print(f"{num_leaves:<12} {learning_rate:<16.3f} {n_estimators:<14} {val_rmse:<12.4f} {val_r2:<10.4f}")

                    if val_rmse < best_lgb_val_rmse:
                        best_lgb_val_rmse = val_rmse
                        best_lgb_model = lgb_model
                        best_lgb_params = {
                            "num_leaves": num_leaves,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                        }

        print("-" * 70)
        print(
            f"✅ Best LightGBM: num_leaves={best_lgb_params['num_leaves']}, learning_rate={best_lgb_params['learning_rate']}, n_estimators={best_lgb_params['n_estimators']}"
        )

        lgb_result = evaluate_model(
            best_lgb_model,
            X_val,
            y_val,
            y_val_raw,
            X_test,
            y_test,
            y_test_raw,
            f"LightGBM (num_leaves={best_lgb_params['num_leaves']}, lr={best_lgb_params['learning_rate']}, n_est={best_lgb_params['n_estimators']})",
            price_transformer,
        )
        lgb_result["model_filename"] = _save_sklearn_model(best_lgb_model, lgb_result["model_name"], variant_name, models_dir)
        all_model_results.append(lgb_result)

    else:
        print("\n" + "=" * 80)
        print("2. GRADIENT BOOSTING (SKLEARN - GradientBoostingRegressor)")
        print("=" * 80)

        best_gb_model = None
        best_gb_params = {}
        best_gb_val_rmse = float("inf")

        max_depths = [3, 5, 7]
        learning_rates = [0.01, 0.05, 0.1]
        n_estimators_list = [50, 100, 200]

        print(f"\nHyperparameter sweep: max_depth={max_depths}, learning_rate={learning_rates}, n_estimators={n_estimators_list}")
        print(f"{'max_depth':<12} {'learning_rate':<16} {'n_estimators':<14} {'Val RMSE':<12} {'Val R²':<10}")
        print("-" * 70)

        for max_depth in max_depths:
            for learning_rate in learning_rates:
                for n_estimators in [100, 200]:
                    gb_model = GradientBoostingRegressor(
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        random_state=42,
                    )
                    gb_model.fit(X_train, y_train)
                    y_val_pred = gb_model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    val_r2 = r2_score(y_val, y_val_pred)

                    print(f"{max_depth:<12} {learning_rate:<16.3f} {n_estimators:<14} {val_rmse:<12.4f} {val_r2:<10.4f}")

                    if val_rmse < best_gb_val_rmse:
                        best_gb_val_rmse = val_rmse
                        best_gb_model = gb_model
                        best_gb_params = {
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                        }

        print("-" * 70)
        print(
            f"✅ Best GradientBoosting: max_depth={best_gb_params['max_depth']}, learning_rate={best_gb_params['learning_rate']}, n_estimators={best_gb_params['n_estimators']}"
        )

        gb_result = evaluate_model(
            best_gb_model,
            X_val,
            y_val,
            y_val_raw,
            X_test,
            y_test,
            y_test_raw,
            f"GradientBoosting (max_depth={best_gb_params['max_depth']}, lr={best_gb_params['learning_rate']}, n_est={best_gb_params['n_estimators']})",
            price_transformer,
        )
        all_model_results.append(gb_result)

    # =========================================================================
    # 3. RIDGE REGRESSION
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. RIDGE REGRESSION (LINEAR REGRESSION WITH L2 REGULARIZATION)")
    print("=" * 80)

    best_ridge_model = None
    best_ridge_alpha = None
    best_ridge_val_rmse = float("inf")

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"\nHyperparameter sweep: alpha={alphas}")
    print(f"{'alpha':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
    print("-" * 70)

    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train, y_train)
        y_val_pred = ridge.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        print(f"{alpha:<12.4f} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")

        if val_rmse < best_ridge_val_rmse:
            best_ridge_val_rmse = val_rmse
            best_ridge_model = ridge
            best_ridge_alpha = alpha

    print("-" * 70)
    print(f"✅ Best Ridge: alpha={best_ridge_alpha}")

    ridge_result = evaluate_model(
        best_ridge_model,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
        f"Ridge Regression (alpha={best_ridge_alpha})",
        price_transformer,
    )
    ridge_result["model_filename"] = _save_sklearn_model(best_ridge_model, ridge_result["model_name"], variant_name, models_dir)
    all_model_results.append(ridge_result)

    # =========================================================================
    # 4. POLYNOMIAL FEATURES + RIDGE
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. POLYNOMIAL FEATURES (DEGREE 2/3) + RIDGE")
    print("=" * 80)

    best_poly_model = None
    best_poly_params = {}
    best_poly_val_rmse = float("inf")

    poly_degrees = [2, 3]
    poly_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    print(f"\nHyperparameter sweep: degree={poly_degrees}, alpha={poly_alphas}")
    print(f"{'degree':<12} {'alpha':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
    print("-" * 70)

    for degree in poly_degrees:
        for alpha in poly_alphas:
            poly_model = Pipeline(
                [("poly_features", PolynomialFeatures(degree=degree, include_bias=False)), ("ridge", Ridge(alpha=alpha, random_state=42))]
            )
            poly_model.fit(X_train, y_train)
            y_val_pred = poly_model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            print(f"{degree:<12} {alpha:<12.4f} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")

            if val_rmse < best_poly_val_rmse:
                best_poly_val_rmse = val_rmse
                best_poly_model = poly_model
                best_poly_params = {"degree": degree, "alpha": alpha}

    print("-" * 70)
    print(f"✅ Best Polynomial+Ridge: degree={best_poly_params['degree']}, alpha={best_poly_params['alpha']}")

    poly_result = evaluate_model(
        best_poly_model,
        X_val,
        y_val,
        y_val_raw,
        X_test,
        y_test,
        y_test_raw,
        f"Polynomial (degree={best_poly_params['degree']}) + Ridge (alpha={best_poly_params['alpha']})",
        price_transformer,
    )
    poly_result["model_filename"] = _save_sklearn_model(best_poly_model, poly_result["model_name"], variant_name, models_dir)
    all_model_results.append(poly_result)

    # =========================================================================
    # 5/6. DEEP LEARNING (MLP)
    # =========================================================================
    if ENABLE_DEEP_LEARNING:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        # Embedding sizes: upstream categoricals are 0..K-1 in train; we shift by +1 and reserve 0 for unknown.
        room_max = int(pd.to_numeric(train_df["room_type"], errors="coerce").fillna(-1).max())
        neigh_max = int(pd.to_numeric(train_df["neighbourhood_cleansed"], errors="coerce").fillna(-1).max())
        prop_max = int(pd.to_numeric(train_df["property_type"], errors="coerce").fillna(-1).max())
        inst_max = int(pd.to_numeric(train_df["instant_bookable"], errors="coerce").fillna(-1).max())

        num_room_types = max(room_max, 0) + 2
        num_neighbourhoods = max(neigh_max, 0) + 2
        num_property_types = max(prop_max, 0) + 2
        num_instant_bookable = max(inst_max, 0) + 2

        train_dlset = AirbnbTabularDataset(train_df)
        val_dlset = AirbnbTabularDataset(val_df)
        test_dlset = AirbnbTabularDataset(test_df)

        train_loader = DataLoader(train_dlset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dlset, batch_size=256, shuffle=False)

        print("\n" + "=" * 80)
        print("5. DEEP LEARNING - MLP (SIMPLE)")
        print("=" * 80)
        print("Architecture: 2 hidden layers [64, 32], embeddings [4, 8, 8, 2]")

        mlp_simple = TabularMLP(
            num_numeric=len(NUMERIC_COLS_DL),
            num_room_types=num_room_types,
            num_neighbourhoods=num_neighbourhoods,
            num_property_types=num_property_types,
            num_instant_bookable=num_instant_bookable,
            hidden_dims=[64, 32],
            embedding_dims=[4, 8, 8, 2],
            dropout_rate=0.2,
        ).to(device)

        mlp_simple = train_torch_model(mlp_simple, train_loader, val_loader, device=device, epochs=150, lr=0.005, patience=20)
        mlp_simple_result = evaluate_torch_model(
            mlp_simple,
            test_dlset,
            y_test_raw,
            "MLP (Simple)",
            price_transformer,
            device=device,
        )
        mlp_simple_result["model_name"] = "MLP (Simple) [64,32] embeddings=[4,8,8,2]"
        mlp_simple_result["best_params"] = "layers=[64,32], embeddings=[4,8,8,2], lr=0.005, dropout=0.2"
        mlp_simple_result["model_filename"] = _save_torch_model(mlp_simple, mlp_simple_result["model_name"], variant_name, models_dir)
        all_model_results.append(mlp_simple_result)

        print("\n" + "=" * 80)
        print("6. DEEP LEARNING - MLP (SOPHISTICATED)")
        print("=" * 80)
        print("Architecture: 4 hidden layers [256, 128, 64, 32], embeddings [8, 16, 16, 4]")

        mlp_sophisticated = TabularMLP(
            num_numeric=len(NUMERIC_COLS_DL),
            num_room_types=num_room_types,
            num_neighbourhoods=num_neighbourhoods,
            num_property_types=num_property_types,
            num_instant_bookable=num_instant_bookable,
            hidden_dims=[256, 128, 64, 32],
            embedding_dims=[8, 16, 16, 4],
            dropout_rate=0.4,
        ).to(device)

        mlp_sophisticated = train_torch_model(
            mlp_sophisticated, train_loader, val_loader, device=device, epochs=150, lr=0.001, patience=20
        )
        mlp_sophisticated_result = evaluate_torch_model(
            mlp_sophisticated,
            test_dlset,
            y_test_raw,
            "MLP (Sophisticated)",
            price_transformer,
            device=device,
        )
        mlp_sophisticated_result["model_name"] = "MLP (Sophisticated) [256,128,64,32] embeddings=[8,16,16,4]"
        mlp_sophisticated_result["best_params"] = "layers=[256,128,64,32], embeddings=[8,16,16,4], lr=0.001, dropout=0.4"
        mlp_sophisticated_result["model_filename"] = _save_torch_model(mlp_sophisticated, mlp_sophisticated_result["model_name"], variant_name, models_dir)
        all_model_results.append(mlp_sophisticated_result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: ALL MODELS COMPARISON ON TEST SET (RAW DOLLARS)")
    print("=" * 80)

    summary_df = pd.DataFrame(all_model_results)
    # Some DL rows only include test metrics
    for col in ["val_rmse_raw", "val_mae_raw", "val_r2_raw", "test_r2_bc"]:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    if {"model_name", "test_rmse_raw", "test_mae_raw", "test_r2_raw"}.issubset(summary_df.columns):
        print(summary_df[["model_name", "test_rmse_raw", "test_mae_raw", "test_r2_raw"]].to_string(index=False))

    best_idx = summary_df["test_r2_raw"].astype(float).idxmax()
    best_model_name = summary_df.loc[best_idx, "model_name"]
    best_test_rmse = float(summary_df.loc[best_idx, "test_rmse_raw"])
    best_test_r2 = float(summary_df.loc[best_idx, "test_r2_raw"])

    print("\n" + "=" * 80)
    print("BEST MODEL BY TEST R² (raw dollars):")
    print(f"✅ {best_model_name}")
    print(f"   Test RMSE: ${best_test_rmse:.2f}")
    print(f"   Test R²: {best_test_r2:.4f}")
    print("=" * 80)

    summary_df["dataset_variant"] = variant_name
    summary_df["timestamp"] = datetime.now().isoformat()
    summary_df["train_size"] = len(X_train)
    summary_df["val_size"] = len(X_val)
    summary_df["test_size"] = len(X_test)

    return summary_df


def main() -> None:
    data_dir = Path("data")

    variants = [
        (None, "normal"),
        ("cleaned", "cleaned"),
    ]

    all_summaries: List[pd.DataFrame] = []

    # Create models directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for tag, name in variants:
        try:
            all_summaries.append(run_variant(data_dir, tag, name, models_dir))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing artifacts for variant '{name}' (tag={tag}). "
                f"Run scripts/data_processor.py with split_and_export_both() (or split_and_export(file_tag='cleaned')). "
                f"Original error: {e}"
            )

    combined_df = pd.concat(all_summaries, ignore_index=True)

    csv_path = output_dir / "model_runs.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined_df = pd.concat([existing, combined_df], ignore_index=True)

    combined_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")


if __name__ == "__main__":
    main()
