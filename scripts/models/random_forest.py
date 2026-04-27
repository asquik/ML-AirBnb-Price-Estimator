"""
Random Forest model — runs #5-6 in the model registry.

Usage:
    python scripts/models/random_forest.py --variant normal_raw
    python scripts/models/random_forest.py --variant cleaned_raw
    python scripts/models/random_forest.py --variant normal_raw --smoke-test

See Model Training Specification.md for the full data contract.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracker import ExperimentTracker

DATA_DIR = Path("data")

FEATURE_COLS = [
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable",
    "accommodates", "bathrooms", "bedrooms", "beds", "host_total_listings_count",
    "latitude", "longitude", "minimum_nights", "availability_365",
    "number_of_reviews", "season_ordinal", "has_valid_image",
]

# Hyperparameter grid
N_ESTIMATORS_LIST = [100, 200, 400]
MAX_DEPTHS        = [8, 15, 25, None]   # None = unlimited
MIN_SAMPLES_LEAFS = [2, 5, 10]


def load_data(variant: str):
    suffix = "_cleaned" if variant.startswith("cleaned") else ""
    train_df = pd.read_parquet(DATA_DIR / f"train{suffix}_tabular.parquet")
    val_df   = pd.read_parquet(DATA_DIR / f"val{suffix}_tabular.parquet")
    test_df  = pd.read_parquet(DATA_DIR / f"test{suffix}_tabular.parquet")

    price_transformer = None
    if variant.endswith("_bc"):
        price_transformer = joblib.load(DATA_DIR / f"price_transformer{suffix}.joblib")

    return train_df, val_df, test_df, price_transformer


def to_raw_dollars(preds: np.ndarray, price_transformer) -> np.ndarray:
    if price_transformer is None:
        return preds
    return price_transformer.inverse_transform(preds.reshape(-1, 1)).ravel()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest model")
    parser.add_argument(
        "--variant", required=True,
        choices=["normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"],
    )
    parser.add_argument("--run-name", default="", dest="run_name")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    variant    = args.variant
    target_col = "price_bc" if variant.endswith("_bc") else "price"

    print(f"\n{'='*70}")
    print(f"Random Forest  |  variant={variant}  |  target={target_col}")
    print(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_df, val_df, test_df, price_transformer = load_data(variant)

    if args.smoke_test:
        print("  [SMOKE TEST] truncating all splits to 100 rows")
        train_df = train_df.iloc[:100].copy()
        val_df   = val_df.iloc[:100].copy()
        test_df  = test_df.iloc[:100].copy()

    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train     = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train     = train_df[target_col].to_numpy(dtype=np.float64)
    y_train_raw = train_df["price"].to_numpy(dtype=np.float64)
    sw_train    = train_df["sample_weight"].to_numpy(dtype=np.float64)

    X_val     = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_val_raw = val_df["price"].to_numpy(dtype=np.float64)

    X_test     = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test_raw = test_df["price"].to_numpy(dtype=np.float64)

    # -------------------------------------------------------------------------
    # Smoke test: collapse grid to a single fast combination
    # -------------------------------------------------------------------------
    sweep_n_est   = [50]          if args.smoke_test else N_ESTIMATORS_LIST
    sweep_depths  = [5]           if args.smoke_test else MAX_DEPTHS
    sweep_leafs   = [10]          if args.smoke_test else MIN_SAMPLES_LEAFS

    tracker = ExperimentTracker(
        model_type="RandomForest",
        modalities="tabular",
        variant=variant,
        run_name=args.run_name,
        is_smoke_test=args.smoke_test,
        config={
            "n_estimators_searched": sweep_n_est,
            "max_depths_searched":   [str(d) for d in sweep_depths],
            "min_samples_leafs_searched": sweep_leafs,
            "feature_cols": FEATURE_COLS,
            "target_column": target_col,
        },
    )

    if price_transformer is not None:
        tracker.set_box_cox_lambda(float(price_transformer.lambdas_[0]))

    # -------------------------------------------------------------------------
    # Hyperparameter sweep — val set only
    # -------------------------------------------------------------------------
    total_configs = len(sweep_n_est) * len(sweep_depths) * len(sweep_leafs)
    print(f"\nSweeping {total_configs} configurations on val set (n_jobs=-1)...")
    print(
        f"{'n_est':<8} {'max_depth':<12} {'min_leaf':<10} "
        f"{'Val RMSE $':<14} {'Val MAE $':<12} {'Val R²':<8}"
    )
    print("-" * 70)

    best_model = None
    best_params: dict = {}
    best_val_rmse = float("inf")
    t0 = time.time()

    for n_est in sweep_n_est:
        for max_depth in sweep_depths:
            for min_leaf in sweep_leafs:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    n_jobs=-1,
                    random_state=42,
                )
                rf.fit(X_train, y_train, sample_weight=sw_train)

                val_pred_raw = to_raw_dollars(rf.predict(X_val), price_transformer)
                val_metrics  = compute_metrics(y_val_raw, val_pred_raw)

                depth_label = str(max_depth) if max_depth is not None else "None"
                print(
                    f"{n_est:<8} {depth_label:<12} {min_leaf:<10} "
                    f"{val_metrics['rmse']:<14.2f} {val_metrics['mae']:<12.2f} {val_metrics['r2']:<8.4f}"
                )

                if val_metrics["rmse"] < best_val_rmse:
                    best_val_rmse = val_metrics["rmse"]
                    best_model    = rf
                    best_params   = {
                        "n_estimators": n_est,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_leaf,
                    }

    training_time = (time.time() - t0) / 60
    print("-" * 70)
    print(f"\n✅ Best: {best_params}  |  Val RMSE: ${best_val_rmse:.2f}")

    # -------------------------------------------------------------------------
    # Final evaluation on all splits
    # -------------------------------------------------------------------------
    train_pred_raw = to_raw_dollars(best_model.predict(X_train), price_transformer)
    val_pred_raw   = to_raw_dollars(best_model.predict(X_val),   price_transformer)
    test_pred_raw  = to_raw_dollars(best_model.predict(X_test),  price_transformer)

    train_metrics = compute_metrics(y_train_raw, train_pred_raw)
    val_metrics   = compute_metrics(y_val_raw,   val_pred_raw)
    test_metrics  = compute_metrics(y_test_raw,  test_pred_raw)

    print(f"\nFinal metrics (raw $):")
    print(f"  Train — RMSE ${train_metrics['rmse']:.2f}  MAE ${train_metrics['mae']:.2f}  R² {train_metrics['r2']:.4f}")
    print(f"  Val   — RMSE ${val_metrics['rmse']:.2f}  MAE ${val_metrics['mae']:.2f}  R² {val_metrics['r2']:.4f}")
    print(f"  Test  — RMSE ${test_metrics['rmse']:.2f}  MAE ${test_metrics['mae']:.2f}  R² {test_metrics['r2']:.4f}")

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------
    feature_importance = {
        col: round(float(imp), 6)
        for col, imp in sorted(
            zip(FEATURE_COLS, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
    }

    tracker.finish(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions={
            "train_y_true": y_train_raw, "train_y_pred": train_pred_raw,
            "val_y_true":   y_val_raw,   "val_y_pred":   val_pred_raw,
            "test_y_true":  y_test_raw,  "test_y_pred":  test_pred_raw,
        },
        trainable_parameters=0,
        training_time_minutes=training_time,
        best_hyperparams=best_params,
        extra_artifacts={"feature_importance.json": feature_importance},
    )


if __name__ == "__main__":
    main()
