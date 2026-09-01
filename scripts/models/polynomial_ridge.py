"""
Polynomial Ridge Regression model — runs #13-14 in the model registry.
Applies degree-2 PolynomialFeatures before Ridge regression.

Usage:
    python scripts/models/polynomial_ridge.py --variant normal_raw
    python scripts/models/polynomial_ridge.py --variant cleaned_raw
    python scripts/models/polynomial_ridge.py --variant normal_raw --smoke-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracker import ExperimentTracker
from training_utils import SKLEARN_FEATURE_COLS as FEATURE_COLS, compute_metrics, to_raw_dollars

DATA_DIR = Path("data")

# degree-2 PolynomialFeatures on 36 columns produces ~702 expanded features.
# interaction_only=False is intentional — squared terms capture non-linear price relationships.

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def load_data(variant: str):
    suffix = "_cleaned" if variant.startswith("cleaned") else ""
    train_df = pd.read_parquet(DATA_DIR / f"train{suffix}_tabular.parquet")
    val_df   = pd.read_parquet(DATA_DIR / f"val{suffix}_tabular.parquet")
    test_df  = pd.read_parquet(DATA_DIR / f"test{suffix}_tabular.parquet")
    price_transformer = None
    if variant.endswith("_bc"):
        price_transformer = joblib.load(DATA_DIR / f"price_transformer{suffix}.joblib")
    return train_df, val_df, test_df, price_transformer



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["normal_raw", "normal_bc", "cleaned_raw", "cleaned_bc"])
    parser.add_argument("--run-name",   default="", dest="run_name")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    args = parser.parse_args()

    variant    = args.variant
    target_col = "price_bc" if variant.endswith("_bc") else "price"

    print(f"\n{'='*70}")
    print(f"PolynomialRidge (degree=2)  |  variant={variant}  |  target={target_col}")
    print(f"{'='*70}\n")

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
    X_val       = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_val_raw   = val_df["price"].to_numpy(dtype=np.float64)
    X_test      = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test_raw  = test_df["price"].to_numpy(dtype=np.float64)

    sweep_alphas = [10.0] if args.smoke_test else ALPHAS

    tracker = ExperimentTracker(
        model_type="PolynomialRidge", modalities="tabular", variant=variant,
        run_name=args.run_name, is_smoke_test=args.smoke_test,
        config={"degree": 2, "alphas_searched": sweep_alphas,
                "feature_cols": FEATURE_COLS, "target_column": target_col},
    )
    if price_transformer is not None:
        tracker.set_box_cox_lambda(float(price_transformer.lambdas_[0]))

    # Fit PolynomialFeatures once on train to show the expanded feature count
    poly_probe = PolynomialFeatures(degree=2, include_bias=False)
    poly_probe.fit(X_train)
    print(f"  Expanded features: {X_train.shape[1]} → {poly_probe.transform(X_train[:1]).shape[1]}")

    print(f"\nSweeping {len(sweep_alphas)} alpha values on val set...")
    print(f"{'alpha':<12} {'Val RMSE $':<14} {'Val MAE $':<12} {'Val R²':<8}")
    print("-" * 50)

    best_model, best_params, best_val_rmse = None, {}, float("inf")
    t0 = time.time()

    for alpha in sweep_alphas:
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipeline.fit(X_train, y_train, ridge__sample_weight=sw_train)
        val_pred_raw = to_raw_dollars(pipeline.predict(X_val), price_transformer)
        vm = compute_metrics(y_val_raw, val_pred_raw)
        print(f"{alpha:<12} {vm['rmse']:<14.2f} {vm['mae']:<12.2f} {vm['r2']:<8.4f}")
        if vm["rmse"] < best_val_rmse:
            best_val_rmse = vm["rmse"]
            best_model    = pipeline
            best_params   = {"alpha": alpha, "degree": 2}

    training_time = (time.time() - t0) / 60
    print("-" * 50)
    print(f"\n✅ Best: alpha={best_params['alpha']}  |  Val RMSE: ${best_val_rmse:.2f}")

    train_pred_raw = to_raw_dollars(best_model.predict(X_train), price_transformer)
    val_pred_raw   = to_raw_dollars(best_model.predict(X_val),   pt)
    test_pred_raw  = to_raw_dollars(best_model.predict(X_test),  pt)

    train_m = compute_metrics(y_train_raw, train_pred_raw)
    val_m   = compute_metrics(y_val_raw,   val_pred_raw)
    test_m  = compute_metrics(y_test_raw,  test_pred_raw)

    print(f"\n  Train — RMSE ${train_m['rmse']:.2f}  MAE ${train_m['mae']:.2f}  R² {train_m['r2']:.4f}")
    print(f"  Val   — RMSE ${val_m['rmse']:.2f}  MAE ${val_m['mae']:.2f}  R² {val_m['r2']:.4f}")
    print(f"  Test  — RMSE ${test_m['rmse']:.2f}  MAE ${test_m['mae']:.2f}  R² {test_m['r2']:.4f}")

    tracker.finish(
        train_metrics=train_m, val_metrics=val_m, test_metrics=test_m,
        predictions={
            "train_y_true": y_train_raw, "train_y_pred": train_pred_raw,
            "val_y_true":   y_val_raw,   "val_y_pred":   val_pred_raw,
            "test_y_true":  y_test_raw,  "test_y_pred":  test_pred_raw,
        },
        trainable_parameters=0, training_time_minutes=training_time,
        best_hyperparams=best_params,
    )


if __name__ == "__main__":
    main()
