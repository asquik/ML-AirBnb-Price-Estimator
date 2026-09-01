"""Shared constants and utility functions for all training scripts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── column groups ──────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable",
]

NUMERIC_COLS = [
    "accommodates", "bathrooms", "bedrooms", "beds", "host_total_listings_count",
    "latitude", "longitude", "minimum_nights", "availability_365",
    "number_of_reviews", "season_ordinal", "has_valid_image",
]

KEYWORD_COLS = [
    "kw_metro", "kw_parking", "kw_wifi", "kw_kitchen", "kw_washer",
    "kw_gym", "kw_pool", "kw_balcony", "kw_air_conditioning", "kw_near_park",
    "kw_near_bars", "kw_downtown", "kw_near_university", "kw_near_airport",
    "kw_pet_friendly", "kw_family", "kw_luxury", "kw_new", "kw_quiet", "kw_view",
]

# 16-column base feature set. Used by LoRA and MLP models that process
# text/images directly and don't need keyword binary features.
TABULAR_BASE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# 36-column feature set for sklearn models (adds keyword binary features
# extracted from listing descriptions by data_processor.py).
SKLEARN_FEATURE_COLS = TABULAR_BASE_COLS + KEYWORD_COLS

# ── metric functions ───────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R² for raw-dollar arrays."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def to_raw_dollars(preds: np.ndarray, price_transformer) -> np.ndarray:
    """Invert Box-Cox transform to raw CAD prices; pass None to return preds unchanged."""
    if price_transformer is None:
        return preds.astype(np.float32, copy=False)
    preds_safe = preds.astype(np.float64, copy=True)
    lam = float(price_transformer.lambdas_[0])
    eps = 1e-6
    if lam < 0:
        preds_safe = np.minimum(preds_safe, (-1.0 / lam) - eps)
    elif lam > 0:
        preds_safe = np.maximum(preds_safe, (-1.0 / lam) + eps)
    raw = price_transformer.inverse_transform(preds_safe.reshape(-1, 1)).ravel()
    return np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32, copy=False)
