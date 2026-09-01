"""Inference engine — loaded once at app startup."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .schemas import ListingInput

# Mirrors training_utils constants — must stay in sync with SKLEARN_FEATURE_COLS.
_CATEGORICAL_COLS = [
    "room_type", "neighbourhood_cleansed", "property_type", "instant_bookable",
]
_NUMERIC_10_COLS = [
    # exactly the 10 columns fitted by numeric_imputer / numeric_scaler
    "accommodates", "bathrooms", "bedrooms", "beds", "host_total_listings_count",
    "latitude", "longitude", "minimum_nights", "availability_365", "number_of_reviews",
]
_KEYWORD_COLS = [
    "kw_metro", "kw_parking", "kw_wifi", "kw_kitchen", "kw_washer",
    "kw_gym", "kw_pool", "kw_balcony", "kw_air_conditioning", "kw_near_park",
    "kw_near_bars", "kw_downtown", "kw_near_university", "kw_near_airport",
    "kw_pet_friendly", "kw_family", "kw_luxury", "kw_new", "kw_quiet", "kw_view",
]

_SEASON_MAP = {"Winter": 1, "Spring": 2, "Summer": 3}

# Model RMSE from the test set — used to build the price range band.
_MODEL_RMSE = 144.0


class AirbnbPredictor:
    def __init__(self, model_path: Path, data_dir: Path) -> None:
        self.model = joblib.load(model_path)
        self._encoders: dict[str, dict[str, int]] = joblib.load(
            data_dir / "tabular_encoders.joblib"
        )
        self._imputer: dict[str, float] = joblib.load(
            data_dir / "numeric_imputer.joblib"
        )
        self._scaler = joblib.load(data_dir / "numeric_scaler.joblib")
        self._keywords: dict[str, list[str]] = joblib.load(
            data_dir / "keyword_features.joblib"
        )
        self._price_transformer = joblib.load(
            data_dir / "price_transformer.joblib"
        )

    @property
    def neighbourhood_vocab(self) -> list[str]:
        return sorted(self._encoders.get("neighbourhood_cleansed", {}).keys())

    def predict(self, listing: ListingInput) -> dict[str, Any]:
        keywords_detected = self._extract_keywords(listing.description)
        feature_vec = self._build_features(listing, keywords_detected)

        raw_pred = self.model.predict(feature_vec.reshape(1, -1))
        price = float(self._invert_box_cox(raw_pred)[0])
        price = max(0.0, round(price, 2))

        return {
            "predicted_price_cad": price,
            "price_range": {
                "low": max(0.0, round(price - _MODEL_RMSE, 2)),
                "high": round(price + _MODEL_RMSE, 2),
            },
            "keywords_detected": keywords_detected,
        }

    def _extract_keywords(self, description: str) -> list[str]:
        lowered = description.lower()
        detected = []
        for feature_name, variants in self._keywords.items():
            pattern = "|".join(re.escape(v) for v in variants)
            if re.search(pattern, lowered):
                detected.append(feature_name)
        return detected

    def _build_features(
        self, listing: ListingInput, keywords_detected: list[str]
    ) -> np.ndarray:
        # 1. Encode categoricals — unknown values map to 0
        cat_values = []
        for col in _CATEGORICAL_COLS:
            raw = _listing_categorical(listing, col)
            mapping = self._encoders.get(col, {})
            cat_values.append(float(mapping.get(raw, 0)))

        # 2. Impute + scale the 10 numeric columns
        numeric_raw = np.array(
            [_listing_numeric(listing, col, self._imputer.get(col, 0.0))
             for col in _NUMERIC_10_COLS],
            dtype=np.float64,
        ).reshape(1, -1)
        numeric_scaled = self._scaler.transform(numeric_raw).ravel()

        # 3. Pass-through features (not scaled)
        season_ordinal = float(_SEASON_MAP[listing.season])
        has_valid_image = 1.0 if listing.has_valid_image else 0.0

        # 4. Keyword binary features
        kw_values = [1.0 if kw in keywords_detected else 0.0 for kw in _KEYWORD_COLS]

        # 5. Assemble in SKLEARN_FEATURE_COLS order:
        #    categorical (4) | numeric_scaled (10) | season_ordinal | has_valid_image | keywords (20)
        return np.array(
            cat_values + list(numeric_scaled) + [season_ordinal, has_valid_image] + kw_values,
            dtype=np.float64,
        )

    def _invert_box_cox(self, preds: np.ndarray) -> np.ndarray:
        pt = self._price_transformer
        preds_safe = preds.astype(np.float64, copy=True)
        lam = float(pt.lambdas_[0])
        eps = 1e-6
        if lam < 0:
            preds_safe = np.minimum(preds_safe, (-1.0 / lam) - eps)
        elif lam > 0:
            preds_safe = np.maximum(preds_safe, (-1.0 / lam) + eps)
        raw = pt.inverse_transform(preds_safe.reshape(-1, 1)).ravel()
        return np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)


def _listing_categorical(listing: ListingInput, col: str) -> str:
    mapping = {
        "room_type": listing.room_type,
        "neighbourhood_cleansed": listing.neighbourhood,
        "property_type": listing.property_type,
        "instant_bookable": "t" if listing.instant_bookable else "f",
    }
    return str(mapping.get(col, ""))


def _listing_numeric(listing: ListingInput, col: str, fallback: float) -> float:
    mapping = {
        "accommodates": listing.accommodates,
        "bathrooms": listing.bathrooms,
        "bedrooms": listing.bedrooms,
        "beds": listing.beds,
        "host_total_listings_count": listing.host_total_listings_count,
        "latitude": listing.latitude,
        "longitude": listing.longitude,
        "minimum_nights": listing.minimum_nights,
        "availability_365": listing.availability_365,
        "number_of_reviews": listing.number_of_reviews,
    }
    val = mapping.get(col)
    if val is None:
        return fallback
    return float(val)
