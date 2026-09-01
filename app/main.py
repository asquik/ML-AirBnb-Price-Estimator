"""FastAPI application — serves the Montreal Airbnb price estimator."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .checks import check_prediction
from .predictor import AirbnbPredictor
from .schemas import ListingInput, PredictionOutput

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
_MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/runs/latest/model.joblib"))
_MODEL_VERSION = "lgbm_v1"

_predictor: AirbnbPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    if _MODEL_PATH.exists() and _DATA_DIR.exists():
        try:
            _predictor = AirbnbPredictor(_MODEL_PATH, _DATA_DIR)
            print(f"Model loaded from {_MODEL_PATH}")
        except Exception as exc:
            print(f"WARNING: model load failed — {exc}")
    else:
        print(
            f"WARNING: model not found at {_MODEL_PATH} or data dir {_DATA_DIR} missing. "
            "Run scripts/models/lightgbm_model.py first, then set MODEL_PATH env var."
        )
    yield
    _predictor = None


app = FastAPI(
    title="Montreal Airbnb Price Estimator",
    description="Predict nightly rental prices for Montreal Airbnb listings.",
    version="1.0.0",
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "version": _MODEL_VERSION,
    }


@app.get("/neighbourhoods")
def neighbourhoods():
    if _predictor is None:
        return {"neighbourhoods": []}
    return {"neighbourhoods": _predictor.neighbourhood_vocab}


@app.post("/predict", response_model=PredictionOutput)
def predict(listing: ListingInput):
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Start the server after running the LightGBM trainer.",
        )

    neighbourhood_vocab = set(_predictor.neighbourhood_vocab)
    neighbourhood_warnings: list[str] = []
    if listing.neighbourhood not in neighbourhood_vocab:
        neighbourhood_warnings.append(
            f"Neighbourhood '{listing.neighbourhood}' was not seen during training — "
            "using fallback encoding. Prediction may be less accurate."
        )

    try:
        result = _predictor.predict(listing)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    price = result["predicted_price_cad"]
    price_warnings = check_prediction(price)

    n_kw = len(result["keywords_detected"])
    confidence_note = (
        f"Based on {n_kw} amenity keyword{'s' if n_kw != 1 else ''} detected in description. "
        f"Typical error ±${144:.0f} CAD (test RMSE)."
    )

    return PredictionOutput(
        predicted_price_cad=price,
        price_range=result["price_range"],
        confidence_note=confidence_note,
        keywords_detected=result["keywords_detected"],
        warnings=neighbourhood_warnings + price_warnings,
        model_version=_MODEL_VERSION,
    )
