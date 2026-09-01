"""Production guards run after prediction — returns warnings for the response."""
from __future__ import annotations

_PRICE_LOW = 20.0
_PRICE_HIGH = 2000.0


def check_prediction(price: float) -> list[str]:
    warnings: list[str] = []
    if price < _PRICE_LOW:
        warnings.append(
            f"Predicted price (${price:.0f}) is unusually low — verify listing details."
        )
    if price > _PRICE_HIGH:
        warnings.append(
            f"Predicted price (${price:.0f}) is unusually high — verify listing details."
        )
    return warnings
