from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ListingInput(BaseModel):
    room_type: Literal["Entire home/apt", "Private room", "Hotel room", "Shared room"]
    neighbourhood: str = Field(..., description="Montreal neighbourhood name")
    accommodates: int = Field(2, ge=1, le=16)
    bathrooms: float = Field(1.0, ge=0.0, le=10.0)
    bedrooms: int = Field(1, ge=0, le=10)
    beds: int = Field(1, ge=0, le=20)
    property_type: str = Field("Apartment")
    instant_bookable: bool = Field(False)
    host_total_listings_count: int = Field(1, ge=1, le=500)
    latitude: float = Field(..., ge=45.4, le=45.7, description="Montreal bbox")
    longitude: float = Field(..., ge=-73.95, le=-73.47, description="Montreal bbox")
    minimum_nights: int = Field(2, ge=1, le=365)
    availability_365: int = Field(180, ge=0, le=365)
    number_of_reviews: int = Field(0, ge=0, le=5000)
    season: Literal["Winter", "Spring", "Summer"] = "Summer"
    description: str = Field("", max_length=2000)
    has_valid_image: bool = Field(True)

    @field_validator("neighbourhood")
    @classmethod
    def neighbourhood_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("neighbourhood cannot be empty")
        return v.strip()


class PredictionOutput(BaseModel):
    predicted_price_cad: float
    price_range: dict[str, float]  # {"low": ..., "high": ...}
    confidence_note: str
    keywords_detected: list[str]
    warnings: list[str]
    model_version: str
