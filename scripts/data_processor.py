from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler


DEFAULT_CATEGORICAL_COLUMNS = [
    "room_type",
    "neighbourhood_cleansed",
    "property_type",
    "instant_bookable",
]

DEFAULT_NUMERIC_COLUMNS = [
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
]

DEFAULT_TARGET_COLUMN = "price"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_SAMPLE_WEIGHT_COLUMN = "sample_weight"
DEFAULT_LISTING_ID_COLUMN = "listing_id"
DEFAULT_SEASON_COLUMN = "season_ordinal"


def _copy_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _as_path(path_like: Path | str) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _extract_snapshot_month(file_path: Path) -> str:
    match = re.search(r"(?:^|[-_])(\d{2})(?:[-_]|$)", file_path.stem)
    if not match:
        raise ValueError(f"Could not infer snapshot month from file name: {file_path.name}")
    return match.group(1)


def _normalize_known_category(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan", "none", "na", "null"}:
        return None
    return text


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _candidate_image_paths(listing_id: str, raw_image_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for extension in DEFAULT_IMAGE_EXTENSIONS:
        candidates.append(raw_image_dir / f"{listing_id}{extension}")
    candidates.extend(sorted(raw_image_dir.glob(f"{listing_id}.*")))
    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            unique_candidates.append(candidate)
    return unique_candidates


def load_raw_csvs(file_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate raw InsideAirbnb CSV snapshots."""
    frames: list[pd.DataFrame] = []
    for file_path_like in file_paths:
        file_path = _as_path(file_path_like)
        if not file_path.exists():
            raise FileNotFoundError(f"Expected data file not found: {file_path}")
        frame = pd.read_csv(file_path)
        frame = frame.copy()
        frame["snapshot_month"] = _extract_snapshot_month(file_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_listing_id(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Standardize the primary key column to listing_id."""
    result = _copy_df(df)
    if id_column not in result.columns:
        if DEFAULT_LISTING_ID_COLUMN in result.columns:
            result[DEFAULT_LISTING_ID_COLUMN] = result[DEFAULT_LISTING_ID_COLUMN].astype(str)
            return result
        raise KeyError(f"Missing id column: {id_column}")

    result[DEFAULT_LISTING_ID_COLUMN] = result[id_column].astype(str)
    if id_column != DEFAULT_LISTING_ID_COLUMN:
        result = result.drop(columns=[id_column])
    return result


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the price column into numeric dollars."""
    result = _copy_df(df)
    result[DEFAULT_TARGET_COLUMN] = (
        result[DEFAULT_TARGET_COLUMN]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    result[DEFAULT_TARGET_COLUMN] = pd.to_numeric(result[DEFAULT_TARGET_COLUMN], errors="coerce")
    return result


def drop_missing_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing, invalid, or non-positive price values."""
    result = _copy_df(df)
    result = result.dropna(subset=[DEFAULT_TARGET_COLUMN])
    result = result[result[DEFAULT_TARGET_COLUMN] > 0].copy()
    return result


def add_season_ordinal(df: pd.DataFrame, month_column: str) -> pd.DataFrame:
    """Map snapshot months to a fixed ordinal season label."""
    month_to_season = {"03": 1, "3": 1, "06": 2, "6": 2, "09": 3, "9": 3}
    result = _copy_df(df)

    def _map_month(value: object) -> int:
        month_token = str(value).strip()
        if month_token not in month_to_season:
            raise ValueError(f"Unsupported snapshot month value: {value}")
        return month_to_season[month_token]

    result[DEFAULT_SEASON_COLUMN] = result[month_column].map(_map_month).astype(int)
    return result


def create_full_text_column(df: pd.DataFrame, description_column: str, amenities_column: str) -> pd.DataFrame:
    """Concatenate description and amenities using the fixed separator token."""
    result = _copy_df(df)
    description = result[description_column].fillna("").astype(str).str.strip()
    amenities = result[amenities_column].fillna("").astype(str).str.strip()
    result["full_text"] = (description + " [SEP] " + amenities).str.strip()
    return result


def fit_language_detector(description_series: pd.Series) -> dict[str, object]:
    """Learn a lightweight French detection rule from the training descriptions."""
    text = " ".join(description_series.fillna("").astype(str).tolist()).lower()
    markers = {
        " le ",
        " la ",
        " les ",
        " des ",
        " une ",
        " un ",
        " appartement ",
        " centre ",
        " avec ",
        " très ",
        " proche ",
        " lumineux ",
    }
    accent_chars = "àâçéèêëîïôùûüÿœæ"
    if text:
        discovered = {marker for marker in markers if marker.strip() in text}
    else:
        discovered = set()
    return {
        "markers": sorted(discovered or markers),
        "accent_chars": accent_chars,
    }


def apply_language_detector(
    df: pd.DataFrame,
    language_detector_artifact: dict[str, object],
    description_column: str,
) -> pd.DataFrame:
    """Flag rows whose descriptions are likely French."""
    result = _copy_df(df)
    markers = [str(marker) for marker in language_detector_artifact.get("markers", [])]
    accent_chars = str(language_detector_artifact.get("accent_chars", ""))

    def _is_french(text: object) -> bool:
        normalized = f" {str(text).lower()} "
        if any(marker in normalized for marker in markers):
            return True
        return any(character in normalized for character in accent_chars)

    result["is_french"] = result[description_column].map(_is_french).astype(bool)
    return result


def mark_image_availability(
    df: pd.DataFrame,
    raw_image_dir: Path,
    listing_id_column: str,
) -> pd.DataFrame:
    """Set a boolean flag when a raw listing image exists on disk."""
    result = _copy_df(df)
    raw_image_dir = _as_path(raw_image_dir)

    def _has_image(listing_id: object) -> bool:
        listing_token = str(listing_id)
        for candidate in _candidate_image_paths(listing_token, raw_image_dir):
            if candidate.exists() and candidate.is_file():
                return True
        return False

    result["has_valid_image"] = result[listing_id_column].map(_has_image).astype(bool)
    return result


def split_data_80_10_10(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministically split the dataframe into train, validation, and test partitions."""
    if DEFAULT_LISTING_ID_COLUMN not in df.columns:
        raise KeyError(f"Missing required split key column: {DEFAULT_LISTING_ID_COLUMN}")

    working_df = _copy_df(df)
    unique_ids = pd.Index(working_df[DEFAULT_LISTING_ID_COLUMN].astype(str).dropna().unique())
    shuffled_ids = np.array(unique_ids, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_ids)

    total_ids = len(shuffled_ids)
    if total_ids == 0:
        empty = working_df.iloc[0:0].copy()
        return empty, empty.copy(), empty.copy()

    raw_counts = np.array([0.8, 0.1, 0.1], dtype=float) * total_ids
    counts = np.floor(raw_counts).astype(int)
    remainder = total_ids - int(counts.sum())
    if remainder > 0:
        fractional_parts = raw_counts - counts
        order = np.argsort(-fractional_parts)
        for index in order[:remainder]:
            counts[index] += 1

    train_count, val_count, test_count = counts.tolist()
    train_ids = shuffled_ids[:train_count]
    val_ids = shuffled_ids[train_count : train_count + val_count]
    test_ids = shuffled_ids[train_count + val_count : train_count + val_count + test_count]

    train_df = working_df[working_df[DEFAULT_LISTING_ID_COLUMN].astype(str).isin(set(train_ids))].copy()
    val_df = working_df[working_df[DEFAULT_LISTING_ID_COLUMN].astype(str).isin(set(val_ids))].copy()
    test_df = working_df[working_df[DEFAULT_LISTING_ID_COLUMN].astype(str).isin(set(test_ids))].copy()

    return train_df, val_df, test_df


def fit_box_cox_transformer(target_series: pd.Series) -> PowerTransformer:
    """Learn the Box-Cox parameters from the training target only."""
    numeric_target = _coerce_numeric_series(target_series).dropna()
    if numeric_target.empty:
        raise ValueError("Box-Cox fitting requires at least one positive target value")
    if (numeric_target <= 0).any():
        raise ValueError("Box-Cox fitting requires strictly positive targets")
    transformer = PowerTransformer(method="box-cox", standardize=False)
    transformer.fit(numeric_target.to_numpy().reshape(-1, 1))
    return transformer


def apply_box_cox_transformer(target_series: pd.Series, box_cox_transformer: PowerTransformer) -> pd.Series:
    """Transform the target into Box-Cox space using the fitted transformer."""
    numeric_target = _coerce_numeric_series(target_series)
    if numeric_target.empty:
        return pd.Series(dtype=float, index=target_series.index, name=target_series.name)
    transformed = box_cox_transformer.transform(numeric_target.to_numpy().reshape(-1, 1)).ravel()
    return pd.Series(transformed, index=target_series.index, name=target_series.name)


def fit_numeric_imputer(train_df: pd.DataFrame, numeric_columns: list[str]) -> Dict[str, float]:
    """Learn train-set medians for numeric missing-value imputation."""
    medians: Dict[str, float] = {}
    for column in numeric_columns:
        numeric_values = _coerce_numeric_series(train_df[column])
        median_value = float(numeric_values.median()) if not numeric_values.dropna().empty else 0.0
        if pd.isna(median_value):
            median_value = 0.0
        medians[column] = median_value
    return medians


def apply_numeric_imputer(
    df: pd.DataFrame,
    numeric_imputer_artifact: Dict[str, float],
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Fill missing numeric values with train medians."""
    result = _copy_df(df)
    for column in numeric_columns:
        result[column] = _coerce_numeric_series(result[column]).fillna(numeric_imputer_artifact[column])
    return result


def fit_categorical_encoder(train_df: pd.DataFrame, categorical_columns: list[str]) -> Dict[str, Dict[str, int]]:
    """Build train-only category vocabularies with 0 reserved for unknown values."""
    encoders: Dict[str, Dict[str, int]] = {}
    for column in categorical_columns:
        values = []
        for value in train_df[column]:
            normalized = _normalize_known_category(value)
            if normalized is not None:
                values.append(normalized)
        unique_values = sorted(set(values))
        encoders[column] = {value: index + 1 for index, value in enumerate(unique_values)}
    return encoders


def apply_categorical_encoder(
    df: pd.DataFrame,
    categorical_encoder_artifact: Dict[str, Dict[str, int]],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Convert categorical values to integer ids using train vocabularies."""
    result = _copy_df(df)
    for column in categorical_columns:
        mapping = categorical_encoder_artifact[column]
        encoded = result[column].map(lambda value: mapping.get(_normalize_known_category(value), 0))
        result[column] = encoded.fillna(0).astype(int)
    return result


def fit_numeric_scaler(train_df: pd.DataFrame, numeric_columns: list[str]) -> StandardScaler:
    """Learn train-only numeric scaling parameters."""
    scaler = StandardScaler()
    numeric_frame = train_df[numeric_columns].apply(_coerce_numeric_series)
    scaler.fit(numeric_frame)
    return scaler


def apply_numeric_scaler(
    df: pd.DataFrame,
    numeric_scaler_artifact: StandardScaler,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Scale numeric features using the train-fitted scaler."""
    result = _copy_df(df)
    numeric_frame = result[numeric_columns].apply(_coerce_numeric_series)
    if len(numeric_frame) == 0:
        return result
    transformed = numeric_scaler_artifact.transform(numeric_frame)
    transformed_frame = pd.DataFrame(transformed, columns=numeric_columns, index=result.index)
    for column in numeric_columns:
        result[column] = transformed_frame[column].astype(float)
    return result


def fit_room_type_weights(train_df: pd.DataFrame, room_type_column: str) -> Dict[str, float]:
    """Compute inverse-frequency weights for each room type from training data only."""
    room_type_series = train_df[room_type_column].map(_normalize_known_category)
    room_type_series = room_type_series.dropna()
    counts = room_type_series.value_counts(dropna=True)
    total_rows = float(len(room_type_series))
    unique_count = float(len(counts)) if len(counts) else 1.0
    return {room_type: total_rows / (unique_count * float(count)) for room_type, count in counts.items()}


def apply_room_type_weights(
    df: pd.DataFrame,
    room_type_weight_map: Dict[str, float],
    room_type_column: str,
) -> pd.DataFrame:
    """Assign a per-row sample weight using the learned room-type frequency map."""
    result = _copy_df(df)
    default_weight = float(np.mean(list(room_type_weight_map.values()))) if room_type_weight_map else 1.0
    result[DEFAULT_SAMPLE_WEIGHT_COLUMN] = result[room_type_column].map(
        lambda value: room_type_weight_map.get(_normalize_known_category(value), default_weight)
    ).astype(float)
    return result


def assemble_tabular_output(
    df: pd.DataFrame,
    target_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Select and order the final columns for tabular export."""
    priority_columns = [
        DEFAULT_LISTING_ID_COLUMN,
        *target_columns,
        DEFAULT_SAMPLE_WEIGHT_COLUMN,
        "has_valid_image",
        "is_french",
        "full_text",
        *feature_columns,
    ]
    ordered_columns: List[str] = []
    for column in priority_columns:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)
    return _copy_df(df[ordered_columns])


def filter_cleaned_variant(df: pd.DataFrame, max_price: float = 1000) -> pd.DataFrame:
    """Remove rows with prices outside the strict cleaned comparison band."""
    result = _copy_df(df)
    return result[result[DEFAULT_TARGET_COLUMN].between(50, max_price, inclusive="both")].copy()


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Persist a dataframe to parquet without changing row order."""
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="gzip", index=False)


def save_artifact(artifact: object, output_path: Path) -> None:
    """Persist a fitted preprocessing artifact for reuse."""
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def build_base_frame(file_paths: list[Path], raw_image_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create the shared raw dataframe used by both the normal and cleaned variants."""
    raw_df = load_raw_csvs(file_paths)
    raw_df = normalize_listing_id(raw_df, id_column="id")
    raw_df = clean_price_column(raw_df)
    raw_df = drop_missing_price_rows(raw_df)
    raw_df = add_season_ordinal(raw_df, month_column="snapshot_month")
    raw_df = create_full_text_column(raw_df, description_column="description", amenities_column="amenities")
    language_detector_artifact = fit_language_detector(raw_df["description"])
    raw_df = apply_language_detector(raw_df, language_detector_artifact, description_column="description")
    raw_df = mark_image_availability(raw_df, raw_image_dir=raw_image_dir, listing_id_column=DEFAULT_LISTING_ID_COLUMN)
    return raw_df, {"language_detector": language_detector_artifact}


def _split_and_export_variant(
    variant_df: pd.DataFrame,
    output_dir: Path,
    file_tag: Optional[str],
) -> dict[str, object]:
    train_df, val_df, test_df = split_data_80_10_10(variant_df, seed=42)

    room_type_weight_map = fit_room_type_weights(train_df, room_type_column="room_type")
    box_cox_transformer = fit_box_cox_transformer(train_df[DEFAULT_TARGET_COLUMN])
    numeric_imputer_artifact = fit_numeric_imputer(train_df, numeric_columns=DEFAULT_NUMERIC_COLUMNS)

    train_df = apply_numeric_imputer(train_df, numeric_imputer_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)
    val_df = apply_numeric_imputer(val_df, numeric_imputer_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)
    test_df = apply_numeric_imputer(test_df, numeric_imputer_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)

    numeric_scaler_artifact = fit_numeric_scaler(train_df, numeric_columns=DEFAULT_NUMERIC_COLUMNS)
    train_df = apply_numeric_scaler(train_df, numeric_scaler_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)
    val_df = apply_numeric_scaler(val_df, numeric_scaler_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)
    test_df = apply_numeric_scaler(test_df, numeric_scaler_artifact, numeric_columns=DEFAULT_NUMERIC_COLUMNS)

    categorical_encoder_artifact = fit_categorical_encoder(train_df, categorical_columns=DEFAULT_CATEGORICAL_COLUMNS)

    train_df["price_bc"] = apply_box_cox_transformer(train_df[DEFAULT_TARGET_COLUMN], box_cox_transformer)
    val_df["price_bc"] = apply_box_cox_transformer(val_df[DEFAULT_TARGET_COLUMN], box_cox_transformer)
    test_df["price_bc"] = apply_box_cox_transformer(test_df[DEFAULT_TARGET_COLUMN], box_cox_transformer)

    train_df = apply_room_type_weights(train_df, room_type_weight_map, room_type_column="room_type")
    val_df = apply_room_type_weights(val_df, room_type_weight_map, room_type_column="room_type")
    test_df = apply_room_type_weights(test_df, room_type_weight_map, room_type_column="room_type")

    train_df = apply_categorical_encoder(train_df, categorical_encoder_artifact, categorical_columns=DEFAULT_CATEGORICAL_COLUMNS)
    val_df = apply_categorical_encoder(val_df, categorical_encoder_artifact, categorical_columns=DEFAULT_CATEGORICAL_COLUMNS)
    test_df = apply_categorical_encoder(test_df, categorical_encoder_artifact, categorical_columns=DEFAULT_CATEGORICAL_COLUMNS)

    feature_columns = [
        *DEFAULT_CATEGORICAL_COLUMNS,
        *DEFAULT_NUMERIC_COLUMNS,
        DEFAULT_SEASON_COLUMN,
    ]

    train_tabular = assemble_tabular_output(
        train_df,
        target_columns=[DEFAULT_TARGET_COLUMN, "price_bc"],
        feature_columns=feature_columns,
    )
    val_tabular = assemble_tabular_output(
        val_df,
        target_columns=[DEFAULT_TARGET_COLUMN, "price_bc"],
        feature_columns=feature_columns,
    )
    test_tabular = assemble_tabular_output(
        test_df,
        target_columns=[DEFAULT_TARGET_COLUMN, "price_bc"],
        feature_columns=feature_columns,
    )

    suffix = "" if not file_tag else f"_{file_tag}"

    save_parquet(train_df, output_dir / f"train{suffix}.parquet")
    save_parquet(val_df, output_dir / f"val{suffix}.parquet")
    save_parquet(test_df, output_dir / f"test{suffix}.parquet")
    save_parquet(train_tabular, output_dir / f"train{suffix}_tabular.parquet")
    save_parquet(val_tabular, output_dir / f"val{suffix}_tabular.parquet")
    save_parquet(test_tabular, output_dir / f"test{suffix}_tabular.parquet")

    save_artifact(box_cox_transformer, output_dir / f"price_transformer{suffix}.joblib")
    save_artifact(numeric_imputer_artifact, output_dir / f"numeric_imputer{suffix}.joblib")
    save_artifact(categorical_encoder_artifact, output_dir / f"tabular_encoders{suffix}.joblib")
    save_artifact(numeric_scaler_artifact, output_dir / f"numeric_scaler{suffix}.joblib")
    save_artifact(room_type_weight_map, output_dir / f"room_type_weights{suffix}.joblib")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_tabular": train_tabular,
        "val_tabular": val_tabular,
        "test_tabular": test_tabular,
        "artifacts": {
            "box_cox_transformer": box_cox_transformer,
            "numeric_imputer": numeric_imputer_artifact,
            "categorical_encoder": categorical_encoder_artifact,
            "numeric_scaler": numeric_scaler_artifact,
            "room_type_weight_map": room_type_weight_map,
        },
    }


def export_normal_and_cleaned_variants(
    file_paths: list[Path],
    raw_image_dir: Path,
    output_dir: Path,
    cleaned_max_price: float = 1000,
) -> dict[str, dict[str, object]]:
    """Run the full pipeline for both the normal and cleaned variants."""
    base_df, shared_artifacts = build_base_frame(file_paths, raw_image_dir=raw_image_dir)
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_artifact(shared_artifacts["language_detector"], output_dir / "language_detector.joblib")

    normal_outputs = _split_and_export_variant(base_df, output_dir, file_tag=None)
    cleaned_df = filter_cleaned_variant(base_df, max_price=cleaned_max_price)
    cleaned_outputs = _split_and_export_variant(cleaned_df, output_dir, file_tag="cleaned")
    return {
        "shared": shared_artifacts,
        "normal": normal_outputs,
        "cleaned": cleaned_outputs,
    }


def main() -> None:
    """Execute the full data pipeline with the repository defaults."""
    project_root = Path(__file__).resolve().parents[1]
    file_paths = [
        project_root / "listings-03-25.csv",
        project_root / "listings-06-25.csv",
        project_root / "listings-09-25.csv",
    ]
    raw_image_dir = project_root / "images" / "raw"
    output_dir = project_root / "data"
    export_normal_and_cleaned_variants(file_paths, raw_image_dir=raw_image_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()