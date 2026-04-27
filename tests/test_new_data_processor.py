import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.new_data_processor import (
    add_season_ordinal,
    apply_box_cox_transformer,
    apply_categorical_encoder,
    apply_language_detector,
    apply_numeric_imputer,
    apply_numeric_scaler,
    apply_room_type_weights,
    assemble_tabular_output,
    clean_price_column,
    create_full_text_column,
    drop_missing_price_rows,
    export_normal_and_cleaned_variants,
    filter_cleaned_variant,
    fit_box_cox_transformer,
    fit_categorical_encoder,
    fit_language_detector,
    fit_numeric_imputer,
    fit_numeric_scaler,
    fit_room_type_weights,
    load_raw_csvs,
    mark_image_availability,
    normalize_listing_id,
    save_artifact,
    save_parquet,
    split_data_80_10_10,
)


@pytest.fixture
def tiny_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "listing_id": ["1", "2", "3"],
            "price": ["$120.00", "$1,234.50", "150"],
            "description": ["Appartement tres lumineux", "Cozy studio", None],
            "amenities": ["WiFi, Kitchen", None, "Desk"],
            "snapshot_month": ["03", "06", "09"],
            "room_type": ["Entire home/apt", "Private room", "Shared room"],
            "neighbourhood_cleansed": ["Ville-Marie", "Le Plateau-Mont-Royal", "Griffintown"],
            "property_type": ["Apartment", "Condominium", "Loft"],
            "instant_bookable": ["t", "f", "t"],
            "accommodates": [2, 4, 6],
            "bathrooms": [1.0, np.nan, 1.5],
            "bedrooms": [1, 2, 3],
            "beds": [1, np.nan, 2],
            "host_total_listings_count": [3, 1, 2],
            "latitude": [45.5, 45.6, 45.7],
            "longitude": [-73.5, -73.6, -73.7],
            "minimum_nights": [2, 30, 365],
            "availability_365": [120, 90, 60],
            "number_of_reviews": [10, 20, 30],
        }
    )


@pytest.fixture
def split_ready_df():
    rows = []
    for listing_id in range(1, 13):
        for month in ["03", "06", "09"]:
            rows.append(
                {
                    "listing_id": str(listing_id),
                    "price": 100 + listing_id,
                    "description": f"desc {listing_id}",
                    "amenities": f"amenity {listing_id}",
                    "snapshot_month": month,
                    "room_type": ["Entire home/apt", "Private room", "Shared room"][listing_id % 3],
                    "neighbourhood_cleansed": ["Ville-Marie", "Le Plateau-Mont-Royal", "Griffintown"][listing_id % 3],
                    "property_type": ["Apartment", "Condominium", "Loft"][listing_id % 3],
                    "instant_bookable": ["t", "f", "t"][listing_id % 3],
                    "accommodates": listing_id % 6 + 1,
                    "bathrooms": 1.0 + (listing_id % 3) * 0.5,
                    "bedrooms": 1 + (listing_id % 4),
                    "beds": 1 + (listing_id % 2),
                    "host_total_listings_count": 1 + (listing_id % 5),
                    "latitude": 45.5 + listing_id * 0.001,
                    "longitude": -73.5 - listing_id * 0.001,
                    "minimum_nights": 1 + (listing_id % 7),
                    "availability_365": 30 + listing_id,
                    "number_of_reviews": listing_id * 2,
                    "season_ordinal": (listing_id % 3) + 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def csv_snapshot_paths(tmp_path, split_ready_df):
    paths = []
    for month in ["03", "06", "09"]:
        month_df = split_ready_df[split_ready_df["snapshot_month"] == month].copy()
        path = tmp_path / f"listings-{month}-25.csv"
        month_df.to_csv(path, index=False)
        paths.append(path)
    return paths


# load_raw_csvs (3 tests)
def test_load_raw_csvs_happy_path(csv_snapshot_paths):
    df = load_raw_csvs(csv_snapshot_paths)
    assert len(df) == 36
    assert set(df["snapshot_month"].unique()) == {"03", "06", "09"}


def test_load_raw_csvs_empty_list_returns_empty_dataframe():
    df = load_raw_csvs([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_raw_csvs_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "listings-03-25.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_csvs([missing])


# normalize_listing_id (3 tests)
def test_normalize_listing_id_renames_id_column(tiny_df):
    out = normalize_listing_id(tiny_df, id_column="id")
    assert "listing_id" in out.columns
    assert "id" not in out.columns
    assert out["listing_id"].tolist() == ["1", "2", "3"]


def test_normalize_listing_id_keeps_existing_listing_id_as_string(tiny_df):
    df = tiny_df.drop(columns=["id"]).copy()
    out = normalize_listing_id(df, id_column="id")
    assert pd.api.types.is_string_dtype(out["listing_id"]) 
    assert out["listing_id"].tolist() == ["1", "2", "3"]


def test_normalize_listing_id_raises_when_no_id_columns(tiny_df):
    df = tiny_df.drop(columns=["id", "listing_id"]).copy()
    with pytest.raises(KeyError):
        normalize_listing_id(df, id_column="id")


# clean_price_column (4 tests)
def test_clean_price_column_standard_currency_format():
    df = pd.DataFrame({"price": ["$120.00"]})
    out = clean_price_column(df)
    assert out.loc[0, "price"] == pytest.approx(120.0)


def test_clean_price_column_handles_commas():
    df = pd.DataFrame({"price": ["$1,234.50"]})
    out = clean_price_column(df)
    assert out.loc[0, "price"] == pytest.approx(1234.5)


def test_clean_price_column_handles_raw_numeric_strings():
    df = pd.DataFrame({"price": ["150"]})
    out = clean_price_column(df)
    assert out.loc[0, "price"] == pytest.approx(150.0)


def test_clean_price_column_coerces_garbage_and_nan_to_nan():
    df = pd.DataFrame({"price": ["Contact host", np.nan]})
    out = clean_price_column(df)
    assert pd.isna(out.loc[0, "price"])
    assert pd.isna(out.loc[1, "price"])


# drop_missing_price_rows (3 tests)
def test_drop_missing_price_rows_keeps_only_positive_prices():
    df = pd.DataFrame({"price": [100.0, 0.0, -1.0, 40.0]})
    out = drop_missing_price_rows(df)
    assert out["price"].tolist() == [100.0, 40.0]


def test_drop_missing_price_rows_drops_nan():
    df = pd.DataFrame({"price": [100.0, np.nan, 200.0]})
    out = drop_missing_price_rows(df)
    assert out["price"].tolist() == [100.0, 200.0]


def test_drop_missing_price_rows_all_invalid_returns_empty():
    df = pd.DataFrame({"price": [np.nan, 0.0, -10.0]})
    out = drop_missing_price_rows(df)
    assert out.empty


# add_season_ordinal (3 tests)
def test_add_season_ordinal_maps_known_months():
    df = pd.DataFrame({"snapshot_month": ["03", "06", "09"]})
    out = add_season_ordinal(df, month_column="snapshot_month")
    assert out["season_ordinal"].tolist() == [1, 2, 3]


def test_add_season_ordinal_accepts_non_zero_padded_months():
    df = pd.DataFrame({"snapshot_month": ["3", "6", "9"]})
    out = add_season_ordinal(df, month_column="snapshot_month")
    assert out["season_ordinal"].tolist() == [1, 2, 3]


def test_add_season_ordinal_raises_on_unsupported_month():
    df = pd.DataFrame({"snapshot_month": ["12"]})
    with pytest.raises(ValueError):
        add_season_ordinal(df, month_column="snapshot_month")


# create_full_text_column (3 tests)
def test_create_full_text_column_happy_path():
    df = pd.DataFrame({"description": ["Nice place"], "amenities": ["WiFi, Desk"]})
    out = create_full_text_column(df, "description", "amenities")
    assert out.loc[0, "full_text"] == "Nice place [SEP] WiFi, Desk"


def test_create_full_text_column_handles_none_values():
    df = pd.DataFrame({"description": [None], "amenities": ["WiFi"]})
    out = create_full_text_column(df, "description", "amenities")
    assert out.loc[0, "full_text"] == "[SEP] WiFi"


def test_create_full_text_column_strips_extra_whitespace():
    df = pd.DataFrame({"description": ["  Nice place  "], "amenities": ["  WiFi  "]})
    out = create_full_text_column(df, "description", "amenities")
    assert out.loc[0, "full_text"] == "Nice place [SEP] WiFi"


# fit_language_detector / apply_language_detector (4 tests)
def test_fit_language_detector_returns_expected_keys(tiny_df):
    artifact = fit_language_detector(tiny_df["description"])
    assert "markers" in artifact
    assert "accent_chars" in artifact


def test_fit_language_detector_empty_series_still_returns_defaults():
    artifact = fit_language_detector(pd.Series([], dtype=object))
    assert isinstance(artifact["markers"], list)
    assert len(artifact["markers"]) > 0


def test_apply_language_detector_happy_path():
    artifact = {"markers": [" appartement "], "accent_chars": ""}
    df = pd.DataFrame({"description": ["Bel appartement central", "Cozy downtown studio"]})
    out = apply_language_detector(df, artifact, "description")
    assert out["is_french"].tolist() == [True, False]


def test_apply_language_detector_uses_accents_as_fallback():
    artifact = {"markers": [], "accent_chars": "é"}
    df = pd.DataFrame({"description": ["Tres beau quartier", "Café tres calme"]})
    out = apply_language_detector(df, artifact, "description")
    assert out["is_french"].tolist() == [False, True]


# mark_image_availability (3 tests)
def test_mark_image_availability_finds_exact_jpg(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "1.jpg").write_bytes(b"x")
    df = pd.DataFrame({"listing_id": ["1", "2"]})
    out = mark_image_availability(df, raw_dir, "listing_id")
    assert out["has_valid_image"].tolist() == [True, False]


def test_mark_image_availability_finds_any_extension(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "2.webp").write_bytes(b"x")
    df = pd.DataFrame({"listing_id": ["2"]})
    out = mark_image_availability(df, raw_dir, "listing_id")
    assert bool(out.loc[0, "has_valid_image"]) is True


def test_mark_image_availability_with_no_files_marks_false(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame({"listing_id": ["1"]})
    out = mark_image_availability(df, raw_dir, "listing_id")
    assert bool(out.loc[0, "has_valid_image"]) is False


# split_data_80_10_10 (4 tests)
def test_split_data_80_10_10_proportions_and_disjoint(split_ready_df):
    train, val, test = split_data_80_10_10(split_ready_df, seed=42)
    assert len(train) + len(val) + len(test) == len(split_ready_df)
    train_ids = set(train["listing_id"])
    val_ids = set(val["listing_id"])
    test_ids = set(test["listing_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_data_80_10_10_is_deterministic(split_ready_df):
    t1, v1, s1 = split_data_80_10_10(split_ready_df, seed=42)
    t2, v2, s2 = split_data_80_10_10(split_ready_df, seed=42)
    assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
    assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
    assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


def test_split_data_80_10_10_empty_dataframe_returns_three_empty_frames():
    empty_df = pd.DataFrame(columns=["listing_id", "price"])
    train, val, test = split_data_80_10_10(empty_df, seed=42)
    assert train.empty and val.empty and test.empty


def test_split_data_80_10_10_small_input_safe_remainder_logic():
    tiny = pd.DataFrame({"listing_id": ["1", "2"], "price": [100, 200]})
    train, val, test = split_data_80_10_10(tiny, seed=42)
    assert len(train) + len(val) + len(test) == 2


# fit_box_cox_transformer / apply_box_cox_transformer (4 tests)
def test_fit_box_cox_transformer_happy_path():
    transformer = fit_box_cox_transformer(pd.Series([100.0, 120.0, 200.0]))
    assert hasattr(transformer, "lambdas_")


def test_fit_box_cox_transformer_raises_on_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_box_cox_transformer(pd.Series([0.0, -1.0]))


def test_fit_box_cox_transformer_raises_on_empty_series():
    with pytest.raises(ValueError, match="at least one positive"):
        fit_box_cox_transformer(pd.Series([], dtype=float))


def test_apply_box_cox_transformer_returns_transformed_series():
    src = pd.Series([100.0, 120.0, 200.0], name="price")
    transformer = fit_box_cox_transformer(src)
    out = apply_box_cox_transformer(src, transformer)
    assert len(out) == len(src)
    assert out.name == "price"
    assert not np.allclose(out.to_numpy(), src.to_numpy())


# fit_numeric_imputer / apply_numeric_imputer (4 tests)
def test_fit_numeric_imputer_happy_path_medians():
    df = pd.DataFrame({"bathrooms": [1.0, 2.0, 3.0], "beds": [1.0, 2.0, 3.0]})
    artifact = fit_numeric_imputer(df, ["bathrooms", "beds"])
    assert artifact["bathrooms"] == pytest.approx(2.0)
    assert artifact["beds"] == pytest.approx(2.0)


def test_fit_numeric_imputer_all_nan_column_defaults_zero():
    df = pd.DataFrame({"bathrooms": [np.nan, np.nan]})
    artifact = fit_numeric_imputer(df, ["bathrooms"])
    assert artifact["bathrooms"] == pytest.approx(0.0)


def test_apply_numeric_imputer_fills_nan_but_preserves_existing_zero():
    artifact = {"bathrooms": 1.5, "beds": 2.0}
    df = pd.DataFrame({"bathrooms": [np.nan, 0.0], "beds": [np.nan, 0.0]})
    out = apply_numeric_imputer(df, artifact, ["bathrooms", "beds"])
    assert out.loc[0, "bathrooms"] == pytest.approx(1.5)
    assert out.loc[1, "bathrooms"] == pytest.approx(0.0)
    assert out.loc[1, "beds"] == pytest.approx(0.0)


def test_apply_numeric_imputer_coerces_numeric_strings():
    artifact = {"bathrooms": 1.0}
    df = pd.DataFrame({"bathrooms": ["2.5", None]})
    out = apply_numeric_imputer(df, artifact, ["bathrooms"])
    assert out["bathrooms"].tolist() == [2.5, 1.0]


# fit_categorical_encoder / apply_categorical_encoder (5 tests)
def test_fit_categorical_encoder_assigns_indices_starting_at_one():
    df = pd.DataFrame({"room_type": ["Entire home/apt", "Private room"]})
    artifact = fit_categorical_encoder(df, ["room_type"])
    assert set(artifact["room_type"].values()) == {1, 2}


def test_fit_categorical_encoder_ignores_unknown_like_values():
    df = pd.DataFrame({"room_type": ["Entire home/apt", "Unknown", None, "  "]})
    artifact = fit_categorical_encoder(df, ["room_type"])
    assert artifact["room_type"] == {"Entire home/apt": 1}


def test_apply_categorical_encoder_maps_known_categories_correctly():
    artifact = {"room_type": {"Entire home/apt": 1, "Private room": 2}}
    df = pd.DataFrame({"room_type": ["Private room", "Entire home/apt"]})
    out = apply_categorical_encoder(df, artifact, ["room_type"])
    assert out["room_type"].tolist() == [2, 1]


def test_apply_categorical_encoder_maps_unseen_to_zero():
    artifact = {"room_type": {"Entire home/apt": 1}}
    df = pd.DataFrame({"room_type": ["Spaceship"]})
    out = apply_categorical_encoder(df, artifact, ["room_type"])
    assert out.loc[0, "room_type"] == 0


def test_apply_categorical_encoder_maps_none_and_nan_to_zero():
    artifact = {"room_type": {"Entire home/apt": 1}}
    df = pd.DataFrame({"room_type": [None, np.nan]})
    out = apply_categorical_encoder(df, artifact, ["room_type"])
    assert out["room_type"].tolist() == [0, 0]


# fit_numeric_scaler / apply_numeric_scaler (4 tests)
def test_fit_numeric_scaler_happy_path_returns_scaler_instance():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    scaler = fit_numeric_scaler(df, ["x", "y"])
    assert isinstance(scaler, StandardScaler)


def test_fit_numeric_scaler_raises_on_empty_input():
    with pytest.raises(ValueError):
        fit_numeric_scaler(pd.DataFrame({"x": []}), ["x"])


def test_apply_numeric_scaler_transforms_values():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = fit_numeric_scaler(train, ["x"])
    out = apply_numeric_scaler(train, scaler, ["x"])
    assert abs(out["x"].mean()) < 1e-8


def test_apply_numeric_scaler_empty_dataframe_returns_empty_safely():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = fit_numeric_scaler(train, ["x"])
    empty = pd.DataFrame(columns=["x"])
    out = apply_numeric_scaler(empty, scaler, ["x"])
    assert out.empty


# fit_room_type_weights / apply_room_type_weights (4 tests)
def test_fit_room_type_weights_happy_path_inverse_frequency():
    df = pd.DataFrame({"room_type": ["A", "A", "B"]})
    weights = fit_room_type_weights(df, "room_type")
    assert weights["B"] > weights["A"]


def test_fit_room_type_weights_empty_after_normalization_returns_empty_dict():
    df = pd.DataFrame({"room_type": [None, "Unknown", np.nan]})
    weights = fit_room_type_weights(df, "room_type")
    assert weights == {}


def test_apply_room_type_weights_happy_path():
    df = pd.DataFrame({"room_type": ["A", "B"]})
    out = apply_room_type_weights(df, {"A": 0.5, "B": 2.0}, "room_type")
    assert out["sample_weight"].tolist() == [0.5, 2.0]


def test_apply_room_type_weights_uses_default_when_missing_in_map():
    df = pd.DataFrame({"room_type": ["X"]})
    out = apply_room_type_weights(df, {}, "room_type")
    assert out.loc[0, "sample_weight"] == pytest.approx(1.0)


# assemble_tabular_output (3 tests)
def test_assemble_tabular_output_happy_path_column_presence(split_ready_df):
    df = split_ready_df.copy()
    df["price_bc"] = df["price"]
    df["sample_weight"] = 1.0
    df["has_valid_image"] = True
    df["is_french"] = False
    df["full_text"] = "text"
    out = assemble_tabular_output(
        df,
        target_columns=["price", "price_bc"],
        feature_columns=["room_type", "accommodates", "season_ordinal"],
    )
    for col in ["listing_id", "price", "price_bc", "sample_weight", "room_type", "accommodates", "season_ordinal"]:
        assert col in out.columns


def test_assemble_tabular_output_skips_missing_columns_gracefully():
    df = pd.DataFrame({"listing_id": ["1"], "price": [100.0]})
    out = assemble_tabular_output(df, target_columns=["price_bc"], feature_columns=["room_type"])
    assert out.columns.tolist() == ["listing_id"]


def test_assemble_tabular_output_deduplicates_columns():
    df = pd.DataFrame({"listing_id": ["1"], "price": [100.0], "sample_weight": [1.0], "room_type": [1]})
    out = assemble_tabular_output(df, target_columns=["price", "price"], feature_columns=["room_type", "room_type"])
    assert out.columns.tolist().count("price") == 1
    assert out.columns.tolist().count("room_type") == 1


# save_parquet / save_artifact (3 tests)
def test_save_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"x": [1, 2]})
    path = tmp_path / "sub" / "file.parquet"
    save_parquet(df, path)
    loaded = pd.read_parquet(path)
    assert_frame_equal(df, loaded)


def test_save_artifact_round_trip(tmp_path):
    artifact = {"a": 1}
    path = tmp_path / "sub" / "artifact.joblib"
    save_artifact(artifact, path)
    loaded = joblib.load(path)
    assert loaded == artifact


def test_save_helpers_create_parent_directories(tmp_path):
    parquet_path = tmp_path / "a" / "b" / "c.parquet"
    artifact_path = tmp_path / "x" / "y" / "z.joblib"
    save_parquet(pd.DataFrame({"x": [1]}), parquet_path)
    save_artifact({"x": 1}, artifact_path)
    assert parquet_path.exists()
    assert artifact_path.exists()


# filter_cleaned_variant (3 tests)
def test_filter_cleaned_variant_happy_path_bounds_inclusive():
    df = pd.DataFrame({"price": [50.0, 1000.0]})
    out = filter_cleaned_variant(df)
    assert out["price"].tolist() == [50.0, 1000.0]


def test_filter_cleaned_variant_drops_outside_bounds():
    df = pd.DataFrame({"price": [49.9, 1000.1, 500.0]})
    out = filter_cleaned_variant(df)
    assert out["price"].tolist() == [500.0]


def test_filter_cleaned_variant_all_filtered_returns_empty():
    df = pd.DataFrame({"price": [1.0, 1001.0]})
    out = filter_cleaned_variant(df)
    assert out.empty


# export_normal_and_cleaned_variants (3 tests)
def test_export_normal_and_cleaned_variants_writes_required_files(tmp_path, split_ready_df):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    raw_image_dir = tmp_path / "images_raw"
    raw_image_dir.mkdir()
    (raw_image_dir / "1.jpg").write_bytes(b"x")

    base = split_ready_df.copy()
    base.loc[base["listing_id"] == "1", "price"] = 25
    base.loc[base["listing_id"] == "2", "price"] = 1500

    paths = []
    for month in ["03", "06", "09"]:
        month_df = base[base["snapshot_month"] == month].copy()
        p = csv_dir / f"listings-{month}-25.csv"
        month_df.to_csv(p, index=False)
        paths.append(p)

    output_dir = tmp_path / "output"
    export_normal_and_cleaned_variants(paths, raw_image_dir=raw_image_dir, output_dir=output_dir)

    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "train_cleaned.parquet").exists()
    assert (output_dir / "train_tabular.parquet").exists()
    assert (output_dir / "train_cleaned_tabular.parquet").exists()
    assert (output_dir / "language_detector.joblib").exists()


def test_export_normal_and_cleaned_variants_persists_independent_transformers(tmp_path, split_ready_df):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    raw_image_dir = tmp_path / "images_raw"
    raw_image_dir.mkdir()

    base = split_ready_df.copy()
    base.loc[base["listing_id"] == "1", "price"] = 25
    base.loc[base["listing_id"] == "2", "price"] = 1500

    paths = []
    for month in ["03", "06", "09"]:
        month_df = base[base["snapshot_month"] == month].copy()
        p = csv_dir / f"listings-{month}-25.csv"
        month_df.to_csv(p, index=False)
        paths.append(p)

    output_dir = tmp_path / "output"
    export_normal_and_cleaned_variants(paths, raw_image_dir=raw_image_dir, output_dir=output_dir)

    normal_boxcox = joblib.load(output_dir / "price_transformer.joblib")
    cleaned_boxcox = joblib.load(output_dir / "price_transformer_cleaned.joblib")
    assert hasattr(normal_boxcox, "lambdas_")
    assert hasattr(cleaned_boxcox, "lambdas_")
    assert not np.allclose(normal_boxcox.lambdas_, cleaned_boxcox.lambdas_)


def test_export_normal_and_cleaned_variants_cleaned_prices_respect_bounds(tmp_path, split_ready_df):
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    raw_image_dir = tmp_path / "images_raw"
    raw_image_dir.mkdir()

    base = split_ready_df.copy()
    base.loc[base["listing_id"] == "1", "price"] = 25
    base.loc[base["listing_id"] == "2", "price"] = 1500

    paths = []
    for month in ["03", "06", "09"]:
        month_df = base[base["snapshot_month"] == month].copy()
        p = csv_dir / f"listings-{month}-25.csv"
        month_df.to_csv(p, index=False)
        paths.append(p)

    output_dir = tmp_path / "output"
    export_normal_and_cleaned_variants(paths, raw_image_dir=raw_image_dir, output_dir=output_dir)

    cleaned_train = pd.read_parquet(output_dir / "train_cleaned.parquet")
    assert cleaned_train["price"].between(50, 1000).all()
