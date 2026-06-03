from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .name_origin import NAME_ORIGIN_COLUMNS, add_name_origin_features
from .spatial import SPATIAL_OUTPUT_COLUMNS, add_spatial_features


RARE_TITLES = {
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
}

TITLE_REPLACEMENTS = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

BASE_NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
]

SPATIAL_NUMERIC_FEATURES = [
    "DeckOrdinal",
    "ApproxVerticalDistanceToBoatDeck",
    "LifeboatAccessScore",
    "StaircaseAccessScore",
]

BINARY_FEATURES = [
    "CabinKnown",
    "CabinMissing",
    "DeckKnown",
    "DeckMissing",
]

ORIGIN_NUMERIC_FEATURES = [
    "OriginInferenceConfidence",
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + SPATIAL_NUMERIC_FEATURES
    + BINARY_FEATURES
    + ORIGIN_NUMERIC_FEATURES
)

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "SpatialAccessTier",
    "ClassDeckConsistency",
    "CabinZone",
    "ClassArea",
    "InferredOriginRegion",
    "InferredPrimaryLanguage",
    "EnglishComprehensionProxy",
    "LanguageBarrierRisk",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_core_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["SibSp", "Parch"]:
        if col not in out.columns:
            out[col] = 0

    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    if "Name" in out.columns:
        title = out["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title = title.replace(TITLE_REPLACEMENTS)
        title = title.where(~title.isin(RARE_TITLES), "Rare")
        out["Title"] = title.fillna("Rare")
    else:
        out["Title"] = "Rare"

    return out


def _merge_new_columns(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if "PassengerId" not in left.columns or "PassengerId" not in right.columns:
        return left
    new_cols = [col for col in right.columns if col == "PassengerId" or col not in left.columns]
    if len(new_cols) <= 1:
        return left
    return left.merge(right[new_cols], on="PassengerId", how="left")


def _overlay_derived_file(
    df: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    if not path.exists() or "PassengerId" not in df.columns:
        return df

    derived = pd.read_csv(path)
    if derived.empty or "PassengerId" not in derived.columns:
        return df

    overlay_cols = [col for col in derived.columns if col != "PassengerId"]
    merged = df.merge(derived, on="PassengerId", how="left", suffixes=("", "_Derived"))
    for col in overlay_cols:
        derived_col = f"{col}_Derived"
        if derived_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[derived_col].combine_first(merged[col])
            else:
                merged[col] = merged[derived_col]
            merged = merged.drop(columns=[derived_col])
    return merged


def ensure_model_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    derived_dir: Path | str = Path("data/derived"),
    external_dir: Path | str = Path("external"),
) -> pd.DataFrame:
    out = add_core_titanic_features(df)
    external_path = Path(external_dir)

    spatial = add_spatial_features(out, external_dir=external_path)
    name_origin = add_name_origin_features(out)
    out = _merge_new_columns(out, spatial)
    out = _merge_new_columns(out, name_origin)

    derived_path = Path(derived_dir)
    out = _overlay_derived_file(out, derived_path / "spatial_features.csv")
    out = _overlay_derived_file(out, derived_path / "name_origin_features.csv")

    return ensure_model_feature_columns(out)


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the first project version."""
    return build_feature_frame(df)
