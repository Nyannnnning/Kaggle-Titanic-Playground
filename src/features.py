from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
    "FarePerFamilyMember",
    "TicketGroupSize",
    "FamilyGroupSize",
    "PclassAgeInteraction",
    "PclassFareInteraction",
    "IsChild",
    "IsAdultMale",
    "IsMother",
    "IsFirstClassFemale",
    "IsThirdClassMale",
]

SPATIAL_NUMERIC_FEATURES = [
    "DeckOrdinal",
    "ApproxVerticalDistanceToBoatDeck",
    "LifeboatAccessScore",
    "StaircaseAccessScore",
]

CLEAN_SPATIAL_NUMERIC_FEATURES = [
    "DeckOrdinal",
    "ApproxVerticalDistanceToBoatDeck",
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

CORE_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES

CORE_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "FareBin",
    "FamilySizeBin",
    "TicketPrefix",
    "SexPclass",
    "TitlePclass",
    "AgeBinSex",
    "FareBinPclass",
]

CLEAN_NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + CLEAN_SPATIAL_NUMERIC_FEATURES
    + BINARY_FEATURES
)

CLEAN_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "FareBin",
    "FamilySizeBin",
    "TicketPrefix",
    "SexPclass",
    "TitlePclass",
    "AgeBinSex",
    "FareBinPclass",
    "Deck",
    "SpatialAccessTier",
    "ClassDeckConsistency",
]

WEAK_SIGNAL_FEATURES = [
    "LifeboatAccessScore",
    "StaircaseAccessScore",
    "CabinZone",
    "ClassArea",
    "OriginInferenceConfidence",
    "InferredOriginRegion",
    "InferredPrimaryLanguage",
    "EnglishComprehensionProxy",
    "LanguageBarrierRisk",
]

FEATURE_SETS = {
    "core": {
        "numeric": CORE_NUMERIC_FEATURES,
        "categorical": CORE_CATEGORICAL_FEATURES,
        "description": "Only stable raw Titanic features plus engineered family/title features.",
    },
    "clean": {
        "numeric": CLEAN_NUMERIC_FEATURES,
        "categorical": CLEAN_CATEGORICAL_FEATURES,
        "description": "Default set: removes uncertain language-origin proxies and hand-scored access proxies.",
    },
    "full": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "description": "Exploration set: includes all provisional proxy features.",
    },
}


def feature_columns_for_set(feature_set: str = "full") -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    config = FEATURE_SETS[feature_set]
    return list(config["numeric"]), list(config["categorical"])


def passenger_surname(name: object) -> str:
    if pd.isna(name):
        return "Unknown"
    surname = str(name).split(",", maxsplit=1)[0].strip()
    return surname if surname else "Unknown"


def normalize_ticket(ticket: object) -> str:
    if pd.isna(ticket):
        return "UNKNOWN"
    ticket_value = str(ticket).upper().strip()
    ticket_value = re.sub(r"[./]", " ", ticket_value)
    ticket_value = re.sub(r"\s+", " ", ticket_value)
    return ticket_value if ticket_value else "UNKNOWN"


def ticket_prefix(ticket: object) -> str:
    ticket_value = normalize_ticket(ticket)
    pieces = ticket_value.split()
    if not pieces:
        return "UNKNOWN"
    prefix_parts = [piece for piece in pieces[:-1] if not piece.isdigit()]
    if prefix_parts:
        return "_".join(prefix_parts)
    if pieces[0].isdigit():
        return "NUMERIC"
    return pieces[0]


def age_bin(age: object) -> str:
    if pd.isna(age):
        return "AgeMissing"
    age_value = float(age)
    if age_value <= 5:
        return "Infant"
    if age_value <= 12:
        return "Child"
    if age_value <= 18:
        return "Teen"
    if age_value <= 30:
        return "YoungAdult"
    if age_value <= 45:
        return "Adult"
    if age_value <= 60:
        return "MiddleAge"
    return "Senior"


def fare_bin(fare: object) -> str:
    if pd.isna(fare):
        return "FareMissing"
    fare_value = float(fare)
    if fare_value <= 0:
        return "FreeOrUnknown"
    if fare_value <= 8:
        return "VeryLowFare"
    if fare_value <= 15:
        return "LowFare"
    if fare_value <= 31:
        return "MidFare"
    if fare_value <= 100:
        return "HighFare"
    return "LuxuryFare"


def family_size_bin(family_size: object) -> str:
    if pd.isna(family_size):
        return "FamilyUnknown"
    size = int(family_size)
    if size <= 1:
        return "Alone"
    if size <= 4:
        return "SmallFamily"
    if size <= 6:
        return "LargeFamily"
    return "VeryLargeFamily"


def add_core_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["SibSp", "Parch"]:
        if col not in out.columns:
            out[col] = 0

    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerFamilyMember"] = out["Fare"] / out["FamilySize"].replace(0, np.nan)

    if "Name" in out.columns:
        title = out["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title = title.replace(TITLE_REPLACEMENTS)
        title = title.where(~title.isin(RARE_TITLES), "Rare")
        out["Title"] = title.fillna("Rare")
        out["Surname"] = out["Name"].map(passenger_surname)
    else:
        out["Title"] = "Rare"
        out["Surname"] = "Unknown"

    if "Ticket" in out.columns:
        out["TicketNormalized"] = out["Ticket"].map(normalize_ticket)
        out["TicketPrefix"] = out["Ticket"].map(ticket_prefix)
    else:
        out["TicketNormalized"] = "UNKNOWN"
        out["TicketPrefix"] = "UNKNOWN"

    out["FamilyKey"] = out["Surname"].astype(str) + "_" + out["Pclass"].astype(str)
    out["AgeBin"] = out["Age"].map(age_bin) if "Age" in out.columns else "AgeMissing"
    out["FareBin"] = out["Fare"].map(fare_bin) if "Fare" in out.columns else "FareMissing"
    out["FamilySizeBin"] = out["FamilySize"].map(family_size_bin)

    out["SexPclass"] = out["Sex"].fillna("Unknown").astype(str) + "_P" + out["Pclass"].astype(str)
    out["TitlePclass"] = out["Title"].astype(str) + "_P" + out["Pclass"].astype(str)
    out["AgeBinSex"] = out["AgeBin"].astype(str) + "_" + out["Sex"].fillna("Unknown").astype(str)
    out["FareBinPclass"] = out["FareBin"].astype(str) + "_P" + out["Pclass"].astype(str)

    out["PclassAgeInteraction"] = out["Pclass"] * out["Age"]
    out["PclassFareInteraction"] = out["Pclass"] * out["Fare"]
    out["IsChild"] = out["Age"].le(12).fillna(False).astype(int)
    out["IsAdultMale"] = (
        out["Sex"].eq("male") & out["Age"].ge(18).fillna(False)
    ).astype(int)
    out["IsMother"] = (
        out["Sex"].eq("female")
        & out["Age"].ge(18).fillna(False)
        & out["Parch"].fillna(0).gt(0)
        & ~out["Title"].eq("Miss")
    ).astype(int)
    out["IsFirstClassFemale"] = (out["Sex"].eq("female") & out["Pclass"].eq(1)).astype(int)
    out["IsThirdClassMale"] = (out["Sex"].eq("male") & out["Pclass"].eq(3)).astype(int)

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


def ensure_model_feature_columns(
    df: pd.DataFrame,
    feature_set: str = "full",
) -> pd.DataFrame:
    out = df.copy()
    numeric_features, categorical_features = feature_columns_for_set(feature_set)
    for col in numeric_features:
        if col not in out.columns:
            out[col] = np.nan
    for col in categorical_features:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    derived_dir: Path | str = Path("data/derived"),
    external_dir: Path | str = Path("external"),
    feature_set: str = "full",
    ticket_group_size_map: dict[str, int] | None = None,
    family_group_size_map: dict[str, int] | None = None,
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

    if ticket_group_size_map is None:
        ticket_group_size_map = out["TicketNormalized"].value_counts().to_dict()
    if family_group_size_map is None:
        family_group_size_map = out["FamilyKey"].value_counts().to_dict()

    out["TicketGroupSize"] = out["TicketNormalized"].map(ticket_group_size_map).fillna(1)
    out["FamilyGroupSize"] = out["FamilyKey"].map(family_group_size_map).fillna(1)

    return ensure_model_feature_columns(out, feature_set=feature_set)


def add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the first project version."""
    return build_feature_frame(df)


class PassengerFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_set: str = "clean",
        derived_dir: Path | str = Path("data/derived"),
        external_dir: Path | str = Path("external"),
    ):
        self.feature_set = feature_set
        self.derived_dir = derived_dir
        self.external_dir = external_dir

    def fit(self, X: pd.DataFrame, y=None):
        frame = add_core_titanic_features(X)
        self.ticket_group_size_map_ = frame["TicketNormalized"].value_counts().to_dict()
        self.family_group_size_map_ = frame["FamilyKey"].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_feature_frame(
            X,
            derived_dir=self.derived_dir,
            external_dir=self.external_dir,
            feature_set=self.feature_set,
            ticket_group_size_map=getattr(self, "ticket_group_size_map_", None),
            family_group_size_map=getattr(self, "family_group_size_map_", None),
        )
