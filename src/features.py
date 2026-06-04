from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .age import add_age_object_features, fit_age_pclass_stats
from .fare import add_fare_object_features
from .group_fate import add_group_fate_features, fit_group_fate_stats
from .name_object import add_name_object_features
from .name_origin import NAME_ORIGIN_COLUMNS, add_name_origin_features
from .spatial import SPATIAL_OUTPUT_COLUMNS, add_spatial_features
from .ticket_group import add_ticket_group_features, fit_ticket_group_stats
from .travel_group import add_travel_group_features, fit_travel_group_stats


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
    "FarePerTicketMember",
    "FarePerFamilyTicketMember",
    "TicketGroupSize",
    "TicketChildCount",
    "TicketFemaleCount",
    "TicketAdultMaleCount",
    "TicketDistinctFamilyCount",
    "TicketMaxFamilySize",
    "TicketChildRatio",
    "TicketFemaleRatio",
    "TicketAdultMaleRatio",
    "TicketHasChild",
    "TicketHasFemale",
    "TicketHasAdultMale",
    "TicketHasFamily",
    "TicketIsMixedSex",
    "FamilyGroupSize",
    "FamilyTicketGroupSize",
    "TicketFamilyMismatch",
    "TicketFamilyDelta",
    "HighRawFareLargeFamilyFlag",
    "ZeroFareFlag",
    "PclassAgeInteraction",
    "PclassFareInteraction",
    "AgeKnown",
    "AgeMissing",
    "AgeZWithinPclass",
    "IsYoungWithinPclass",
    "IsOldWithinPclass",
    "IsOldMaleWithinPclass",
    "IsYoungMasterWithinPclass",
    "IsChild",
    "IsAdultMale",
    "IsMother",
    "IsFirstClassFemale",
    "IsThirdClassMale",
]

AGE_RELATIVE_NUMERIC_FEATURES = {
    "AgeZWithinPclass",
    "IsYoungWithinPclass",
    "IsOldWithinPclass",
    "IsOldMaleWithinPclass",
    "IsYoungMasterWithinPclass",
}

AGE_RELATIVE_CATEGORICAL_FEATURES = {
    "AgeQuartileWithinPclass",
    "AgeRelativeBucketWithinPclass",
    "AgeRelativeBucketWithinPclassSex",
}

TICKET_GROUP_COMPOSITION_NUMERIC_FEATURES = {
    "TicketChildCount",
    "TicketFemaleCount",
    "TicketAdultMaleCount",
    "TicketDistinctFamilyCount",
    "TicketMaxFamilySize",
    "TicketChildRatio",
    "TicketFemaleRatio",
    "TicketAdultMaleRatio",
    "TicketHasChild",
    "TicketHasFemale",
    "TicketHasAdultMale",
    "TicketHasFamily",
    "TicketIsMixedSex",
}

TICKET_GROUP_COMPOSITION_CATEGORICAL_FEATURES = {
    "TicketCompositionType",
    "TicketFamilyPattern",
    "TicketChildFemalePattern",
    "TicketChildCountBin",
    "TicketFemaleCountBin",
    "TicketChildRatioBin",
    "TicketFemaleRatioBin",
}

TICKET_ROUTE_CATEGORICAL_FEATURES = [
    "TicketPrefix",
    "TicketNumberBand",
    "TicketPrefixEmbarked",
    "TicketNumberBandEmbarked",
    "TicketPrefixPclassEmbarked",
]

TRAVEL_GROUP_NUMERIC_FEATURES = [
    "SelfChildProxy",
    "SelfFemaleProxy",
    "SelfMaleProxy",
    "SelfAdultMaleProxy",
    "SelfMasterProxy",
    "FamilyChildCount",
    "FamilyFemaleCount",
    "FamilyMaleCount",
    "FamilyAdultMaleCount",
    "FamilyMasterCount",
    "FamilyChildRatio",
    "FamilyFemaleRatio",
    "FamilyAdultMaleRatio",
    "FamilyHasChild",
    "FamilyHasFemale",
    "FamilyHasAdultMale",
    "FamilyIsMixedSex",
    "FamilyAccompanyingChildCount",
    "FamilyAccompanyingFemaleCount",
    "FamilyAccompanyingAdultMaleCount",
    "FamilyTicketChildCount",
    "FamilyTicketFemaleCount",
    "FamilyTicketMaleCount",
    "FamilyTicketAdultMaleCount",
    "FamilyTicketMasterCount",
    "FamilyTicketChildRatio",
    "FamilyTicketFemaleRatio",
    "FamilyTicketAdultMaleRatio",
    "FamilyTicketHasChild",
    "FamilyTicketHasFemale",
    "FamilyTicketHasAdultMale",
    "FamilyTicketIsMixedSex",
    "FamilyTicketAccompanyingChildCount",
    "FamilyTicketAccompanyingFemaleCount",
    "FamilyTicketAccompanyingAdultMaleCount",
    "CompanionGroupSize",
    "CompanionChildCount",
    "CompanionFemaleCount",
    "CompanionAdultMaleCount",
    "CompanionChildRatio",
    "CompanionFemaleRatio",
    "CompanionAdultMaleRatio",
    "CompanionAccompanyingChildCount",
    "CompanionAccompanyingFemaleCount",
    "CompanionAccompanyingAdultMaleCount",
]

TRAVEL_GROUP_CATEGORICAL_FEATURES = [
    "SelfGroupRole",
    "FamilyChildCountBin",
    "FamilyFemaleCountBin",
    "FamilyAdultMaleCountBin",
    "FamilyAccompanyingChildCountBin",
    "FamilyAccompanyingFemaleCountBin",
    "FamilyAccompanyingAdultMaleCountBin",
    "FamilyChildRatioBin",
    "FamilyFemaleRatioBin",
    "FamilyAdultMaleRatioBin",
    "FamilyChildFemalePattern",
    "FamilyTicketChildCountBin",
    "FamilyTicketFemaleCountBin",
    "FamilyTicketAdultMaleCountBin",
    "FamilyTicketAccompanyingChildCountBin",
    "FamilyTicketAccompanyingFemaleCountBin",
    "FamilyTicketAccompanyingAdultMaleCountBin",
    "FamilyTicketChildRatioBin",
    "FamilyTicketFemaleRatioBin",
    "FamilyTicketAdultMaleRatioBin",
    "FamilyTicketChildFemalePattern",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "ChildFemaleCompanionPattern",
    "CompanionChildCountBin",
    "CompanionFemaleCountBin",
    "CompanionAdultMaleCountBin",
    "CompanionAccompanyingChildCountBin",
    "CompanionAccompanyingFemaleCountBin",
    "CompanionAccompanyingAdultMaleCountBin",
    "CompanionChildRatioBin",
    "CompanionFemaleRatioBin",
    "CompanionAdultMaleRatioBin",
]

COMPACT_TRAVEL_GROUP_NUMERIC_FEATURES = [
    "CompanionGroupSize",
    "CompanionChildCount",
    "CompanionFemaleCount",
    "CompanionAdultMaleCount",
    "CompanionAccompanyingChildCount",
    "CompanionAccompanyingFemaleCount",
    "CompanionAccompanyingAdultMaleCount",
]

COMPACT_TRAVEL_GROUP_CATEGORICAL_FEATURES = [
    "SelfGroupRole",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "ChildFemaleCompanionPattern",
    "FamilyTicketChildFemalePattern",
]

GROUP_FATE_NUMERIC_FEATURES = [
    "TicketFateKnown",
    "TicketFateKnownCount",
    "TicketFateSurvivorCount",
    "TicketFateSurvivalRate",
    "TicketFateSignalValue",
    "FamilyFateKnown",
    "FamilyFateKnownCount",
    "FamilyFateSurvivorCount",
    "FamilyFateSurvivalRate",
    "FamilyFateSignalValue",
    "FamilyTicketFateKnown",
    "FamilyTicketFateKnownCount",
    "FamilyTicketFateSurvivorCount",
    "FamilyTicketFateSurvivalRate",
    "FamilyTicketFateSignalValue",
    "PrimaryFateKnown",
    "PrimaryFateKnownCount",
    "PrimaryFateSurvivorCount",
    "PrimaryFateSurvivalRate",
    "PrimaryFateSignalValue",
]

GROUP_FATE_CATEGORICAL_FEATURES = [
    "TicketFateSignal",
    "TicketFateSupportBin",
    "FamilyFateSignal",
    "FamilyFateSupportBin",
    "FamilyTicketFateSignal",
    "FamilyTicketFateSupportBin",
    "PrimaryFateScope",
    "PrimaryFateSignal",
    "PrimaryFateSupportBin",
]

COMPACT_GROUP_FATE_NUMERIC_FEATURES = [
    "TicketFateKnown",
    "TicketFateKnownCount",
    "TicketFateSurvivalRate",
    "TicketFateSignalValue",
    "FamilyTicketFateKnown",
    "FamilyTicketFateKnownCount",
    "FamilyTicketFateSurvivalRate",
    "FamilyTicketFateSignalValue",
    "PrimaryFateKnown",
    "PrimaryFateKnownCount",
    "PrimaryFateSurvivalRate",
    "PrimaryFateSignalValue",
]

COMPACT_GROUP_FATE_CATEGORICAL_FEATURES = [
    "TicketFateSignal",
    "FamilyTicketFateSignal",
    "PrimaryFateScope",
    "PrimaryFateSignal",
    "PrimaryFateSupportBin",
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
    + TRAVEL_GROUP_NUMERIC_FEATURES
    + GROUP_FATE_NUMERIC_FEATURES
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
] + TICKET_ROUTE_CATEGORICAL_FEATURES + TRAVEL_GROUP_CATEGORICAL_FEATURES + GROUP_FATE_CATEGORICAL_FEATURES

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CORE_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES

CORE_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "AgeQuartileWithinPclass",
    "AgeRelativeBucketWithinPclass",
    "AgeRelativeBucketWithinPclassSex",
    "AgeMissingPatternByPclassTitle",
    "FareBin",
    "FarePerTicketBin",
    "FamilySizeBin",
    *TICKET_ROUTE_CATEGORICAL_FEATURES,
    "TicketCompositionType",
    "TicketFamilyPattern",
    "TicketChildFemalePattern",
    "TicketChildCountBin",
    "TicketFemaleCountBin",
    "TicketChildRatioBin",
    "TicketFemaleRatioBin",
    "TravelPartyType",
    "FareObjectInterpretation",
    "SexPclass",
    "TitlePclass",
    "AgeBinSex",
    "FareBinPclass",
    "FarePerTicketBinPclass",
]

CLEAN_NUMERIC_FEATURES = (
    [
        feature
        for feature in BASE_NUMERIC_FEATURES
        if feature not in AGE_RELATIVE_NUMERIC_FEATURES
        and feature not in TICKET_GROUP_COMPOSITION_NUMERIC_FEATURES
    ]
    + CLEAN_SPATIAL_NUMERIC_FEATURES
    + BINARY_FEATURES
)

CLEAN_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "AgeMissingPatternByPclassTitle",
    "FareBin",
    "FarePerTicketBin",
    "FamilySizeBin",
    *TICKET_ROUTE_CATEGORICAL_FEATURES,
    "TravelPartyType",
    "FareObjectInterpretation",
    "SexPclass",
    "TitlePclass",
    "AgeBinSex",
    "FareBinPclass",
    "FarePerTicketBinPclass",
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
    "clean_group_objects": {
        "numeric": CLEAN_NUMERIC_FEATURES + COMPACT_TRAVEL_GROUP_NUMERIC_FEATURES,
        "categorical": CLEAN_CATEGORICAL_FEATURES + COMPACT_TRAVEL_GROUP_CATEGORICAL_FEATURES,
        "description": "Clean set plus compact entity-style family/ticket companion object features.",
    },
    "clean_group_fate": {
        "numeric": CLEAN_NUMERIC_FEATURES + COMPACT_GROUP_FATE_NUMERIC_FEATURES,
        "categorical": CLEAN_CATEGORICAL_FEATURES + COMPACT_GROUP_FATE_CATEGORICAL_FEATURES,
        "description": "Clean set plus target-safe leave-one-out group fate features.",
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


def ticket_number(ticket: object) -> float:
    ticket_value = normalize_ticket(ticket)
    numbers = re.findall(r"\d+", ticket_value)
    if not numbers:
        return np.nan
    return float(int(numbers[-1]))


def ticket_number_band(number: object) -> str:
    if pd.isna(number):
        return "NoNumber"
    value = int(number)
    if value < 1000:
        return "00000_00999"
    if value < 10000:
        return "01000_09999"
    if value < 20000:
        return "10000_19999"
    if value < 50000:
        return "20000_49999"
    if value < 100000:
        return "50000_99999"
    if value < 200000:
        return "100000_199999"
    if value < 300000:
        return "200000_299999"
    if value < 350000:
        return "300000_349999"
    if value < 400000:
        return "350000_399999"
    return "400000_plus"


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
        out["TicketNumber"] = out["Ticket"].map(ticket_number)
    else:
        out["TicketNormalized"] = "UNKNOWN"
        out["TicketPrefix"] = "UNKNOWN"
        out["TicketNumber"] = np.nan

    out["TicketNumberBand"] = out["TicketNumber"].map(ticket_number_band)
    embarked_value = out["Embarked"].fillna("Unknown").astype(str)
    pclass_value = out["Pclass"].fillna("Unknown").astype(str)
    out["TicketPrefixEmbarked"] = (
        out["TicketPrefix"].astype(str) + "_" + embarked_value
    )
    out["TicketNumberBandEmbarked"] = (
        out["TicketNumberBand"].astype(str) + "_" + embarked_value
    )
    out["TicketPrefixPclassEmbarked"] = (
        out["TicketPrefix"].astype(str) + "_P" + pclass_value + "_" + embarked_value
    )

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
    family_ticket_group_size_map: dict[str, int] | None = None,
    age_pclass_stats: dict | None = None,
    ticket_group_stats: dict | None = None,
    travel_group_stats: dict | None = None,
    group_fate_stats: dict | None = None,
    group_fate_y: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    out = add_core_titanic_features(df)
    out = add_name_object_features(out)
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
    if family_ticket_group_size_map is None:
        family_ticket_key = out["FamilyKey"].astype(str) + "_" + out["TicketNormalized"].astype(str)
        family_ticket_group_size_map = family_ticket_key.value_counts().to_dict()

    out["TicketGroupSize"] = out["TicketNormalized"].map(ticket_group_size_map).fillna(1)
    out["FamilyGroupSize"] = out["FamilyKey"].map(family_group_size_map).fillna(1)
    out = add_ticket_group_features(out, ticket_group_stats=ticket_group_stats)
    out = add_travel_group_features(out, travel_group_stats=travel_group_stats)
    out = add_group_fate_features(
        out,
        group_fate_stats=group_fate_stats,
        y=group_fate_y,
    )
    out = add_fare_object_features(
        out,
        ticket_group_size_map=ticket_group_size_map,
        family_ticket_group_size_map=family_ticket_group_size_map,
    )
    out = add_age_object_features(out, age_pclass_stats=age_pclass_stats)

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
        self.ticket_group_stats_ = fit_ticket_group_stats(frame)
        self.travel_group_stats_ = fit_travel_group_stats(frame)
        self.group_fate_stats_ = fit_group_fate_stats(frame, y)
        self.family_group_size_map_ = frame["FamilyKey"].value_counts().to_dict()
        family_ticket_key = frame["FamilyKey"].astype(str) + "_" + frame[
            "TicketNormalized"
        ].astype(str)
        self.family_ticket_group_size_map_ = family_ticket_key.value_counts().to_dict()
        self.age_pclass_stats_ = fit_age_pclass_stats(frame)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        self.fit(X, y)
        return build_feature_frame(
            X,
            derived_dir=self.derived_dir,
            external_dir=self.external_dir,
            feature_set=self.feature_set,
            ticket_group_size_map=getattr(self, "ticket_group_size_map_", None),
            family_group_size_map=getattr(self, "family_group_size_map_", None),
            family_ticket_group_size_map=getattr(
                self,
                "family_ticket_group_size_map_",
                None,
            ),
            age_pclass_stats=getattr(self, "age_pclass_stats_", None),
            ticket_group_stats=getattr(self, "ticket_group_stats_", None),
            travel_group_stats=getattr(self, "travel_group_stats_", None),
            group_fate_stats=getattr(self, "group_fate_stats_", None),
            group_fate_y=y,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_feature_frame(
            X,
            derived_dir=self.derived_dir,
            external_dir=self.external_dir,
            feature_set=self.feature_set,
            ticket_group_size_map=getattr(self, "ticket_group_size_map_", None),
            family_group_size_map=getattr(self, "family_group_size_map_", None),
            family_ticket_group_size_map=getattr(
                self,
                "family_ticket_group_size_map_",
                None,
            ),
            age_pclass_stats=getattr(self, "age_pclass_stats_", None),
            ticket_group_stats=getattr(self, "ticket_group_stats_", None),
            travel_group_stats=getattr(self, "travel_group_stats_", None),
            group_fate_stats=getattr(self, "group_fate_stats_", None),
        )
