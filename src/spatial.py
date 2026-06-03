from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


DECK_ORDERING = {
    "Boat": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "Unknown": np.nan,
}

DECK_LAYOUT_COLUMNS = [
    "Deck",
    "DeckOrdinal",
    "ApproxVerticalDistanceToBoatDeck",
    "SpatialAccessTier",
    "CabinZone",
    "ClassArea",
    "LifeboatAccessScore",
    "StaircaseAccessScore",
    "Notes",
]

SPATIAL_OUTPUT_COLUMNS = [
    "PassengerId",
    "Cabin",
    "CabinKnown",
    "CabinMissing",
    "Deck",
    "DeckKnown",
    "DeckMissing",
    "CabinNumber",
    "DeckOrdinal",
    "ApproxVerticalDistanceToBoatDeck",
    "SpatialAccessTier",
    "ClassDeckConsistency",
    "CabinZone",
    "ClassArea",
    "LifeboatAccessScore",
    "StaircaseAccessScore",
]


def built_in_deck_layout() -> pd.DataFrame:
    rows = [
        ("Boat", 0, 0, "near_boat_deck", "top_deck", "lifeboat_area", 1.00, 0.90),
        ("A", 1, 1, "near_boat_deck", "upper_forward_mid", "mostly_first_class", 0.90, 0.85),
        ("B", 2, 2, "upper_deck", "upper_mid", "mostly_first_class", 0.80, 0.80),
        ("C", 3, 3, "upper_deck", "upper_mid_aft", "mixed_first_second", 0.70, 0.75),
        ("D", 4, 4, "middle_deck", "middle_mid", "mixed_first_second", 0.60, 0.65),
        ("E", 5, 5, "middle_deck", "middle_lower", "mixed_all_classes", 0.45, 0.55),
        ("F", 6, 6, "lower_deck", "lower_mid", "mixed_second_third", 0.30, 0.40),
        ("G", 7, 7, "lower_deck", "lower_low", "mostly_third_class", 0.20, 0.30),
        ("Unknown", np.nan, np.nan, "unknown", "unknown", "unknown", np.nan, np.nan),
    ]
    notes = (
        "provisional; manually review against historical deck plans; "
        "not exact walking distance; not exact lifeboat access"
    )
    return pd.DataFrame(
        rows,
        columns=[
            "Deck",
            "DeckOrdinal",
            "ApproxVerticalDistanceToBoatDeck",
            "SpatialAccessTier",
            "CabinZone",
            "ClassArea",
            "LifeboatAccessScore",
            "StaircaseAccessScore",
        ],
    ).assign(Notes=notes)


def normalize_cabin_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_deck(cabin: object) -> str:
    cabin_value = normalize_cabin_value(cabin)
    if not cabin_value:
        return "Unknown"

    match = re.search(r"[A-Za-z]", cabin_value)
    if not match:
        return "Unknown"

    deck = match.group(0).upper()
    if deck == "T":
        return "Boat"
    if deck in {"A", "B", "C", "D", "E", "F", "G"}:
        return deck
    return "Unknown"


def extract_cabin_number(cabin: object) -> float:
    cabin_value = normalize_cabin_value(cabin)
    match = re.search(r"\d+", cabin_value)
    if not match:
        return np.nan
    return float(match.group(0))


def spatial_access_tier(deck: str) -> str:
    if deck in {"Boat", "A"}:
        return "near_boat_deck"
    if deck in {"B", "C"}:
        return "upper_deck"
    if deck in {"D", "E"}:
        return "middle_deck"
    if deck in {"F", "G"}:
        return "lower_deck"
    return "unknown"


def class_deck_consistency(pclass: object, deck: str) -> str:
    if deck == "Unknown" or pd.isna(pclass):
        return "unknown"

    try:
        pclass_int = int(pclass)
    except (TypeError, ValueError):
        return "unknown"

    if pclass_int == 1 and deck in {"Boat", "A", "B", "C", "D", "E"}:
        return "first_class_upper"
    if pclass_int == 2 and deck in {"C", "D", "E", "F"}:
        return "second_class_middle"
    if pclass_int == 3 and deck in {"E", "F", "G"}:
        return "third_class_lower"
    return "class_deck_mismatch"


def load_deck_layout(external_dir: Path = Path("external")) -> pd.DataFrame:
    layout_path = external_dir / "deck_layout_features.csv"
    if layout_path.exists():
        layout = pd.read_csv(layout_path)
        if not layout.empty:
            return layout
    return built_in_deck_layout()


def load_cabin_map(external_dir: Path = Path("external")) -> pd.DataFrame:
    cabin_map_path = external_dir / "cabin_deck_map.csv"
    if not cabin_map_path.exists():
        return pd.DataFrame()

    cabin_map = pd.read_csv(cabin_map_path)
    if cabin_map.empty or "Cabin" not in cabin_map.columns:
        return pd.DataFrame()

    cabin_map = cabin_map.copy()
    cabin_map["CabinMapKey"] = cabin_map["Cabin"].map(normalize_cabin_value)
    return cabin_map


def add_spatial_features(
    df: pd.DataFrame,
    external_dir: Path = Path("external"),
) -> pd.DataFrame:
    out = df.copy()
    if "Cabin" not in out.columns:
        out["Cabin"] = np.nan
    if "Pclass" not in out.columns:
        out["Pclass"] = np.nan

    cabin_values = out["Cabin"].map(normalize_cabin_value)
    out["CabinKnown"] = cabin_values.ne("").astype(int)
    out["CabinMissing"] = (1 - out["CabinKnown"]).astype(int)
    out["Deck"] = out["Cabin"].map(extract_deck)
    out["CabinNumber"] = out["Cabin"].map(extract_cabin_number)

    cabin_map = load_cabin_map(external_dir)
    if not cabin_map.empty:
        out["CabinMapKey"] = cabin_values
        map_cols = [
            col
            for col in [
                "CabinMapKey",
                "Deck",
                "CabinNumber",
                "CabinZone",
                "ApproxDistanceToLifeboats",
            ]
            if col in cabin_map.columns
        ]
        out = out.merge(
            cabin_map[map_cols],
            on="CabinMapKey",
            how="left",
            suffixes=("", "_CabinMap"),
        )
        for col in ["Deck", "CabinNumber", "CabinZone", "ApproxDistanceToLifeboats"]:
            mapped_col = f"{col}_CabinMap"
            if mapped_col in out.columns:
                out[col] = out[mapped_col].combine_first(out[col])
                out = out.drop(columns=[mapped_col])
        out = out.drop(columns=["CabinMapKey"])

    out["Deck"] = out["Deck"].fillna("Unknown")
    out["DeckKnown"] = out["Deck"].ne("Unknown").astype(int)
    out["DeckMissing"] = (1 - out["DeckKnown"]).astype(int)

    layout = load_deck_layout(external_dir)
    layout_cols = [col for col in DECK_LAYOUT_COLUMNS if col in layout.columns]
    out = out.merge(layout[layout_cols], on="Deck", how="left", suffixes=("", "_DeckLayout"))

    fallback = built_in_deck_layout().set_index("Deck")
    for deck, row in fallback.iterrows():
        mask = out["Deck"].eq(deck)
        for col in [
            "DeckOrdinal",
            "ApproxVerticalDistanceToBoatDeck",
            "SpatialAccessTier",
            "CabinZone",
            "ClassArea",
            "LifeboatAccessScore",
            "StaircaseAccessScore",
        ]:
            if col not in out.columns:
                out[col] = np.nan
            out.loc[mask, col] = out.loc[mask, col].fillna(row[col])

    out["DeckOrdinal"] = pd.to_numeric(out["DeckOrdinal"], errors="coerce")
    out["ApproxVerticalDistanceToBoatDeck"] = pd.to_numeric(
        out["ApproxVerticalDistanceToBoatDeck"],
        errors="coerce",
    )
    out["LifeboatAccessScore"] = pd.to_numeric(out["LifeboatAccessScore"], errors="coerce")
    out["StaircaseAccessScore"] = pd.to_numeric(out["StaircaseAccessScore"], errors="coerce")

    out["SpatialAccessTier"] = out["SpatialAccessTier"].fillna(
        out["Deck"].map(spatial_access_tier)
    )
    out["CabinZone"] = out["CabinZone"].fillna("unknown")
    out["ClassArea"] = out["ClassArea"].fillna("unknown")
    out["ClassDeckConsistency"] = [
        class_deck_consistency(pclass, deck)
        for pclass, deck in zip(out["Pclass"], out["Deck"], strict=False)
    ]

    for col in SPATIAL_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[SPATIAL_OUTPUT_COLUMNS]
