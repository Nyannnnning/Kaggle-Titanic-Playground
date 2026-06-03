from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_child_proxy(frame: pd.DataFrame) -> pd.Series:
    age = frame["Age"] if "Age" in frame.columns else pd.Series(np.nan, index=frame.index)
    title = (
        frame["Title"]
        if "Title" in frame.columns
        else pd.Series("Unknown", index=frame.index)
    )
    return (age.le(12).fillna(False) | title.eq("Master")).astype(int)


def _is_adult_male(frame: pd.DataFrame) -> pd.Series:
    sex = frame["Sex"] if "Sex" in frame.columns else pd.Series("Unknown", index=frame.index)
    age = frame["Age"] if "Age" in frame.columns else pd.Series(np.nan, index=frame.index)
    title = (
        frame["Title"]
        if "Title" in frame.columns
        else pd.Series("Unknown", index=frame.index)
    )
    return (sex.eq("male") & (age.ge(18).fillna(False) | title.eq("Mr"))).astype(int)


def _mode_or_unknown(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return "Unknown"
    return clean.mode().iloc[0]


def count_bin(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    count = int(value)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def ratio_bin(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    ratio = float(value)
    if ratio <= 0:
        return "none"
    if ratio < 0.5:
        return "minority"
    if ratio == 0.5:
        return "half"
    if ratio < 1:
        return "majority"
    return "all"


def fit_ticket_group_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = df.copy()
    for col, default in {
        "TicketNormalized": "UNKNOWN",
        "FamilyKey": "Unknown",
        "FamilySize": 1,
        "Sex": "Unknown",
        "Title": "Unknown",
        "Age": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    out["TicketChildProxy"] = _is_child_proxy(out)
    out["TicketFemaleProxy"] = out["Sex"].eq("female").astype(int)
    out["TicketMaleProxy"] = out["Sex"].eq("male").astype(int)
    out["TicketAdultMaleProxy"] = _is_adult_male(out)
    out["TicketMasterProxy"] = out["Title"].eq("Master").astype(int)

    grouped = out.groupby("TicketNormalized", dropna=False).agg(
        TicketGroupSize=("TicketNormalized", "size"),
        TicketChildCount=("TicketChildProxy", "sum"),
        TicketFemaleCount=("TicketFemaleProxy", "sum"),
        TicketMaleCount=("TicketMaleProxy", "sum"),
        TicketAdultMaleCount=("TicketAdultMaleProxy", "sum"),
        TicketMasterCount=("TicketMasterProxy", "sum"),
        TicketDistinctFamilyCount=("FamilyKey", "nunique"),
        TicketMaxFamilySize=("FamilySize", "max"),
        TicketPclassMode=("Pclass", _mode_or_unknown),
        TicketEmbarkedMode=("Embarked", _mode_or_unknown),
    )

    return {
        column: grouped[column].to_dict()
        for column in grouped.columns
    }


def _map_or_fallback(
    ticket: pd.Series,
    mapping: dict[str, Any] | None,
    fallback: pd.Series | int | float,
) -> pd.Series:
    if mapping is None:
        if isinstance(fallback, pd.Series):
            return fallback
        return pd.Series(fallback, index=ticket.index)
    mapped = ticket.map(mapping)
    if isinstance(fallback, pd.Series):
        return mapped.combine_first(fallback)
    return mapped.fillna(fallback)


def _composition_type(row: pd.Series) -> str:
    group_size = int(row.get("TicketGroupSize", 1))
    child_count = int(row.get("TicketChildCount", 0))
    female_count = int(row.get("TicketFemaleCount", 0))
    adult_male_count = int(row.get("TicketAdultMaleCount", 0))

    if group_size <= 1:
        return "solo_ticket"
    if child_count > 0 and female_count > 0 and adult_male_count > 0:
        return "mixed_family_like_with_child"
    if child_count > 0 and adult_male_count == 0:
        return "protected_group_with_child"
    if child_count > 0:
        return "child_with_adult_male"
    if female_count > 0 and adult_male_count > 0:
        return "mixed_adult_group"
    if female_count > 0:
        return "female_group"
    if adult_male_count > 0:
        return "adult_male_group"
    return "shared_ticket_other"


def _family_pattern(row: pd.Series) -> str:
    group_size = int(row.get("TicketGroupSize", 1))
    distinct_family_count = int(row.get("TicketDistinctFamilyCount", 1))
    max_family_size = int(row.get("TicketMaxFamilySize", 1))

    if group_size <= 1:
        return "solo_ticket"
    if distinct_family_count <= 1 and max_family_size > 1:
        return "family_ticket_aligned"
    if distinct_family_count > 1 and max_family_size <= 1:
        return "shared_nonfamily_ticket"
    if distinct_family_count > 1 and max_family_size > 1:
        return "mixed_family_nonfamily_ticket"
    return "shared_ticket_unknown_family"


def _child_female_pattern(row: pd.Series) -> str:
    has_child = bool(row.get("TicketHasChild", 0))
    has_female = bool(row.get("TicketHasFemale", 0))
    has_adult_male = bool(row.get("TicketHasAdultMale", 0))
    if has_child and has_female and has_adult_male:
        return "child_female_adult_male"
    if has_child and has_female:
        return "child_female_no_adult_male"
    if has_child and has_adult_male:
        return "child_adult_male_no_female"
    if has_female and has_adult_male:
        return "female_adult_male_no_child"
    if has_child:
        return "child_only_or_child_male"
    if has_female:
        return "female_no_child"
    if has_adult_male:
        return "adult_male_only"
    return "no_child_no_female_no_adult_male"


def add_ticket_group_features(
    df: pd.DataFrame,
    ticket_group_stats: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Add target-free ticket group composition features."""
    out = df.copy()
    for col, default in {
        "TicketNormalized": "UNKNOWN",
        "FamilyKey": "Unknown",
        "FamilySize": 1,
        "Sex": "Unknown",
        "Title": "Unknown",
        "Age": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    if ticket_group_stats is None:
        ticket_group_stats = fit_ticket_group_stats(out)

    ticket = out["TicketNormalized"]
    self_child = _is_child_proxy(out)
    self_female = out["Sex"].eq("female").astype(int)
    self_male = out["Sex"].eq("male").astype(int)
    self_adult_male = _is_adult_male(out)
    self_master = out["Title"].eq("Master").astype(int)

    fallback_values: dict[str, pd.Series | int] = {
        "TicketGroupSize": pd.Series(1, index=out.index),
        "TicketChildCount": self_child,
        "TicketFemaleCount": self_female,
        "TicketMaleCount": self_male,
        "TicketAdultMaleCount": self_adult_male,
        "TicketMasterCount": self_master,
        "TicketDistinctFamilyCount": pd.Series(1, index=out.index),
        "TicketMaxFamilySize": out["FamilySize"].fillna(1),
    }

    for col, fallback in fallback_values.items():
        out[col] = _map_or_fallback(
            ticket,
            ticket_group_stats.get(col),
            fallback,
        ).astype(float)

    out["TicketGroupSize"] = out["TicketGroupSize"].clip(lower=1)
    out["TicketChildRatio"] = out["TicketChildCount"] / out["TicketGroupSize"]
    out["TicketFemaleRatio"] = out["TicketFemaleCount"] / out["TicketGroupSize"]
    out["TicketAdultMaleRatio"] = out["TicketAdultMaleCount"] / out["TicketGroupSize"]

    out["TicketHasChild"] = out["TicketChildCount"].gt(0).astype(int)
    out["TicketHasFemale"] = out["TicketFemaleCount"].gt(0).astype(int)
    out["TicketHasAdultMale"] = out["TicketAdultMaleCount"].gt(0).astype(int)
    out["TicketHasFamily"] = out["TicketMaxFamilySize"].gt(1).astype(int)
    out["TicketIsMixedSex"] = (
        out["TicketFemaleCount"].gt(0) & out["TicketMaleCount"].gt(0)
    ).astype(int)

    out["TicketChildCountBin"] = out["TicketChildCount"].map(count_bin)
    out["TicketFemaleCountBin"] = out["TicketFemaleCount"].map(count_bin)
    out["TicketChildRatioBin"] = out["TicketChildRatio"].map(ratio_bin)
    out["TicketFemaleRatioBin"] = out["TicketFemaleRatio"].map(ratio_bin)
    out["TicketCompositionType"] = out.apply(_composition_type, axis=1)
    out["TicketFamilyPattern"] = out.apply(_family_pattern, axis=1)
    out["TicketChildFemalePattern"] = out.apply(_child_female_pattern, axis=1)

    return out
