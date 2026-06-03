from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GROUP_SCOPES = ("Family", "FamilyTicket")


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


def is_child_proxy(frame: pd.DataFrame) -> pd.Series:
    age = frame["Age"] if "Age" in frame.columns else pd.Series(np.nan, index=frame.index)
    title = (
        frame["Title"]
        if "Title" in frame.columns
        else pd.Series("Unknown", index=frame.index)
    )
    return (age.le(12).fillna(False) | title.eq("Master")).astype(int)


def is_adult_male(frame: pd.DataFrame) -> pd.Series:
    sex = frame["Sex"] if "Sex" in frame.columns else pd.Series("Unknown", index=frame.index)
    age = frame["Age"] if "Age" in frame.columns else pd.Series(np.nan, index=frame.index)
    title = (
        frame["Title"]
        if "Title" in frame.columns
        else pd.Series("Unknown", index=frame.index)
    )
    return (sex.eq("male") & (age.ge(18).fillna(False) | title.eq("Mr"))).astype(int)


def family_ticket_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["FamilyKey"].fillna("Unknown").astype(str)
        + "_"
        + frame["TicketNormalized"].fillna("UNKNOWN").astype(str)
    )


def _child_female_pattern(
    group_size: object,
    child_count: object,
    female_count: object,
    adult_male_count: object,
) -> str:
    if pd.isna(group_size):
        return "unknown_group"

    size = int(group_size)
    children = int(0 if pd.isna(child_count) else child_count)
    females = int(0 if pd.isna(female_count) else female_count)
    adult_males = int(0 if pd.isna(adult_male_count) else adult_male_count)

    if size <= 1:
        return "solo"
    if children > 0 and females > 0 and adult_males > 0:
        return "child_female_adult_male"
    if children > 0 and females > 0:
        return "child_female_no_adult_male"
    if children > 0 and adult_males > 0:
        return "child_adult_male_no_female"
    if females > 0 and adult_males > 0:
        return "female_adult_male_no_child"
    if children > 0:
        return "child_only_or_child_male"
    if females > 0:
        return "female_no_child"
    if adult_males > 0:
        return "adult_male_only"
    return "no_child_no_female_no_adult_male"


def _self_role(row: pd.Series) -> str:
    if int(row.get("SelfChildProxy", 0)) == 1:
        return "child"
    if int(row.get("SelfFemaleProxy", 0)) == 1:
        return "female"
    if int(row.get("SelfAdultMaleProxy", 0)) == 1:
        return "adult_male"
    return "other"


def _primary_scope(row: pd.Series) -> str:
    if row.get("FamilyTicketGroupSize", 1) > 1:
        return "family_ticket"
    if row.get("FamilySize", 1) > 1 and row.get("FamilyGroupSize", 1) > 1:
        return "family"
    if row.get("TicketGroupSize", 1) > 1:
        return "ticket"
    return "solo"


def _first_available(row: pd.Series, scope: str, metric: str) -> object:
    if scope == "family_ticket":
        return row.get(f"FamilyTicket{metric}")
    if scope == "family":
        return row.get(f"Family{metric}")
    if scope == "ticket":
        return row.get(f"Ticket{metric}")
    if metric == "GroupSize":
        return 1
    if metric == "ChildFemalePattern":
        return "solo"
    return 0


def _group_stats(frame: pd.DataFrame, key_col: str) -> pd.DataFrame:
    return frame.groupby(key_col, dropna=False).agg(
        GroupSize=(key_col, "size"),
        ChildCount=("SelfChildProxy", "sum"),
        FemaleCount=("SelfFemaleProxy", "sum"),
        MaleCount=("SelfMaleProxy", "sum"),
        AdultMaleCount=("SelfAdultMaleProxy", "sum"),
        MasterCount=("SelfMasterProxy", "sum"),
    )


def fit_travel_group_stats(df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    out = df.copy()
    for col, default in {
        "FamilyKey": "Unknown",
        "TicketNormalized": "UNKNOWN",
        "Sex": "Unknown",
        "Title": "Unknown",
        "Age": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    out["FamilyTicketKey"] = family_ticket_key(out)
    out["SelfChildProxy"] = is_child_proxy(out)
    out["SelfFemaleProxy"] = out["Sex"].eq("female").astype(int)
    out["SelfMaleProxy"] = out["Sex"].eq("male").astype(int)
    out["SelfAdultMaleProxy"] = is_adult_male(out)
    out["SelfMasterProxy"] = out["Title"].eq("Master").astype(int)

    grouped = {
        "Family": _group_stats(out, "FamilyKey"),
        "FamilyTicket": _group_stats(out, "FamilyTicketKey"),
    }
    return {
        scope: {column: stats[column].to_dict() for column in stats.columns}
        for scope, stats in grouped.items()
    }


def _map_group_stat(
    key: pd.Series,
    mapping: dict[str, Any] | None,
    fallback: pd.Series | int | float,
) -> pd.Series:
    if mapping is None:
        if isinstance(fallback, pd.Series):
            return fallback
        return pd.Series(fallback, index=key.index)
    mapped = key.map(mapping)
    if isinstance(fallback, pd.Series):
        return mapped.combine_first(fallback)
    return mapped.fillna(fallback)


def _add_scope_features(
    out: pd.DataFrame,
    scope: str,
    key_col: str,
    stats: dict[str, dict[str, Any]] | None,
) -> pd.DataFrame:
    key = out[key_col]
    self_values: dict[str, pd.Series | int] = {
        "GroupSize": pd.Series(1, index=out.index),
        "ChildCount": out["SelfChildProxy"],
        "FemaleCount": out["SelfFemaleProxy"],
        "MaleCount": out["SelfMaleProxy"],
        "AdultMaleCount": out["SelfAdultMaleProxy"],
        "MasterCount": out["SelfMasterProxy"],
    }

    for metric, fallback in self_values.items():
        out[f"{scope}{metric}"] = _map_group_stat(
            key,
            None if stats is None else stats.get(metric),
            fallback,
        ).astype(float)

    out[f"{scope}GroupSize"] = out[f"{scope}GroupSize"].clip(lower=1)
    size = out[f"{scope}GroupSize"].replace(0, np.nan)
    out[f"{scope}ChildRatio"] = out[f"{scope}ChildCount"] / size
    out[f"{scope}FemaleRatio"] = out[f"{scope}FemaleCount"] / size
    out[f"{scope}AdultMaleRatio"] = out[f"{scope}AdultMaleCount"] / size

    out[f"{scope}HasChild"] = out[f"{scope}ChildCount"].gt(0).astype(int)
    out[f"{scope}HasFemale"] = out[f"{scope}FemaleCount"].gt(0).astype(int)
    out[f"{scope}HasAdultMale"] = out[f"{scope}AdultMaleCount"].gt(0).astype(int)
    out[f"{scope}IsMixedSex"] = (
        out[f"{scope}FemaleCount"].gt(0) & out[f"{scope}MaleCount"].gt(0)
    ).astype(int)

    out[f"{scope}AccompanyingChildCount"] = (
        out[f"{scope}ChildCount"] - out["SelfChildProxy"]
    ).clip(lower=0)
    out[f"{scope}AccompanyingFemaleCount"] = (
        out[f"{scope}FemaleCount"] - out["SelfFemaleProxy"]
    ).clip(lower=0)
    out[f"{scope}AccompanyingAdultMaleCount"] = (
        out[f"{scope}AdultMaleCount"] - out["SelfAdultMaleProxy"]
    ).clip(lower=0)

    for metric in [
        "ChildCount",
        "FemaleCount",
        "AdultMaleCount",
        "AccompanyingChildCount",
        "AccompanyingFemaleCount",
        "AccompanyingAdultMaleCount",
    ]:
        out[f"{scope}{metric}Bin"] = out[f"{scope}{metric}"].map(count_bin)

    for metric in ["ChildRatio", "FemaleRatio", "AdultMaleRatio"]:
        out[f"{scope}{metric}Bin"] = out[f"{scope}{metric}"].map(ratio_bin)

    out[f"{scope}ChildFemalePattern"] = [
        _child_female_pattern(size, child, female, adult_male)
        for size, child, female, adult_male in zip(
            out[f"{scope}GroupSize"],
            out[f"{scope}ChildCount"],
            out[f"{scope}FemaleCount"],
            out[f"{scope}AdultMaleCount"],
        )
    ]

    return out


def add_travel_group_features(
    df: pd.DataFrame,
    travel_group_stats: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """Build entity-style group features from family and ticket objects.

    These features are target-free. They describe who travels together under
    conservative grouping keys rather than asserting true household structure.
    """
    out = df.copy()
    for col, default in {
        "FamilyKey": "Unknown",
        "TicketNormalized": "UNKNOWN",
        "Sex": "Unknown",
        "Title": "Unknown",
        "Age": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    if "FamilyTicketKey" not in out.columns:
        out["FamilyTicketKey"] = family_ticket_key(out)

    out["SelfChildProxy"] = is_child_proxy(out)
    out["SelfFemaleProxy"] = out["Sex"].eq("female").astype(int)
    out["SelfMaleProxy"] = out["Sex"].eq("male").astype(int)
    out["SelfAdultMaleProxy"] = is_adult_male(out)
    out["SelfMasterProxy"] = out["Title"].eq("Master").astype(int)
    out["SelfGroupRole"] = out.apply(_self_role, axis=1)

    if travel_group_stats is None:
        travel_group_stats = fit_travel_group_stats(out)

    out = _add_scope_features(
        out,
        "Family",
        "FamilyKey",
        travel_group_stats.get("Family"),
    )
    out = _add_scope_features(
        out,
        "FamilyTicket",
        "FamilyTicketKey",
        travel_group_stats.get("FamilyTicket"),
    )

    out["PrimaryCompanionScope"] = out.apply(_primary_scope, axis=1)
    out["CompanionGroupSize"] = out.apply(
        lambda row: _first_available(row, row["PrimaryCompanionScope"], "GroupSize"),
        axis=1,
    )
    for metric in [
        "ChildCount",
        "FemaleCount",
        "AdultMaleCount",
        "ChildRatio",
        "FemaleRatio",
        "AdultMaleRatio",
        "AccompanyingChildCount",
        "AccompanyingFemaleCount",
        "AccompanyingAdultMaleCount",
    ]:
        out[f"Companion{metric}"] = out.apply(
            lambda row, current_metric=metric: _first_available(
                row,
                row["PrimaryCompanionScope"],
                current_metric,
            ),
            axis=1,
        )

    out["PrimaryGroupChildFemalePattern"] = out.apply(
        lambda row: _first_available(
            row,
            row["PrimaryCompanionScope"],
            "ChildFemalePattern",
        ),
        axis=1,
    )
    out["ChildFemaleCompanionPattern"] = (
        out["SelfGroupRole"].astype(str)
        + "_in_"
        + out["PrimaryGroupChildFemalePattern"].astype(str)
    )

    for metric in [
        "ChildCount",
        "FemaleCount",
        "AdultMaleCount",
        "AccompanyingChildCount",
        "AccompanyingFemaleCount",
        "AccompanyingAdultMaleCount",
    ]:
        out[f"Companion{metric}Bin"] = out[f"Companion{metric}"].map(count_bin)

    for metric in ["ChildRatio", "FemaleRatio", "AdultMaleRatio"]:
        out[f"Companion{metric}Bin"] = out[f"Companion{metric}"].map(ratio_bin)

    return out
