from __future__ import annotations

import pandas as pd

from .features import add_core_titanic_features


def child_count_bin(value: object) -> str:
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


def add_child_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add child-count context for passenger travel groups.

    FamilyKey is a weak proxy because unrelated passengers can share surname
    and class. Ticket-based counts are usually the more conservative signal.
    """
    required = {"Title", "TicketNormalized", "FamilyKey", "FamilySize"}
    out = df.copy()
    if not required.issubset(out.columns):
        out = add_core_titanic_features(out)

    if "Age" not in out.columns:
        out["Age"] = pd.NA

    out["IsChildProxy"] = (
        out["Age"].le(12).fillna(False) | out["Title"].eq("Master")
    ).astype(int)

    out["FamilyTicketKey"] = (
        out["FamilyKey"].fillna("Unknown").astype(str)
        + "_"
        + out["TicketNormalized"].fillna("UNKNOWN").astype(str)
    )

    group_keys = {
        "FamilyKey": "FamilyKey",
        "Ticket": "TicketNormalized",
        "FamilyTicket": "FamilyTicketKey",
    }
    for label, key in group_keys.items():
        child_count = out.groupby(key, dropna=False)["IsChildProxy"].transform("sum")
        out[f"{label}ChildCountTotal"] = child_count.fillna(0).astype(int)
        out[f"{label}AccompanyingChildCount"] = (
            out[f"{label}ChildCountTotal"] - out["IsChildProxy"]
        ).clip(lower=0).astype(int)
        out[f"{label}ChildCountTotalBin"] = out[f"{label}ChildCountTotal"].map(
            child_count_bin
        )
        out[f"{label}AccompanyingChildCountBin"] = out[
            f"{label}AccompanyingChildCount"
        ].map(child_count_bin)

    out["AnyGroupChildCountTotal"] = out[
        ["FamilyKeyChildCountTotal", "TicketChildCountTotal"]
    ].max(axis=1)
    out["AnyAccompanyingChildCount"] = out[
        ["FamilyKeyAccompanyingChildCount", "TicketAccompanyingChildCount"]
    ].max(axis=1)
    out["AnyGroupChildCountTotalBin"] = out["AnyGroupChildCountTotal"].map(
        child_count_bin
    )
    out["AnyAccompanyingChildCountBin"] = out["AnyAccompanyingChildCount"].map(
        child_count_bin
    )

    return out
