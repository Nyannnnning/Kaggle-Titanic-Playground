from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GROUP_SCOPES = {
    "Ticket": "TicketNormalized",
    "Family": "FamilyKey",
    "FamilyTicket": "FamilyTicketKey",
}

PRIMARY_SCOPE_PRIORITY = ["FamilyTicket", "Ticket", "Family"]


def support_bin(value: object) -> str:
    if pd.isna(value):
        return "0"
    count = int(value)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def family_ticket_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["FamilyKey"].fillna("Unknown").astype(str)
        + "_"
        + frame["TicketNormalized"].fillna("UNKNOWN").astype(str)
    )


def ensure_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in {
        "TicketNormalized": "UNKNOWN",
        "FamilyKey": "Unknown",
    }.items():
        if col not in out.columns:
            out[col] = default
    if "FamilyTicketKey" not in out.columns:
        out["FamilyTicketKey"] = family_ticket_key(out)
    return out


def _strict_signal(known_count: object, survivor_count: object) -> str:
    if pd.isna(known_count) or int(known_count) <= 0:
        return "unknown"
    known = int(known_count)
    survived = int(0 if pd.isna(survivor_count) else survivor_count)
    if survived <= 0:
        return "all_died"
    if survived >= known:
        return "all_survived"
    return "mixed"


def _signal_value(signal: str) -> int:
    if signal == "all_survived":
        return 1
    if signal == "all_died":
        return -1
    return 0


def fit_group_fate_stats(
    df: pd.DataFrame,
    y: pd.Series | np.ndarray | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit group-level survival counts from a training fold only."""
    if y is None:
        return {}

    out = ensure_group_keys(df).reset_index(drop=True)
    target = pd.Series(y).reset_index(drop=True).astype(int)
    out["_SurvivedTarget"] = target

    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for scope, key_col in GROUP_SCOPES.items():
        grouped = out.groupby(key_col, dropna=False)["_SurvivedTarget"].agg(
            known_count="count",
            survivor_count="sum",
        )
        stats[scope] = {
            "known_count": grouped["known_count"].to_dict(),
            "survivor_count": grouped["survivor_count"].to_dict(),
        }
    return stats


def _mapped_counts(
    frame: pd.DataFrame,
    scope: str,
    key_col: str,
    group_fate_stats: dict[str, dict[str, dict[str, Any]]] | None,
) -> tuple[pd.Series, pd.Series]:
    scope_stats = (group_fate_stats or {}).get(scope, {})
    known = frame[key_col].map(scope_stats.get("known_count", {})).fillna(0)
    survived = frame[key_col].map(scope_stats.get("survivor_count", {})).fillna(0)
    return known.astype(float), survived.astype(float)


def _leave_one_out_counts(
    frame: pd.DataFrame,
    y: pd.Series | np.ndarray,
    key_col: str,
) -> tuple[pd.Series, pd.Series]:
    target = pd.Series(y).reset_index(drop=True).astype(int)
    grouped = pd.DataFrame(
        {
            key_col: frame[key_col].reset_index(drop=True),
            "_SurvivedTarget": target,
        }
    )
    totals = grouped.groupby(key_col, dropna=False)["_SurvivedTarget"].transform("count")
    survivors = grouped.groupby(key_col, dropna=False)["_SurvivedTarget"].transform("sum")
    known = (totals - 1).clip(lower=0).astype(float)
    survived = (survivors - target).clip(lower=0).astype(float)
    return known, survived


def _add_scope_fate(
    out: pd.DataFrame,
    scope: str,
    key_col: str,
    group_fate_stats: dict[str, dict[str, dict[str, Any]]] | None,
    y: pd.Series | np.ndarray | None,
) -> pd.DataFrame:
    if y is not None:
        known, survived = _leave_one_out_counts(out, y, key_col)
    else:
        known, survived = _mapped_counts(out, scope, key_col, group_fate_stats)

    rate = survived / known.replace(0, np.nan)
    rate = rate.fillna(0.5)
    signals = [
        _strict_signal(known_count, survivor_count)
        for known_count, survivor_count in zip(known, survived)
    ]

    out[f"{scope}FateKnown"] = known.gt(0).astype(int)
    out[f"{scope}FateKnownCount"] = known
    out[f"{scope}FateSurvivorCount"] = survived
    out[f"{scope}FateSurvivalRate"] = rate
    out[f"{scope}FateSignal"] = signals
    out[f"{scope}FateSignalValue"] = [_signal_value(signal) for signal in signals]
    out[f"{scope}FateSupportBin"] = known.map(support_bin)
    return out


def _primary_scope(row: pd.Series) -> str:
    for scope in PRIMARY_SCOPE_PRIORITY:
        if int(row.get(f"{scope}FateKnown", 0)) == 1:
            return scope
    return "Unknown"


def _primary_value(row: pd.Series, metric: str) -> object:
    scope = row.get("PrimaryFateScope", "Unknown")
    if scope == "Unknown":
        if metric == "FateSignal":
            return "unknown"
        if metric == "FateSupportBin":
            return "0"
        if metric == "FateSurvivalRate":
            return 0.5
        return 0
    return row.get(f"{scope}{metric}")


def add_group_fate_features(
    df: pd.DataFrame,
    group_fate_stats: dict[str, dict[str, dict[str, Any]]] | None = None,
    y: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Add target-safe group fate features.

    When `y` is provided, every row uses leave-one-out counts from its own
    group. When `y` is absent, only fitted training-fold group counts are used.
    """
    out = ensure_group_keys(df)
    for scope, key_col in GROUP_SCOPES.items():
        out = _add_scope_fate(
            out,
            scope,
            key_col,
            group_fate_stats=group_fate_stats,
            y=y,
        )

    out["PrimaryFateScope"] = out.apply(_primary_scope, axis=1)
    out["PrimaryFateKnown"] = out["PrimaryFateScope"].ne("Unknown").astype(int)
    out["PrimaryFateKnownCount"] = out.apply(
        lambda row: _primary_value(row, "FateKnownCount"),
        axis=1,
    )
    out["PrimaryFateSurvivorCount"] = out.apply(
        lambda row: _primary_value(row, "FateSurvivorCount"),
        axis=1,
    )
    out["PrimaryFateSurvivalRate"] = out.apply(
        lambda row: _primary_value(row, "FateSurvivalRate"),
        axis=1,
    )
    out["PrimaryFateSignal"] = out.apply(
        lambda row: _primary_value(row, "FateSignal"),
        axis=1,
    )
    out["PrimaryFateSignalValue"] = out["PrimaryFateSignal"].map(_signal_value)
    out["PrimaryFateSupportBin"] = out.apply(
        lambda row: _primary_value(row, "FateSupportBin"),
        axis=1,
    )
    return out
