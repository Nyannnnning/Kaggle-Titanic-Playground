from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


AGE_RELATIVE_BINS = [-math.inf, -1.0, -0.35, 0.35, 1.0, math.inf]
AGE_RELATIVE_LABELS = [
    "much_younger",
    "younger",
    "class_average",
    "older",
    "much_older",
]


@dataclass(frozen=True)
class AgePclassStats:
    mean: float
    std: float
    q25: float
    q50: float
    q75: float


def age_known(age: object) -> int:
    return int(pd.notna(age))


def age_missing(age: object) -> int:
    return int(pd.isna(age))


def fit_age_pclass_stats(df: pd.DataFrame) -> dict[int, AgePclassStats]:
    if "Pclass" not in df.columns or "Age" not in df.columns:
        return {}

    stats: dict[int, AgePclassStats] = {}
    for pclass, group in df.groupby("Pclass"):
        ages = group["Age"].dropna().astype(float)
        if ages.empty:
            continue
        std = float(ages.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        stats[int(pclass)] = AgePclassStats(
            mean=float(ages.mean()),
            std=std,
            q25=float(ages.quantile(0.25)),
            q50=float(ages.quantile(0.50)),
            q75=float(ages.quantile(0.75)),
        )
    return stats


def _coerce_stats(stats: dict[Any, Any] | None) -> dict[int, AgePclassStats]:
    if not stats:
        return {}
    coerced: dict[int, AgePclassStats] = {}
    for key, value in stats.items():
        if isinstance(value, AgePclassStats):
            coerced[int(key)] = value
        elif isinstance(value, dict):
            coerced[int(key)] = AgePclassStats(
                mean=float(value["mean"]),
                std=float(value["std"]),
                q25=float(value["q25"]),
                q50=float(value["q50"]),
                q75=float(value["q75"]),
            )
        else:
            raise TypeError(f"Unsupported age stats value: {type(value)!r}")
    return coerced


def _age_z(row: pd.Series, stats: dict[int, AgePclassStats]) -> float:
    age = row.get("Age")
    pclass = row.get("Pclass")
    if pd.isna(age) or pd.isna(pclass):
        return np.nan
    class_stats = stats.get(int(pclass))
    if class_stats is None:
        return np.nan
    return (float(age) - class_stats.mean) / class_stats.std


def _age_quartile(row: pd.Series, stats: dict[int, AgePclassStats]) -> str:
    age = row.get("Age")
    pclass = row.get("Pclass")
    if pd.isna(age):
        return "AgeMissing"
    if pd.isna(pclass):
        return "PclassMissing"

    class_stats = stats.get(int(pclass))
    if class_stats is None:
        return "AgeStatsMissing"

    value = float(age)
    if value <= class_stats.q25:
        return "Q1_youngest_in_class"
    if value <= class_stats.q50:
        return "Q2"
    if value <= class_stats.q75:
        return "Q3"
    return "Q4_oldest_in_class"


def _age_relative_bucket(z_value: object) -> str:
    if pd.isna(z_value):
        return "AgeMissing"
    value = float(z_value)
    if value <= AGE_RELATIVE_BINS[1]:
        return AGE_RELATIVE_LABELS[0]
    if value <= AGE_RELATIVE_BINS[2]:
        return AGE_RELATIVE_LABELS[1]
    if value <= AGE_RELATIVE_BINS[3]:
        return AGE_RELATIVE_LABELS[2]
    if value <= AGE_RELATIVE_BINS[4]:
        return AGE_RELATIVE_LABELS[3]
    return AGE_RELATIVE_LABELS[4]


def _age_missing_pattern(row: pd.Series) -> str:
    if not pd.isna(row.get("Age")):
        return "age_known"
    return f"P{row.get('Pclass', 'Unknown')}_{row.get('Title', 'Unknown')}_age_missing"


def add_age_object_features(
    df: pd.DataFrame,
    age_pclass_stats: dict[Any, Any] | None = None,
) -> pd.DataFrame:
    """Add class-relative age features.

    Age is interpreted both as an absolute life-stage signal and as a
    class-relative position. The stats should be fitted on the training fold.
    """
    out = df.copy()
    if "Age" not in out.columns:
        out["Age"] = np.nan
    if "Pclass" not in out.columns:
        out["Pclass"] = np.nan
    if "Title" not in out.columns:
        out["Title"] = "Unknown"
    if "Sex" not in out.columns:
        out["Sex"] = "Unknown"

    stats = _coerce_stats(age_pclass_stats)
    if not stats:
        stats = fit_age_pclass_stats(out)

    out["AgeKnown"] = out["Age"].map(age_known)
    out["AgeMissing"] = out["Age"].map(age_missing)
    out["AgeZWithinPclass"] = out.apply(_age_z, axis=1, stats=stats)
    out["AgeQuartileWithinPclass"] = out.apply(_age_quartile, axis=1, stats=stats)
    out["AgeRelativeBucketWithinPclass"] = out["AgeZWithinPclass"].map(
        _age_relative_bucket
    )
    out["AgeRelativeBucketWithinPclassSex"] = (
        out["AgeRelativeBucketWithinPclass"].astype(str)
        + "_"
        + out["Sex"].fillna("Unknown").astype(str)
    )
    out["AgeMissingPatternByPclassTitle"] = out.apply(_age_missing_pattern, axis=1)
    out["IsYoungWithinPclass"] = out["AgeRelativeBucketWithinPclass"].isin(
        ["much_younger", "younger"]
    ).astype(int)
    out["IsOldWithinPclass"] = out["AgeRelativeBucketWithinPclass"].isin(
        ["older", "much_older"]
    ).astype(int)
    out["IsOldMaleWithinPclass"] = (
        out["Sex"].eq("male") & out["IsOldWithinPclass"].eq(1)
    ).astype(int)
    out["IsYoungMasterWithinPclass"] = (
        out["Title"].eq("Master") & out["IsYoungWithinPclass"].eq(1)
    ).astype(int)

    return out
