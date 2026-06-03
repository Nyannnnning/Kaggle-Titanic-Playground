from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_feature_frame
from src.geometry import (
    write_passenger_clusters,
    write_prototype_profiles,
)
from src.model import predict_survival_probability


DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/titanic_model.pkl")
REPORTS_DIR = Path("reports")


def dominant_value(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return None
    return clean.mode().iloc[0]


def section(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        body = "(no data)"
    else:
        body = df.to_string(index=True)
    return f"\n## {title}\n\n```text\n{body}\n```\n"


def survival_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)["Survived"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "survival_rate"})
        .sort_values("survival_rate", ascending=False)
    )


def cluster_summary(features: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(clusters[["PassengerId", "ClusterId"]], on="PassengerId", how="left")
    grouped = merged.groupby("ClusterId", dropna=False)
    return grouped.agg(
        cluster_size=("PassengerId", "count"),
        survival_rate=("Survived", "mean"),
        dominant_Pclass=("Pclass", dominant_value),
        dominant_Sex=("Sex", dominant_value),
        dominant_Title=("Title", dominant_value),
        average_Age=("Age", "mean"),
        average_Fare=("Fare", "mean"),
        average_FamilySize=("FamilySize", "mean"),
        dominant_Deck=("Deck", dominant_value),
        dominant_InferredOriginRegion=("InferredOriginRegion", dominant_value),
        dominant_InferredPrimaryLanguage=("InferredPrimaryLanguage", dominant_value),
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(DATA_PATH)
    features = build_feature_frame(train_df)
    features["Survived"] = train_df["Survived"].astype(int)

    report_parts = ["# Titanic Exploratory Summary\n"]
    report_parts.append(section("Survival Rate by Deck", survival_rate_by(features, "Deck")))
    report_parts.append(
        section(
            "Survival Rate by SpatialAccessTier",
            survival_rate_by(features, "SpatialAccessTier"),
        )
    )
    report_parts.append(
        section(
            "Survival Rate by ClassDeckConsistency",
            survival_rate_by(features, "ClassDeckConsistency"),
        )
    )
    report_parts.append(
        section("Survival Rate by CabinKnown", survival_rate_by(features, "CabinKnown"))
    )
    report_parts.append(
        section(
            "Average DeckOrdinal by Survived",
            features.groupby("Survived")["DeckOrdinal"].mean().to_frame("avg_deck_ordinal"),
        )
    )
    report_parts.append(
        section(
            "Cabin Missing Survival Rate",
            survival_rate_by(features, "CabinMissing"),
        )
    )
    report_parts.append(
        section(
            "Pclass + Deck Count",
            pd.crosstab(features["Pclass"], features["Deck"], dropna=False),
        )
    )
    report_parts.append(
        section(
            "Sex + SpatialAccessTier Count",
            pd.crosstab(features["Sex"], features["SpatialAccessTier"], dropna=False),
        )
    )

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        X = train_df.drop(columns=["Survived"])
        y = train_df["Survived"].astype(int)
        probabilities = predict_survival_probability(model, X)
        predictions = model.predict(X).astype(int)

        clusters = write_passenger_clusters(
            model,
            X,
            y,
            probabilities,
            REPORTS_DIR / "passenger_clusters.csv",
        )
        summary = cluster_summary(features, clusters)
        summary.to_csv(REPORTS_DIR / "cluster_summary.csv")
        report_parts.append(section("Cluster Summary", summary))

        write_prototype_profiles(
            X,
            y,
            predictions,
            probabilities,
            REPORTS_DIR / "prototype_profiles.json",
        )
    else:
        report_parts.append("\n## Cluster Summary\n\nModel not found; skipping cluster analysis.\n")

    (REPORTS_DIR / "exploratory_summary.md").write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )
    print(f"Exploratory summary saved to: {REPORTS_DIR / 'exploratory_summary.md'}")
    if MODEL_PATH.exists():
        print(f"Passenger clusters saved to: {REPORTS_DIR / 'passenger_clusters.csv'}")
        print(f"Cluster summary saved to: {REPORTS_DIR / 'cluster_summary.csv'}")
        print(f"Prototype profiles saved to: {REPORTS_DIR / 'prototype_profiles.json'}")


if __name__ == "__main__":
    main()
