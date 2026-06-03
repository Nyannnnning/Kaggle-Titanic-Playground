from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.child_groups import add_child_group_features
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


def grouped_survival_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            count=("PassengerId", "size"),
            survived=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
            average_age=("Age", "mean"),
            average_fare=("Fare", "mean"),
            average_family_size=("FamilySize", "mean"),
        )
        .sort_values(group_cols)
    )


def pclass3_child_group_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p3 = features[features["Pclass"].eq(3)].copy()

    tables = {
        "pclass3_sex_any_accompanying_child_count": grouped_survival_rate(
            p3,
            ["Sex", "AnyAccompanyingChildCountBin"],
        ),
        "pclass3_sex_ticket_accompanying_child_count": grouped_survival_rate(
            p3,
            ["Sex", "TicketAccompanyingChildCountBin"],
        ),
        "pclass3_sex_family_size_any_accompanying_child_count": grouped_survival_rate(
            p3,
            ["Sex", "FamilySizeBin", "AnyAccompanyingChildCountBin"],
        ),
        "pclass3_title_age_family_child_window": grouped_survival_rate(
            p3,
            ["Sex", "Title", "AgeBin", "FamilySizeBin", "AnyAccompanyingChildCountBin"],
        ),
    }

    return tables


def fare_object_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fare_by_family = (
        features.groupby(["Pclass", "FamilySizeBin"], dropna=False)
        .agg(
            count=("PassengerId", "size"),
            survival_rate=("Survived", "mean"),
            fare_median=("Fare", "median"),
            fare_mean=("Fare", "mean"),
            fare_per_ticket_median=("FarePerTicketMember", "median"),
            fare_per_family_median=("FarePerFamilyMember", "median"),
            fare_per_family_ticket_median=("FarePerFamilyTicketMember", "median"),
            ticket_group_median=("TicketGroupSize", "median"),
            family_ticket_group_median=("FamilyTicketGroupSize", "median"),
        )
        .sort_index()
    )

    travel_party_survival = grouped_survival_rate(
        features,
        ["Pclass", "TravelPartyType"],
    )

    fare_per_ticket_survival = grouped_survival_rate(
        features,
        ["Pclass", "FarePerTicketBin"],
    )

    alignment = grouped_survival_rate(
        features,
        ["Pclass", "FareObjectInterpretation"],
    )

    high_raw_fare_large_family = grouped_survival_rate(
        features,
        ["Pclass", "HighRawFareLargeFamilyFlag"],
    )

    return {
        "fare_object_by_pclass_family_size": fare_by_family,
        "survival_by_pclass_travel_party_type": travel_party_survival,
        "survival_by_pclass_fare_per_ticket_bin": fare_per_ticket_survival,
        "survival_by_pclass_fare_object_interpretation": alignment,
        "survival_by_pclass_high_raw_fare_large_family": high_raw_fare_large_family,
    }


def age_object_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    age_distribution = (
        features.groupby("Pclass", dropna=False)
        .agg(
            count=("PassengerId", "size"),
            age_known=("AgeKnown", "sum"),
            age_missing=("AgeMissing", "sum"),
            age_mean=("Age", "mean"),
            age_median=("Age", "median"),
            age_std=("Age", "std"),
            age_q25=("Age", lambda s: s.quantile(0.25)),
            age_q75=("Age", lambda s: s.quantile(0.75)),
            survival_rate=("Survived", "mean"),
        )
        .sort_index()
    )
    age_distribution["age_missing_rate"] = (
        age_distribution["age_missing"] / age_distribution["count"]
    )

    age_quartile_survival = grouped_survival_rate(
        features,
        ["Pclass", "AgeQuartileWithinPclass"],
    )

    age_relative_survival = grouped_survival_rate(
        features,
        ["Pclass", "AgeRelativeBucketWithinPclass"],
    )

    age_sex_relative_survival = grouped_survival_rate(
        features,
        ["Pclass", "Sex", "AgeRelativeBucketWithinPclass"],
    )

    age_missing_pattern = grouped_survival_rate(
        features,
        ["Pclass", "AgeMissingPatternByPclassTitle"],
    )

    return {
        "age_distribution_by_pclass": age_distribution,
        "survival_by_pclass_age_quartile": age_quartile_survival,
        "survival_by_pclass_age_relative_bucket": age_relative_survival,
        "survival_by_pclass_sex_age_relative_bucket": age_sex_relative_survival,
        "survival_by_pclass_age_missing_pattern": age_missing_pattern,
    }


def name_object_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pattern_summary = grouped_survival_rate(
        features,
        ["Pclass", "Sex", "FormalNamePattern"],
    )

    title_token_summary = grouped_survival_rate(
        features,
        ["Pclass", "FormalTitle", "FormalGivenTokenCountBin"],
    )

    surname_shape_summary = grouped_survival_rate(
        features,
        ["Pclass", "FormalSurnameShape"],
    )

    parenthetical_presence_summary = grouped_survival_rate(
        features,
        ["Pclass", "FormalTitle", "HasParentheticalName"],
    )

    first_token_summary = (
        features.groupby(["FormalTitle", "FormalGivenFirstToken"], dropna=False)
        .agg(
            count=("PassengerId", "size"),
            survived=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
            dominant_pclass=("Pclass", dominant_value),
            female_rate=("Sex", lambda s: s.eq("female").mean()),
        )
        .reset_index()
    )
    first_token_summary = first_token_summary[
        first_token_summary["count"].ge(5)
        & ~first_token_summary["FormalGivenFirstToken"].eq("Unknown")
    ].sort_values(["FormalTitle", "count"], ascending=[True, False])

    example_cols = [
        "PassengerId",
        "Survived",
        "Name",
        "FormalName",
        "FormalTitle",
        "FormalSurname",
        "FormalGivenNames",
        "FormalNamePattern",
        "FormalSurnameShape",
        "Pclass",
        "Sex",
        "Age",
    ]
    examples = features[example_cols].sort_values(["FormalTitle", "PassengerId"])

    return {
        "formal_name_pattern_summary": pattern_summary,
        "formal_name_title_token_summary": title_token_summary,
        "formal_name_surname_shape_summary": surname_shape_summary,
        "formal_name_parenthetical_presence_summary": parenthetical_presence_summary,
        "formal_name_top_first_tokens": first_token_summary,
        "formal_name_examples": examples,
    }


def ticket_group_pattern_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ticket_groups = (
        features.groupby("TicketNormalized", dropna=False)
        .agg(
            ticket_group_size=("PassengerId", "size"),
            survived=("Survived", "sum"),
            survival_rate=("Survived", "mean"),
            pclass_mode=("Pclass", dominant_value),
            embarked_mode=("Embarked", dominant_value),
            fare=("Fare", "first"),
            ticket_prefix=("TicketPrefix", dominant_value),
            ticket_child_count=("TicketChildCount", "first"),
            ticket_female_count=("TicketFemaleCount", "first"),
            ticket_adult_male_count=("TicketAdultMaleCount", "first"),
            ticket_distinct_family_count=("TicketDistinctFamilyCount", "first"),
            ticket_max_family_size=("TicketMaxFamilySize", "first"),
            ticket_child_ratio=("TicketChildRatio", "first"),
            ticket_female_ratio=("TicketFemaleRatio", "first"),
            ticket_composition_type=("TicketCompositionType", "first"),
            ticket_family_pattern=("TicketFamilyPattern", "first"),
            ticket_child_female_pattern=("TicketChildFemalePattern", "first"),
            travel_party_type=("TravelPartyType", "first"),
        )
        .reset_index()
    )
    ticket_groups["survival_pattern"] = "mixed"
    ticket_groups.loc[ticket_groups["survival_rate"].eq(0), "survival_pattern"] = "all_died"
    ticket_groups.loc[ticket_groups["survival_rate"].eq(1), "survival_pattern"] = (
        "all_survived"
    )

    shared_ticket_groups = ticket_groups[ticket_groups["ticket_group_size"].gt(1)].copy()

    def summarize_groups(group_cols: list[str]) -> pd.DataFrame:
        return (
            shared_ticket_groups.groupby(group_cols, dropna=False)
            .agg(
                ticket_group_count=("TicketNormalized", "size"),
                passenger_count=("ticket_group_size", "sum"),
                survived=("survived", "sum"),
                passenger_weighted_survival_rate=(
                    "survived",
                    lambda s: s.sum()
                    / shared_ticket_groups.loc[s.index, "ticket_group_size"].sum(),
                ),
                average_group_survival_rate=("survival_rate", "mean"),
                all_died_groups=(
                    "survival_pattern",
                    lambda s: s.eq("all_died").sum(),
                ),
                all_survived_groups=(
                    "survival_pattern",
                    lambda s: s.eq("all_survived").sum(),
                ),
                mixed_groups=("survival_pattern", lambda s: s.eq("mixed").sum()),
            )
            .sort_values("passenger_weighted_survival_rate", ascending=False)
        )

    passenger_composition = grouped_survival_rate(
        features,
        ["Pclass", "TicketCompositionType"],
    )
    passenger_family_pattern = grouped_survival_rate(
        features,
        ["Pclass", "TicketFamilyPattern"],
    )
    passenger_child_female = grouped_survival_rate(
        features,
        ["Pclass", "TicketChildFemalePattern"],
    )

    return {
        "ticket_group_survival_patterns": ticket_groups.sort_values(
            ["ticket_group_size", "TicketNormalized"],
            ascending=[False, True],
        ),
        "shared_ticket_group_summary_by_composition": summarize_groups(
            ["pclass_mode", "ticket_composition_type"]
        ),
        "shared_ticket_group_summary_by_family_pattern": summarize_groups(
            ["pclass_mode", "ticket_family_pattern"]
        ),
        "shared_ticket_group_summary_by_child_female_pattern": summarize_groups(
            ["pclass_mode", "ticket_child_female_pattern"]
        ),
        "survival_by_pclass_ticket_composition_type": passenger_composition,
        "survival_by_pclass_ticket_family_pattern": passenger_family_pattern,
        "survival_by_pclass_ticket_child_female_pattern": passenger_child_female,
    }


def travel_group_object_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary_pattern = grouped_survival_rate(
        features,
        ["Pclass", "Sex", "PrimaryCompanionScope", "PrimaryGroupChildFemalePattern"],
    )

    child_female_companion = grouped_survival_rate(
        features,
        ["Pclass", "ChildFemaleCompanionPattern"],
    )

    family_ticket_pattern = grouped_survival_rate(
        features,
        ["Pclass", "Sex", "FamilyTicketChildFemalePattern"],
    )

    pclass3_child_female = grouped_survival_rate(
        features[features["Pclass"].eq(3)],
        [
            "Sex",
            "Title",
            "FamilySizeBin",
            "PrimaryCompanionScope",
            "PrimaryGroupChildFemalePattern",
        ],
    )

    companion_counts = grouped_survival_rate(
        features,
        [
            "Pclass",
            "Sex",
            "CompanionAccompanyingChildCountBin",
            "CompanionAccompanyingFemaleCountBin",
            "CompanionAccompanyingAdultMaleCountBin",
        ],
    )

    examples = features[
        [
            "PassengerId",
            "Survived",
            "Name",
            "Pclass",
            "Sex",
            "Title",
            "Age",
            "FamilySize",
            "Ticket",
            "Fare",
            "PrimaryCompanionScope",
            "CompanionGroupSize",
            "CompanionChildCount",
            "CompanionFemaleCount",
            "CompanionAdultMaleCount",
            "PrimaryGroupChildFemalePattern",
            "ChildFemaleCompanionPattern",
            "FareObjectInterpretation",
            "TravelPartyType",
        ]
    ].sort_values(["Pclass", "PrimaryGroupChildFemalePattern", "PassengerId"])

    return {
        "survival_by_primary_companion_pattern": primary_pattern,
        "survival_by_child_female_companion_pattern": child_female_companion,
        "survival_by_family_ticket_child_female_pattern": family_ticket_pattern,
        "pclass3_child_female_companion_pattern": pclass3_child_female,
        "survival_by_companion_count_bins": companion_counts,
        "travel_group_object_examples": examples,
    }


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
    features = add_child_group_features(features)
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

    child_tables = pclass3_child_group_tables(features)
    for name, table in child_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Pclass 3 Survival by Sex and Any Accompanying Child Count",
            child_tables["pclass3_sex_any_accompanying_child_count"],
        )
    )
    report_parts.append(
        section(
            "Pclass 3 Survival by Sex and Ticket Accompanying Child Count",
            child_tables["pclass3_sex_ticket_accompanying_child_count"],
        )
    )
    report_parts.append(
        section(
            "Pclass 3 Survival by Sex, Family Size, and Any Accompanying Child Count",
            child_tables["pclass3_sex_family_size_any_accompanying_child_count"],
        )
    )
    report_parts.append(
        section(
            "Pclass 3 Title Age Family Child Window",
            child_tables["pclass3_title_age_family_child_window"],
        )
    )

    fare_tables = fare_object_tables(features)
    for name, table in fare_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Fare Object by Pclass and Family Size",
            fare_tables["fare_object_by_pclass_family_size"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and TravelPartyType",
            fare_tables["survival_by_pclass_travel_party_type"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and FarePerTicketBin",
            fare_tables["survival_by_pclass_fare_per_ticket_bin"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and FareObjectInterpretation",
            fare_tables["survival_by_pclass_fare_object_interpretation"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and HighRawFareLargeFamilyFlag",
            fare_tables["survival_by_pclass_high_raw_fare_large_family"],
        )
    )

    age_tables = age_object_tables(features)
    for name, table in age_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Age Distribution by Pclass",
            age_tables["age_distribution_by_pclass"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and Age Quartile Within Pclass",
            age_tables["survival_by_pclass_age_quartile"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and Age Relative Bucket Within Pclass",
            age_tables["survival_by_pclass_age_relative_bucket"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass, Sex, and Age Relative Bucket Within Pclass",
            age_tables["survival_by_pclass_sex_age_relative_bucket"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and Age Missing Pattern",
            age_tables["survival_by_pclass_age_missing_pattern"],
        )
    )

    name_tables = name_object_tables(features)
    for name, table in name_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Formal Name Pattern Summary",
            name_tables["formal_name_pattern_summary"],
        )
    )
    report_parts.append(
        section(
            "Formal Name Title Token Summary",
            name_tables["formal_name_title_token_summary"],
        )
    )
    report_parts.append(
        section(
            "Formal Name Surname Shape Summary",
            name_tables["formal_name_surname_shape_summary"],
        )
    )

    ticket_group_tables = ticket_group_pattern_tables(features)
    for name, table in ticket_group_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Shared Ticket Group Summary by Composition",
            ticket_group_tables["shared_ticket_group_summary_by_composition"],
        )
    )
    report_parts.append(
        section(
            "Shared Ticket Group Summary by Family Pattern",
            ticket_group_tables["shared_ticket_group_summary_by_family_pattern"],
        )
    )
    report_parts.append(
        section(
            "Survival by Pclass and Ticket Child/Female Pattern",
            ticket_group_tables["survival_by_pclass_ticket_child_female_pattern"],
        )
    )

    travel_group_tables = travel_group_object_tables(features)
    for name, table in travel_group_tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv")

    report_parts.append(
        section(
            "Survival by Primary Companion Pattern",
            travel_group_tables["survival_by_primary_companion_pattern"],
        )
    )
    report_parts.append(
        section(
            "Survival by Child/Female Companion Pattern",
            travel_group_tables["survival_by_child_female_companion_pattern"],
        )
    )
    report_parts.append(
        section(
            "Pclass 3 Child/Female Companion Pattern",
            travel_group_tables["pclass3_child_female_companion_pattern"],
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
