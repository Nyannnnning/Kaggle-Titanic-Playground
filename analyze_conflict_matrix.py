from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from analyze_rule_candidates import add_rule_columns
from optimize_submission import CURRENT_BEST_STRATEGY, predict_strategy
from src.child_groups import add_child_group_features
from src.features import PassengerFeatureBuilder
from src.regime_rules import (
    has_any_all_died_group_fate,
    has_any_all_survived_group_fate,
    p1_isham_like_older_solo_miss_midfare,
    p1_luxury_small_family_mixed_fate,
    p3_master_small_family_all_survived,
    p3_master_small_family_mixed_without_all_died,
    p3_master_small_family_without_all_died,
)


TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
REPORTS_DIR = Path("reports")

MaskFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ConflictSpec:
    name: str
    regime: str
    positive_evidence: str
    negative_evidence: str
    mask_fn: MaskFn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build evidence-conflict reports for Titanic entity rules."
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--strategy",
        default=CURRENT_BEST_STRATEGY,
        help=f"Current-best strategy to audit. Default: {CURRENT_BEST_STRATEGY}",
    )
    return parser.parse_args()


def p3_child_evidence(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(3)
        & (
            features["Title"].eq("Master")
            | pd.to_numeric(features["Age"], errors="coerce").le(12)
        )
    )


def p3_female_evidence(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(3) & features["Sex"].eq("female")


def p3_large_family_death_evidence(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(3) & pd.to_numeric(
        features["FamilySize"],
        errors="coerce",
    ).ge(5)


def p3_lowfare_no_ticket_child(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(3)
        & features["FareBin"].eq("LowFare")
        & pd.to_numeric(features["TicketChildCount"], errors="coerce").eq(0)
        & ~pd.to_numeric(features["Age"], errors="coerce").le(12).fillna(False)
    )


def p1_female_evidence(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(1) & features["Sex"].eq("female")


def p1_first_class_male(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(1) & features["Sex"].eq("male")


def p1_adult_male_or_no_female_companion(features: pd.DataFrame) -> pd.Series:
    return features["TicketChildFemalePattern"].eq("adult_male_only") | features[
        "CompanionFemaleCountBin"
    ].eq("0")


def high_fare_evidence(features: pd.DataFrame) -> pd.Series:
    return features["FareBin"].isin(["HighFare", "LuxuryFare"])


def conflict_specs() -> list[ConflictSpec]:
    return [
        ConflictSpec(
            name="p3_child_vs_group_all_died",
            regime="p3_survival_rescue",
            positive_evidence="child_or_master",
            negative_evidence="any_all_died_group_fate",
            mask_fn=lambda f: p3_child_evidence(f) & has_any_all_died_group_fate(f),
        ),
        ConflictSpec(
            name="p3_master_small_family_no_all_died",
            regime="p3_survival_rescue",
            positive_evidence="master_small_family",
            negative_evidence="no_all_died_group_fate",
            mask_fn=p3_master_small_family_without_all_died,
        ),
        ConflictSpec(
            name="p3_master_small_family_all_survived_split",
            regime="p3_survival_rescue",
            positive_evidence="master_small_family_all_survived_group",
            negative_evidence="no_all_died_group_fate",
            mask_fn=p3_master_small_family_all_survived,
        ),
        ConflictSpec(
            name="p3_master_small_family_mixed_split",
            regime="p3_survival_rescue",
            positive_evidence="master_small_family_mixed_group",
            negative_evidence="no_all_died_group_fate",
            mask_fn=p3_master_small_family_mixed_without_all_died,
        ),
        ConflictSpec(
            name="p3_female_vs_large_family",
            regime="p3_survival_rescue",
            positive_evidence="female",
            negative_evidence="large_family",
            mask_fn=lambda f: p3_female_evidence(f) & p3_large_family_death_evidence(f),
        ),
        ConflictSpec(
            name="p3_all_survived_group_vs_lowfare_no_child",
            regime="p3_survival_rescue",
            positive_evidence="any_all_survived_group_fate",
            negative_evidence="lowfare_no_ticket_child",
            mask_fn=lambda f: (
                f["Pclass"].eq(3)
                & has_any_all_survived_group_fate(f)
                & p3_lowfare_no_ticket_child(f)
            ),
        ),
        ConflictSpec(
            name="p1_female_vs_any_all_died_group",
            regime="p1_death_override",
            positive_evidence="first_class_female",
            negative_evidence="any_all_died_group_fate",
            mask_fn=lambda f: p1_female_evidence(f) & has_any_all_died_group_fate(f),
        ),
        ConflictSpec(
            name="p1_female_luxury_mixed_family",
            regime="p1_death_override",
            positive_evidence="first_class_female_luxury",
            negative_evidence="mixed_family_ticket_fate",
            mask_fn=p1_luxury_small_family_mixed_fate,
        ),
        ConflictSpec(
            name="p1_isham_like_older_solo_miss_midfare",
            regime="p1_death_override",
            positive_evidence="first_class_female",
            negative_evidence="older_solo_miss_midfare",
            mask_fn=p1_isham_like_older_solo_miss_midfare,
        ),
        ConflictSpec(
            name="p1_male_vs_adult_male_object",
            regime="p1_death_override",
            positive_evidence="first_class",
            negative_evidence="adult_male_or_no_female_companion",
            mask_fn=lambda f: p1_first_class_male(f)
            & p1_adult_male_or_no_female_companion(f),
        ),
        ConflictSpec(
            name="high_fare_vs_group_all_died",
            regime="fare_group_conflict",
            positive_evidence="high_or_luxury_fare",
            negative_evidence="any_all_died_group_fate",
            mask_fn=lambda f: high_fare_evidence(f) & has_any_all_died_group_fate(f),
        ),
        ConflictSpec(
            name="low_fare_vs_group_all_survived",
            regime="fare_group_conflict",
            positive_evidence="any_all_survived_group_fate",
            negative_evidence="very_low_or_low_fare",
            mask_fn=lambda f: f["FareBin"].isin(["VeryLowFare", "LowFare"])
            & has_any_all_survived_group_fate(f),
        ),
    ]


def passenger_ids(frame: pd.DataFrame, mask: pd.Series) -> str:
    if "PassengerId" not in frame.columns:
        return ""
    return "|".join(frame.loc[mask, "PassengerId"].astype(int).astype(str))


def summary_rows(
    features: pd.DataFrame,
    specs: list[ConflictSpec],
    prediction: pd.Series,
    y: pd.Series | None,
    dataset: str,
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        mask = spec.mask_fn(features).fillna(False).astype(bool)
        predicted_alive = prediction.loc[mask].eq(1)
        row = {
            "dataset": dataset,
            "conflict": spec.name,
            "regime": spec.regime,
            "positive_evidence": spec.positive_evidence,
            "negative_evidence": spec.negative_evidence,
            "count": int(mask.sum()),
            "current_best_predicted_survivors": int(predicted_alive.sum()),
            "current_best_predicted_deaths": int((~predicted_alive).sum()),
            "passenger_ids": passenger_ids(features, mask),
        }
        if y is not None:
            survived = int(y.loc[mask].sum())
            current_correct = int(prediction.loc[mask].eq(y.loc[mask]).sum())
            row.update(
                {
                    "survived": survived,
                    "died": int(mask.sum()) - survived,
                    "survival_rate": float(y.loc[mask].mean()) if int(mask.sum()) else np.nan,
                    "current_best_accuracy": current_correct / int(mask.sum())
                    if int(mask.sum())
                    else np.nan,
                    "current_best_correct": current_correct,
                    "current_best_errors": int(mask.sum()) - current_correct,
                }
            )
        rows.append(row)
    return rows


def example_frame(
    features: pd.DataFrame,
    specs: list[ConflictSpec],
    prediction: pd.Series,
    dataset: str,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    columns = [
        "PassengerId",
        "Name",
        "Sex",
        "Pclass",
        "Title",
        "Age",
        "AgeBin",
        "Fare",
        "FareBin",
        "FamilySize",
        "FamilySizeBin",
        "Ticket",
        "TicketNormalized",
        "TicketGroupSize",
        "TicketChildFemalePattern",
        "PrimaryFateScope",
        "PrimaryFateSignal",
        "PrimaryFateSupportBin",
        "FamilyTicketFateSignal",
        "PrimaryCompanionScope",
        "PrimaryGroupChildFemalePattern",
        "Cabin",
        "Deck",
    ]
    columns = [column for column in columns if column in features.columns]
    for spec in specs:
        mask = spec.mask_fn(features).fillna(False).astype(bool)
        examples = features.loc[mask, columns].copy()
        if examples.empty:
            continue
        examples.insert(0, "dataset", dataset)
        examples.insert(1, "conflict", spec.name)
        examples.insert(2, "regime", spec.regime)
        examples["CurrentBestPrediction"] = prediction.loc[mask].astype(int).to_numpy()
        if y is not None:
            examples["Survived"] = y.loc[mask].astype(int).to_numpy()
            examples["CurrentBestCorrect"] = examples["CurrentBestPrediction"].eq(
                examples["Survived"]
            )
        rows.append(examples)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_features(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "Survived" not in train_df.columns:
        raise ValueError(f"{train_path} must contain Survived.")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)
    builder = PassengerFeatureBuilder(feature_set="full")
    train_features = add_rule_columns(add_child_group_features(builder.fit_transform(X, y)))
    test_features = add_rule_columns(add_child_group_features(builder.transform(test_df)))
    return train_features, y, test_features


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_features, y, test_features = build_features(args.train, args.test)
    train_prediction = predict_strategy(train_features, args.strategy)
    test_prediction = predict_strategy(test_features, args.strategy)
    specs = conflict_specs()

    summary = pd.DataFrame(
        summary_rows(train_features, specs, train_prediction, y, "train")
        + summary_rows(test_features, specs, test_prediction, None, "test")
    )
    summary.to_csv(args.output_dir / "conflict_matrix_summary.csv", index=False)

    train_examples = example_frame(train_features, specs, train_prediction, "train", y)
    test_examples = example_frame(test_features, specs, test_prediction, "test")
    examples = pd.concat([train_examples, test_examples], ignore_index=True)
    examples.to_csv(args.output_dir / "conflict_matrix_examples.csv", index=False)

    p1_candidates = examples[
        examples["regime"].eq("p1_death_override")
        & examples["dataset"].eq("test")
        & examples["CurrentBestPrediction"].eq(1)
    ].copy()
    p1_candidates.to_csv(
        args.output_dir / "p1_death_override_candidates.csv",
        index=False,
    )

    print(f"Strategy: {args.strategy}")
    print(f"Train accuracy: {accuracy_score(y, train_prediction):.4f}")
    print(f"Conflict summary saved to: {args.output_dir / 'conflict_matrix_summary.csv'}")
    print(f"Conflict examples saved to: {args.output_dir / 'conflict_matrix_examples.csv'}")
    print(
        "P1 death override candidates saved to: "
        f"{args.output_dir / 'p1_death_override_candidates.csv'}"
    )


if __name__ == "__main__":
    main()
