from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_submission import CURRENT_BEST_STRATEGY, predict_strategy
from src.features import PassengerFeatureBuilder
from src.model import build_model, predict_survival_probability


TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
REPORTS_DIR = Path("reports")

DEFAULT_MODEL_TYPES = [
    "logistic_score",
    "hist_gradient_boosting",
    "random_forest_optional",
    "xgboost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find passengers current best predicts dead but multiple models "
            "rank as likely survivors."
        )
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--feature-set", default="clean")
    parser.add_argument("--probability-threshold", type=float, default=0.60)
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument(
        "--strategy",
        default=CURRENT_BEST_STRATEGY,
        help=f"Rule strategy used as current best. Default: {CURRENT_BEST_STRATEGY}",
    )
    return parser.parse_args()


def build_rule_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)
    builder = PassengerFeatureBuilder(feature_set="full")
    builder.fit(X, y)
    return builder.transform(test_df)


def fit_model_probabilities(
    model_type: str,
    feature_set: str,
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
) -> tuple[str, np.ndarray] | None:
    try:
        model = build_model(model_type, feature_set=feature_set)
        model.fit(X, y)
        probabilities = predict_survival_probability(model, test_df)
    except ImportError as exc:
        print(f"Skipping {model_type}: {exc}")
        return None
    return model_type, probabilities


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    if "Survived" not in train_df.columns:
        raise ValueError(f"{args.train} must contain Survived.")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)
    rule_features = build_rule_features(train_df, test_df)
    current_best_prediction = predict_strategy(rule_features, args.strategy)

    probability_columns = {}
    for model_type in DEFAULT_MODEL_TYPES:
        result = fit_model_probabilities(model_type, args.feature_set, X, y, test_df)
        if result is None:
            continue
        name, probabilities = result
        probability_columns[f"{name}_probability"] = probabilities

    if not probability_columns:
        raise RuntimeError("No model probabilities were produced.")

    output = test_df.copy()
    output["CurrentBestPrediction"] = current_best_prediction.astype(int).to_numpy()
    for column, values in probability_columns.items():
        output[column] = values

    probability_cols = list(probability_columns)
    probability_frame = output[probability_cols]
    high_probability = probability_frame.ge(args.probability_threshold)
    output["HighProbabilityModelCount"] = high_probability.sum(axis=1)
    output["ModelVoteCountAt50"] = probability_frame.ge(0.50).sum(axis=1)
    output["AverageModelProbability"] = probability_frame.mean(axis=1)
    output["MaxModelProbability"] = probability_frame.max(axis=1)
    output["MinModelProbability"] = probability_frame.min(axis=1)
    output["ProbabilitySpread"] = output["MaxModelProbability"] - output["MinModelProbability"]

    entity_cols = [
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
        "TicketFateSignal",
        "FamilyFateSignal",
        "PrimaryFateScope",
        "PrimaryFateSignal",
        "PrimaryFateSupportBin",
        "FamilyTicketFateSignal",
        "PrimaryCompanionScope",
        "PrimaryGroupChildFemalePattern",
        "Cabin",
        "Deck",
    ]
    available_entity_cols = [col for col in entity_cols if col in rule_features.columns]
    output = output.merge(
        rule_features[["PassengerId", *[col for col in available_entity_cols if col != "PassengerId"]]],
        on="PassengerId",
        how="left",
        suffixes=("", "_Feature"),
    )

    candidates = output[
        output["CurrentBestPrediction"].eq(0)
        & output["HighProbabilityModelCount"].ge(args.min_models)
    ].copy()
    candidates = candidates.sort_values(
        ["HighProbabilityModelCount", "AverageModelProbability", "ModelVoteCountAt50"],
        ascending=[False, False, False],
    )

    report_cols = [
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
        "CurrentBestPrediction",
        "HighProbabilityModelCount",
        "ModelVoteCountAt50",
        "AverageModelProbability",
        "MinModelProbability",
        "MaxModelProbability",
        "ProbabilitySpread",
        *probability_cols,
    ]
    report_cols = [col for col in report_cols if col in candidates.columns]
    candidates[report_cols].to_csv(
        args.output_dir / "model_rescue_candidates.csv",
        index=False,
    )

    fate_cols = [
        col
        for col in [
            "TicketFateSignal",
            "FamilyFateSignal",
            "FamilyTicketFateSignal",
            "PrimaryFateSignal",
        ]
        if col in candidates.columns
    ]
    if fate_cols:
        all_died_mask = candidates[fate_cols].eq("all_died").any(axis=1)
    else:
        all_died_mask = pd.Series(False, index=candidates.index)

    candidates.loc[all_died_mask, report_cols].to_csv(
        args.output_dir / "model_rescue_all_died_conflicts.csv",
        index=False,
    )
    candidates.loc[~all_died_mask, report_cols].to_csv(
        args.output_dir / "model_rescue_no_all_died_candidates.csv",
        index=False,
    )

    all_scores = output.sort_values(
        ["CurrentBestPrediction", "AverageModelProbability"],
        ascending=[True, False],
    )
    all_scores_cols = [col for col in report_cols if col in all_scores.columns]
    all_scores[all_scores_cols].to_csv(
        args.output_dir / "model_rescue_all_scores.csv",
        index=False,
    )

    print(f"Strategy: {args.strategy}")
    print(f"Feature set: {args.feature_set}")
    print(f"Probability threshold: {args.probability_threshold}")
    print(f"Models used: {', '.join(col.replace('_probability', '') for col in probability_cols)}")
    print(f"Candidates found: {len(candidates)}")
    print(f"Candidates saved to: {args.output_dir / 'model_rescue_candidates.csv'}")
    print(
        "All-died conflict candidates saved to: "
        f"{args.output_dir / 'model_rescue_all_died_conflicts.csv'}"
    )
    print(
        "No-all-died candidates saved to: "
        f"{args.output_dir / 'model_rescue_no_all_died_candidates.csv'}"
    )
    print(f"All scores saved to: {args.output_dir / 'model_rescue_all_scores.csv'}")


if __name__ == "__main__":
    main()
