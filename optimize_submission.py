from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from src.features import PassengerFeatureBuilder


DEFAULT_TRAIN_PATH = Path("data/train.csv")
DEFAULT_TEST_PATH = Path("data/test.csv")
DEFAULT_OUTPUT_PATH = Path("submission.csv")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")
DEFAULT_REPORTS_DIR = Path("reports")

STRATEGIES = {
    "gender_baseline",
    "conservative_hybrid",
    "child_count_hybrid",
    "optimized_hybrid",
    "rule_mined_hybrid",
    "rule_mined_known_age",
    "rule_mined_female_no_child",
    "rule_mined_solo_lowfare",
    "rule_mined_alone_lowfare",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create conservative Titanic submissions from audited rules."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default="conservative_hybrid",
        help="Strategy to write to submission.csv. Default: conservative_hybrid",
    )
    return parser.parse_args()


def gender_baseline(features: pd.DataFrame) -> pd.Series:
    return features["Sex"].eq("female").astype(int)


def conservative_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = gender_baseline(features)

    # Stable train signal: large 3rd-class female families had very low survival.
    prediction[
        features["Sex"].eq("female")
        & features["Pclass"].eq(3)
        & features["FamilySize"].ge(5)
    ] = 0

    # Stable train signal: male Masters in 1st/2nd class survived in the sample.
    prediction[
        features["Sex"].eq("male")
        & features["Title"].eq("Master")
        & features["Pclass"].le(2)
    ] = 1

    # Conservative child override: young 3rd-class Masters in small families.
    prediction[
        features["Sex"].eq("male")
        & features["Title"].eq("Master")
        & features["Pclass"].eq(3)
        & features["Age"].le(8)
        & features["FamilySize"].le(4)
    ] = 1

    return prediction.astype(int)


def child_count_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)

    # Pclass 3 male Masters in small families have a clear train-set survival
    # window up to age 12. Larger child-heavy families remain negative.
    prediction[
        features["Sex"].eq("male")
        & features["Title"].eq("Master")
        & features["Pclass"].eq(3)
        & features["Age"].le(12)
        & features["FamilySize"].le(4)
    ] = 1

    return prediction.astype(int)


def optimized_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)

    # Higher train accuracy, but more public-leaderboard risk than conservative_hybrid.
    prediction[
        features["Sex"].eq("female")
        & features["Pclass"].eq(3)
        & features["AgeBin"].isin(["Adult", "MiddleAge"])
        & features["FareBin"].isin(["VeryLowFare", "LowFare"])
    ] = 0

    return prediction.astype(int)


def rule_mined_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)

    # Rule-mined candidate, evaluated with repeated CV:
    # 3rd-class low-fare passengers without ticket-group children had weak
    # survival outcomes among conservative-hybrid positives.
    prediction[
        rule_p3_lowfare_no_ticket_child(features)
    ] = 0

    return prediction.astype(int)


def rule_p3_lowfare_no_ticket_child(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(3)
        & features["FareBin"].eq("LowFare")
        & features["TicketChildCount"].eq(0)
        & ~features["Age"].le(12).fillna(False)
    )


def rule_mined_known_age(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)
    prediction[
        features["Pclass"].eq(3)
        & features["FareBin"].eq("LowFare")
        & features["TicketChildCount"].eq(0)
        & features["Age"].gt(12).fillna(False)
    ] = 0
    return prediction.astype(int)


def rule_mined_female_no_child(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)
    prediction[
        features["Pclass"].eq(3)
        & features["FareBin"].eq("LowFare")
        & features["TicketChildFemalePattern"].eq("female_no_child")
        & ~features["Age"].le(12).fillna(False)
    ] = 0
    return prediction.astype(int)


def rule_mined_solo_lowfare(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)
    prediction[
        rule_p3_lowfare_no_ticket_child(features)
        & features["TicketGroupSize"].eq(1)
    ] = 0
    return prediction.astype(int)


def rule_mined_alone_lowfare(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)
    prediction[
        rule_p3_lowfare_no_ticket_child(features)
        & features["FamilySize"].eq(1)
    ] = 0
    return prediction.astype(int)


def predict_strategy(features: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "gender_baseline":
        return gender_baseline(features)
    if strategy == "conservative_hybrid":
        return conservative_hybrid(features)
    if strategy == "child_count_hybrid":
        return child_count_hybrid(features)
    if strategy == "optimized_hybrid":
        return optimized_hybrid(features)
    if strategy == "rule_mined_hybrid":
        return rule_mined_hybrid(features)
    if strategy == "rule_mined_known_age":
        return rule_mined_known_age(features)
    if strategy == "rule_mined_female_no_child":
        return rule_mined_female_no_child(features)
    if strategy == "rule_mined_solo_lowfare":
        return rule_mined_solo_lowfare(features)
    if strategy == "rule_mined_alone_lowfare":
        return rule_mined_alone_lowfare(features)
    raise ValueError(f"Unsupported strategy: {strategy}")


def submission_frame(features: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": features["PassengerId"].astype(int),
            "Survived": prediction.astype(int),
        }
    )


def strategy_audit(
    train_features: pd.DataFrame,
    y: pd.Series,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    train_baseline = gender_baseline(train_features)
    test_baseline = gender_baseline(test_features)

    for strategy in sorted(STRATEGIES):
        train_pred = predict_strategy(train_features, strategy)
        test_pred = predict_strategy(test_features, strategy)
        rows.append(
            {
                "strategy": strategy,
                "train_accuracy": accuracy_score(y, train_pred),
                "train_survivor_count": int(train_pred.sum()),
                "train_changed_from_gender": int(train_pred.ne(train_baseline).sum()),
                "test_survivor_count": int(test_pred.sum()),
                "test_changed_from_gender": int(test_pred.ne(test_baseline).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("strategy")


def override_audit(
    test_features: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    baseline = gender_baseline(test_features)
    prediction = predict_strategy(test_features, strategy)
    changed = test_features.loc[
        prediction.ne(baseline),
        [
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
            "TicketGroupSize",
            "TicketChildCount",
            "TicketChildFemalePattern",
            "PrimaryCompanionScope",
            "PrimaryGroupChildFemalePattern",
            "FamilyGroupSize",
            "Deck",
            "SpatialAccessTier",
        ],
    ].copy()
    changed["Baseline"] = baseline[prediction.ne(baseline)].to_numpy()
    changed["Prediction"] = prediction[prediction.ne(baseline)].to_numpy()
    changed["Strategy"] = strategy
    return changed


def strategy_delta_audit(
    test_features: pd.DataFrame,
    strategy: str,
    reference_strategy: str = "conservative_hybrid",
) -> pd.DataFrame:
    reference = predict_strategy(test_features, reference_strategy)
    prediction = predict_strategy(test_features, strategy)
    changed_mask = prediction.ne(reference)
    changed = test_features.loc[
        changed_mask,
        [
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
            "TicketGroupSize",
            "TicketChildCount",
            "TicketChildFemalePattern",
            "PrimaryCompanionScope",
            "PrimaryGroupChildFemalePattern",
            "FamilyGroupSize",
            "Deck",
            "SpatialAccessTier",
        ],
    ].copy()
    changed["ReferenceStrategy"] = reference_strategy
    changed["ReferencePrediction"] = reference[changed_mask].to_numpy()
    changed["Strategy"] = strategy
    changed["Prediction"] = prediction[changed_mask].to_numpy()
    return changed


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    if "Survived" not in train_df.columns:
        raise ValueError(f"{args.train} must contain Survived.")

    X_train = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    feature_builder = PassengerFeatureBuilder(feature_set="clean")
    feature_builder.fit(X_train)
    train_features = feature_builder.transform(X_train)
    test_features = feature_builder.transform(test_df)

    DEFAULT_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for strategy in sorted(STRATEGIES):
        prediction = predict_strategy(test_features, strategy)
        output = DEFAULT_SUBMISSIONS_DIR / f"{strategy}.csv"
        submission_frame(test_features, prediction).to_csv(output, index=False)

    chosen_prediction = predict_strategy(test_features, args.strategy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission_frame(test_features, chosen_prediction).to_csv(args.output, index=False)

    audit = strategy_audit(train_features, y, test_features)
    audit.to_csv(DEFAULT_REPORTS_DIR / "submission_strategy_audit.csv", index=False)

    overrides = override_audit(test_features, args.strategy)
    overrides.to_csv(DEFAULT_REPORTS_DIR / "submission_override_audit.csv", index=False)

    delta = strategy_delta_audit(test_features, args.strategy)
    delta.to_csv(DEFAULT_REPORTS_DIR / "submission_strategy_delta_audit.csv", index=False)

    chosen_row = audit[audit["strategy"].eq(args.strategy)].iloc[0]
    print(f"Strategy: {args.strategy}")
    print(f"Train accuracy: {chosen_row['train_accuracy']:.4f}")
    print(f"Test survivor count: {int(chosen_row['test_survivor_count'])}")
    print(f"Changed from gender baseline: {int(chosen_row['test_changed_from_gender'])}")
    print(f"Submission saved to: {args.output}")
    print(f"Strategy audit saved to: {DEFAULT_REPORTS_DIR / 'submission_strategy_audit.csv'}")
    print(f"Override audit saved to: {DEFAULT_REPORTS_DIR / 'submission_override_audit.csv'}")
    print(
        f"Strategy delta audit saved to: "
        f"{DEFAULT_REPORTS_DIR / 'submission_strategy_delta_audit.csv'}"
    )


if __name__ == "__main__":
    main()
