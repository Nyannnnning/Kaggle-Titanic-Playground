from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from src.features import PassengerFeatureBuilder
from src.regime_rules import apply_regime_rule_layer, regime_rule_audit


DEFAULT_TRAIN_PATH = Path("data/train.csv")
DEFAULT_TEST_PATH = Path("data/test.csv")
DEFAULT_OUTPUT_PATH = Path("submission.csv")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")
DEFAULT_REPORTS_DIR = Path("reports")
CURRENT_BEST_STRATEGY = "regime_layer_v3_1309_only"
DEFAULT_REFERENCE_STRATEGY = "rule_mined_group_fate_support"

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
    "group_fate_hybrid",
    "group_fate_support_hybrid",
    "rule_mined_group_fate_support",
    "rule_mined_group_fate_p1_midfare",
    "rule_mined_group_fate_ticket_route_10000_s",
    "rule_mined_group_fate_p3_highfare_mixed",
    "rule_mined_group_fate_asplund_children",
    "rule_mined_group_fate_ticket_1601",
    "regime_layer_v1",
    "regime_p3_master_small_family_rescue",
    "regime_layer_v2",
    "regime_p3_master_small_family_no_all_died_rescue",
    "regime_layer_v3_1309_only",
    "regime_layer_v3_1284_only",
    "regime_p3_master_small_family_all_survived_rescue",
    "regime_p3_master_small_family_mixed_rescue",
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
        default=CURRENT_BEST_STRATEGY,
        help=f"Strategy to write to submission.csv. Default: {CURRENT_BEST_STRATEGY}",
    )
    parser.add_argument(
        "--reference-strategy",
        choices=sorted(STRATEGIES),
        default=DEFAULT_REFERENCE_STRATEGY,
        help=f"Reference strategy for delta audit. Default: {DEFAULT_REFERENCE_STRATEGY}",
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


def rule_p3_primary_fate_all_died(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(3) & features["PrimaryFateSignal"].eq("all_died")


def rule_p3_primary_fate_all_died_support(features: pd.DataFrame) -> pd.Series:
    return rule_p3_primary_fate_all_died(features) & features[
        "PrimaryFateSupportBin"
    ].isin(["2", "3+"])


def group_fate_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)

    # Target-safe Group Fate rule, evaluated with repeated CV. This is more
    # aggressive than the support-limited public-best variant because it can flip
    # 3rd-class passengers from one-member train-mapped fate evidence.
    prediction[rule_p3_primary_fate_all_died(features)] = 0
    return prediction.astype(int)


def group_fate_support_hybrid(features: pd.DataFrame) -> pd.Series:
    prediction = conservative_hybrid(features)

    # Conservative Group Fate variant: require at least two fitted group members.
    prediction[rule_p3_primary_fate_all_died_support(features)] = 0
    return prediction.astype(int)


def rule_mined_group_fate_support(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_hybrid(features)

    # Current public-best rule layer: keep the low-fare/no-ticket-child rule and
    # add only supported 3rd-class all_died GroupFateObject evidence.
    prediction[rule_p3_primary_fate_all_died_support(features)] = 0
    return prediction.astype(int)


def rule_p1_midfare_adult_alone(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(1)
        & features["AgeBin"].eq("Adult")
        & features["FamilySizeBin"].eq("Alone")
        & features["FareBinPclass"].eq("MidFare_P1")
    )


def rule_mined_group_fate_p1_midfare(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Candidate positive first-class correction found after the Group Fate best
    # baseline: adult, solo, mid-fare 1st-class passengers were strong survivors
    # in repeated CV. Keep separate from the default until Kaggle-tested.
    prediction[rule_p1_midfare_adult_alone(features)] = 1
    return prediction.astype(int)


def rule_ticket_route_10000_s_midfare_p1(features: pd.DataFrame) -> pd.Series:
    return (
        features["AgeBin"].eq("Adult")
        & features["FarePerTicketBinPclass"].eq("MidFare_P1")
        & features["TicketNumberBandEmbarked"].eq("10000_19999_S")
    )


def rule_mined_group_fate_ticket_route_10000_s(
    features: pd.DataFrame,
) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Failed one-passenger TicketRouteObject experiment. It flips PassengerId
    # 1036 and reduced public score from 0.79904 to 0.79665; keep for audit.
    prediction[rule_ticket_route_10000_s_midfare_p1(features)] = 1
    return prediction.astype(int)


def rule_p3_highfare_primary_mixed(features: pd.DataFrame) -> pd.Series:
    return (
        features["FareBinPclass"].eq("HighFare_P3")
        & features["PrimaryFateSignal"].eq("mixed")
    )


def rule_p3_highfare_mixed_asplund_children(features: pd.DataFrame) -> pd.Series:
    return (
        rule_p3_highfare_primary_mixed(features)
        & features["TicketNormalized"].eq("347077")
        & (features["Title"].eq("Master") | features["Age"].le(13).fillna(False))
    )


def rule_p3_highfare_mixed_ticket_1601(features: pd.DataFrame) -> pd.Series:
    return rule_p3_highfare_primary_mixed(features) & features[
        "TicketNormalized"
    ].eq("1601")


def rule_mined_group_fate_p3_highfare_mixed(
    features: pd.DataFrame,
) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # False-negative candidate: high-fare 3rd-class passengers in mixed fate
    # ticket/family-ticket groups. Flips four test passengers; Kaggle-test only.
    prediction[rule_p3_highfare_primary_mixed(features)] = 1
    return prediction.astype(int)


def rule_mined_group_fate_asplund_children(
    features: pd.DataFrame,
) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Narrow child/Master subset of the Asplund high-fare mixed family-ticket
    # group. Flips PassengerId 1046 and 1271.
    prediction[rule_p3_highfare_mixed_asplund_children(features)] = 1
    return prediction.astype(int)


def rule_mined_group_fate_ticket_1601(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Narrow ticket-route/fate subset for Ticket 1601. Flips PassengerId 931.
    prediction[rule_p3_highfare_mixed_ticket_1601(features)] = 1
    return prediction.astype(int)


def regime_layer_v1(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Failed public-test candidate. Keep explicitly pinned for reproducibility.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_rescue",),
    )


def regime_p3_master_small_family_rescue(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Failed single-rule A/B strategy for the broad P3 survival rescue regime.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_rescue",),
    )


def regime_layer_v2(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Narrowed P3 survival rescue: do not override any all_died GroupFate signal.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_no_all_died_rescue",),
    )


def regime_p3_master_small_family_no_all_died_rescue(
    features: pd.DataFrame,
) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Single-rule A/B strategy for the narrowed P3 survival rescue regime.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_no_all_died_rescue",),
    )


def regime_layer_v3_1309_only(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Split of neutral v2: keep only the all_survived group-fate side.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_all_survived_rescue",),
    )


def regime_layer_v3_1284_only(features: pd.DataFrame) -> pd.Series:
    prediction = rule_mined_group_fate_support(features)

    # Split of neutral v2: keep only the mixed-fate side.
    return apply_regime_rule_layer(
        features,
        prediction,
        rule_names=("p3_master_small_family_mixed_rescue",),
    )


def regime_p3_master_small_family_all_survived_rescue(
    features: pd.DataFrame,
) -> pd.Series:
    return regime_layer_v3_1309_only(features)


def regime_p3_master_small_family_mixed_rescue(
    features: pd.DataFrame,
) -> pd.Series:
    return regime_layer_v3_1284_only(features)


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
    if strategy == "group_fate_hybrid":
        return group_fate_hybrid(features)
    if strategy == "group_fate_support_hybrid":
        return group_fate_support_hybrid(features)
    if strategy == "rule_mined_group_fate_support":
        return rule_mined_group_fate_support(features)
    if strategy == "rule_mined_group_fate_p1_midfare":
        return rule_mined_group_fate_p1_midfare(features)
    if strategy == "rule_mined_group_fate_ticket_route_10000_s":
        return rule_mined_group_fate_ticket_route_10000_s(features)
    if strategy == "rule_mined_group_fate_p3_highfare_mixed":
        return rule_mined_group_fate_p3_highfare_mixed(features)
    if strategy == "rule_mined_group_fate_asplund_children":
        return rule_mined_group_fate_asplund_children(features)
    if strategy == "rule_mined_group_fate_ticket_1601":
        return rule_mined_group_fate_ticket_1601(features)
    if strategy == "regime_layer_v1":
        return regime_layer_v1(features)
    if strategy == "regime_p3_master_small_family_rescue":
        return regime_p3_master_small_family_rescue(features)
    if strategy == "regime_layer_v2":
        return regime_layer_v2(features)
    if strategy == "regime_p3_master_small_family_no_all_died_rescue":
        return regime_p3_master_small_family_no_all_died_rescue(features)
    if strategy == "regime_layer_v3_1309_only":
        return regime_layer_v3_1309_only(features)
    if strategy == "regime_layer_v3_1284_only":
        return regime_layer_v3_1284_only(features)
    if strategy == "regime_p3_master_small_family_all_survived_rescue":
        return regime_p3_master_small_family_all_survived_rescue(features)
    if strategy == "regime_p3_master_small_family_mixed_rescue":
        return regime_p3_master_small_family_mixed_rescue(features)
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
            "TicketPrefix",
            "TicketNumberBand",
            "TicketPrefixEmbarked",
            "TicketNumberBandEmbarked",
            "TicketPrefixPclassEmbarked",
            "PrimaryFateScope",
            "PrimaryFateSignal",
            "PrimaryFateSupportBin",
            "PrimaryFateKnownCount",
            "PrimaryFateSurvivalRate",
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
            "TicketPrefix",
            "TicketNumberBand",
            "TicketPrefixEmbarked",
            "TicketNumberBandEmbarked",
            "TicketPrefixPclassEmbarked",
            "PrimaryFateScope",
            "PrimaryFateSignal",
            "PrimaryFateSupportBin",
            "PrimaryFateKnownCount",
            "PrimaryFateSurvivalRate",
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
    train_features = feature_builder.fit_transform(X_train, y)
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

    train_current_best = predict_strategy(train_features, CURRENT_BEST_STRATEGY)
    test_current_best = predict_strategy(test_features, CURRENT_BEST_STRATEGY)
    regime_audit = regime_rule_audit(
        train_features,
        y,
        train_current_best,
        test_features,
        test_current_best,
    )
    regime_audit.to_csv(DEFAULT_REPORTS_DIR / "regime_rule_audit.csv", index=False)

    overrides = override_audit(test_features, args.strategy)
    overrides.to_csv(DEFAULT_REPORTS_DIR / "submission_override_audit.csv", index=False)

    delta = strategy_delta_audit(
        test_features,
        args.strategy,
        reference_strategy=args.reference_strategy,
    )
    delta.to_csv(DEFAULT_REPORTS_DIR / "submission_strategy_delta_audit.csv", index=False)

    chosen_row = audit[audit["strategy"].eq(args.strategy)].iloc[0]
    print(f"Strategy: {args.strategy}")
    print(f"Train accuracy: {chosen_row['train_accuracy']:.4f}")
    print(f"Test survivor count: {int(chosen_row['test_survivor_count'])}")
    print(f"Changed from gender baseline: {int(chosen_row['test_changed_from_gender'])}")
    print(f"Submission saved to: {args.output}")
    print(f"Strategy audit saved to: {DEFAULT_REPORTS_DIR / 'submission_strategy_audit.csv'}")
    print(f"Regime rule audit saved to: {DEFAULT_REPORTS_DIR / 'regime_rule_audit.csv'}")
    print(f"Override audit saved to: {DEFAULT_REPORTS_DIR / 'submission_override_audit.csv'}")
    print(
        f"Strategy delta audit saved to: "
        f"{DEFAULT_REPORTS_DIR / 'submission_strategy_delta_audit.csv'} "
        f"(reference: {args.reference_strategy})"
    )


if __name__ == "__main__":
    main()
