from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from analyze_rule_candidates import add_rule_columns
from optimize_submission import CURRENT_BEST_STRATEGY, STRATEGIES, predict_strategy
from src.child_groups import add_child_group_features
from src.features import PassengerFeatureBuilder


TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
REPORTS_DIR = Path("reports")

PROFILE_CATEGORICAL_FEATURES = [
    "Sex",
    "Title",
    "AgeBin",
    "AgeRelativeBucketWithinPclass",
    "AgeMissingPatternByPclassTitle",
    "Embarked",
    "FareBin",
    "FarePerTicketBin",
    "FareBinPclass",
    "FarePerTicketBinPclass",
    "FamilySizeBin",
    "IsAloneRule",
    "TicketGroupSizeBin",
    "FamilyTicketGroupSizeBin",
    "TravelPartyType",
    "FareObjectInterpretation",
    "TicketPrefix",
    "TicketNumberBand",
    "TicketPrefixEmbarked",
    "TicketNumberBandEmbarked",
    "TicketCompositionType",
    "TicketFamilyPattern",
    "TicketChildFemalePattern",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "ChildFemaleCompanionPattern",
    "CompanionGroupSizeBin",
    "CompanionChildCountBin",
    "CompanionFemaleCountBin",
    "CompanionAdultMaleCountBin",
    "CompanionAccompanyingChildCountBin",
    "CompanionAccompanyingFemaleCountBin",
    "CompanionAccompanyingAdultMaleCountBin",
    "TicketFateSignal",
    "TicketFateSupportBin",
    "FamilyTicketFateSignal",
    "FamilyTicketFateSupportBin",
    "PrimaryFateScope",
    "PrimaryFateSignal",
    "PrimaryFateSupportBin",
    "CabinKnown",
    "Deck",
    "SpatialAccessTier",
    "ClassDeckConsistency",
    "FormalNamePattern",
    "FormalTitle",
]

PROFILE_NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FarePerFamilyMember",
    "FarePerTicketMember",
    "FarePerFamilyTicketMember",
    "FamilySize",
    "TicketGroupSize",
    "FamilyTicketGroupSize",
    "CompanionGroupSize",
    "CompanionChildCount",
    "CompanionFemaleCount",
    "CompanionAdultMaleCount",
    "TicketChildCount",
    "TicketFemaleCount",
    "TicketAdultMaleCount",
    "PrimaryFateKnownCount",
    "PrimaryFateSurvivalRate",
    "DeckOrdinal",
]

EXAMPLE_COLUMNS = [
    "PassengerId",
    "Survived",
    "Predicted",
    "ErrorType",
    "Name",
    "Sex",
    "Pclass",
    "Title",
    "Age",
    "AgeBin",
    "Embarked",
    "Fare",
    "FareBin",
    "FarePerTicketMember",
    "FarePerTicketBin",
    "FamilySize",
    "FamilySizeBin",
    "Ticket",
    "TicketNormalized",
    "TicketGroupSize",
    "TicketChildFemalePattern",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "PrimaryFateScope",
    "PrimaryFateSignal",
    "PrimaryFateSupportBin",
    "Cabin",
    "CabinKnown",
    "Deck",
    "SpatialAccessTier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze exception populations: low-class survivors and "
            "high-class deaths."
        )
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default=CURRENT_BEST_STRATEGY,
        help=f"Current rule baseline used for error slices. Default: {CURRENT_BEST_STRATEGY}",
    )
    parser.add_argument("--min-p3-count", type=int, default=5)
    parser.add_argument("--min-p1-count", type=int, default=3)
    return parser.parse_args()


def safe_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def add_prediction_columns(features: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    out = features.copy()
    out["Predicted"] = prediction.astype(int).to_numpy()
    if "Survived" not in out.columns:
        out["Correct"] = np.nan
        out["ErrorType"] = "unknown"
        return out

    out["Correct"] = out["Survived"].eq(out["Predicted"])
    out["ErrorType"] = np.select(
        [
            out["Survived"].eq(1) & out["Predicted"].eq(1),
            out["Survived"].eq(0) & out["Predicted"].eq(0),
            out["Survived"].eq(0) & out["Predicted"].eq(1),
            out["Survived"].eq(1) & out["Predicted"].eq(0),
        ],
        [
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ],
        default="unknown",
    )
    return out


def categorical_lift(
    features: pd.DataFrame,
    target_mask: pd.Series,
    contrast_mask: pd.Series,
    scope: str,
    min_target_count: int = 2,
) -> pd.DataFrame:
    target = features.loc[target_mask]
    contrast = features.loc[contrast_mask]
    rows = []
    target_n = len(target)
    contrast_n = len(contrast)

    for feature in safe_columns(features, PROFILE_CATEGORICAL_FEATURES):
        target_counts = target[feature].fillna("Unknown").astype(str).value_counts()
        contrast_counts = contrast[feature].fillna("Unknown").astype(str).value_counts()
        values = sorted(set(target_counts.index) | set(contrast_counts.index))
        for value in values:
            target_count = int(target_counts.get(value, 0))
            contrast_count = int(contrast_counts.get(value, 0))
            if target_count < min_target_count and contrast_count < min_target_count:
                continue
            target_share = target_count / target_n if target_n else 0.0
            contrast_share = contrast_count / contrast_n if contrast_n else 0.0
            rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "value": value,
                    "target_count": target_count,
                    "contrast_count": contrast_count,
                    "target_share": target_share,
                    "contrast_share": contrast_share,
                    "share_delta": target_share - contrast_share,
                    "lift": target_share / contrast_share
                    if contrast_share > 0
                    else np.inf,
                    "direction": "enriched_in_target"
                    if target_share > contrast_share
                    else "depleted_in_target",
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["scope", "share_delta", "target_count"],
        ascending=[True, False, False],
    )


def numeric_profile(
    features: pd.DataFrame,
    target_mask: pd.Series,
    contrast_mask: pd.Series,
    scope: str,
) -> pd.DataFrame:
    rows = []
    target = features.loc[target_mask]
    contrast = features.loc[contrast_mask]

    for feature in safe_columns(features, PROFILE_NUMERIC_FEATURES):
        target_values = pd.to_numeric(target[feature], errors="coerce")
        contrast_values = pd.to_numeric(contrast[feature], errors="coerce")
        rows.append(
            {
                "scope": scope,
                "feature": feature,
                "target_mean": target_values.mean(),
                "contrast_mean": contrast_values.mean(),
                "mean_delta": target_values.mean() - contrast_values.mean(),
                "target_median": target_values.median(),
                "contrast_median": contrast_values.median(),
                "median_delta": target_values.median() - contrast_values.median(),
                "target_known_count": int(target_values.notna().sum()),
                "contrast_known_count": int(contrast_values.notna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["scope", "mean_delta"],
        ascending=[True, False],
    )


def condition_label(parts: list[tuple[str, object]]) -> str:
    return " AND ".join(f"{feature} == {value}" for feature, value in parts)


def rule_rows_for_group(
    frame: pd.DataFrame,
    feature_sets: list[tuple[str, ...]],
    scope: str,
    target: str,
    min_count: int,
) -> list[dict[str, object]]:
    rows = []
    baseline_survival_rate = frame["Survived"].mean()
    baseline_target_rate = (
        baseline_survival_rate if target == "survival" else 1 - baseline_survival_rate
    )

    for feature_set in feature_sets:
        group_cols = list(feature_set)
        grouped = frame.groupby(group_cols, dropna=False)
        for values, group in grouped:
            if len(group_cols) == 1 and not isinstance(values, tuple):
                values = (values,)
            count = len(group)
            if count < min_count:
                continue
            survived = int(group["Survived"].sum())
            died = count - survived
            survival_rate = survived / count
            death_rate = died / count
            target_rate = survival_rate if target == "survival" else death_rate
            target_hits = survived if target == "survival" else died
            if target_hits < 2:
                continue

            parts = list(zip(group_cols, values))
            rows.append(
                {
                    "scope": scope,
                    "target": target,
                    "condition": condition_label(parts),
                    "count": count,
                    "survived": survived,
                    "died": died,
                    "survival_rate": survival_rate,
                    "death_rate": death_rate,
                    "target_rate": target_rate,
                    "class_baseline_target_rate": baseline_target_rate,
                    "target_rate_delta": target_rate - baseline_target_rate,
                    "feature_count": len(group_cols),
                }
            )
    return rows


def candidate_rules(
    features: pd.DataFrame,
    min_p3_count: int,
    min_p1_count: int,
) -> pd.DataFrame:
    profile_features = safe_columns(features, PROFILE_CATEGORICAL_FEATURES)
    one_feature_sets = [(feature,) for feature in profile_features]
    pair_feature_sets = list(combinations(profile_features, 2))

    rows = []
    p3 = features[features["Pclass"].eq(3)]
    p1 = features[features["Pclass"].eq(1)]
    rows.extend(
        rule_rows_for_group(
            p3,
            one_feature_sets + pair_feature_sets,
            scope="pclass3_survival",
            target="survival",
            min_count=min_p3_count,
        )
    )
    rows.extend(
        rule_rows_for_group(
            p1,
            one_feature_sets + pair_feature_sets,
            scope="pclass1_death",
            target="death",
            min_count=min_p1_count,
        )
    )

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)
    candidates = candidates[candidates["target_rate_delta"].gt(0)].copy()
    return candidates.sort_values(
        ["scope", "target_rate", "target_rate_delta", "count"],
        ascending=[True, False, False, False],
    )


def write_markdown_summary(
    output_path: Path,
    features: pd.DataFrame,
    strategy: str,
    strategy_accuracy: float,
    lift: pd.DataFrame,
    rules: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    p3 = features[features["Pclass"].eq(3)]
    p1 = features[features["Pclass"].eq(1)]
    p3_fn = errors[errors["ExceptionSlice"].eq("pclass3_survivor_false_negative")]
    p1_fp = errors[errors["ExceptionSlice"].eq("pclass1_death_false_positive")]

    parts = ["# Titanic Exception Profiles\n"]
    parts.append(
        "This report compares exception populations inside the same Pclass, "
        "then lists rule candidates for manual audit. It is an exploration report, "
        "not a new submission rule layer.\n"
    )
    parts.append("## Baselines\n")
    parts.append(
        "\n".join(
            [
                f"- Strategy audited on train: `{strategy}`",
                f"- Train accuracy: `{strategy_accuracy:.4f}`",
                f"- Pclass 3 survival rate: `{p3['Survived'].mean():.4f}` ({int(p3['Survived'].sum())}/{len(p3)})",
                f"- Pclass 1 death rate: `{1 - p1['Survived'].mean():.4f}` ({int((1 - p1['Survived']).sum())}/{len(p1)})",
                f"- Pclass 3 survivors currently predicted dead: `{len(p3_fn)}`",
                f"- Pclass 1 deaths currently predicted alive: `{len(p1_fp)}`",
            ]
        )
    )

    def table(title: str, frame: pd.DataFrame, columns: list[str], n: int = 12) -> str:
        if frame.empty:
            body = "(no rows)"
        else:
            visible = frame[columns].head(n).copy()
            for col in visible.select_dtypes(include=[float]).columns:
                visible[col] = visible[col].round(4)
            body = visible.to_string(index=False)
        return f"\n\n## {title}\n\n```text\n{body}\n```\n"

    lift_cols = [
        "feature",
        "value",
        "target_count",
        "contrast_count",
        "target_share",
        "contrast_share",
        "share_delta",
    ]
    rules_cols = [
        "condition",
        "count",
        "survived",
        "died",
        "survival_rate",
        "death_rate",
        "target_rate_delta",
    ]
    error_cols = safe_columns(errors, EXAMPLE_COLUMNS + ["ExceptionSlice"])

    parts.append(
        table(
            "Pclass 3 Survivor Enriched Features",
            lift[
                lift["scope"].eq("pclass3_survivors_vs_pclass3_deaths")
                & lift["direction"].eq("enriched_in_target")
            ],
            lift_cols,
        )
    )
    parts.append(
        table(
            "Pclass 1 Death Enriched Features",
            lift[
                lift["scope"].eq("pclass1_deaths_vs_pclass1_survivors")
                & lift["direction"].eq("enriched_in_target")
            ],
            lift_cols,
        )
    )
    parts.append(
        table(
            "Pclass 3 Survival Rule Candidates",
            rules[rules["scope"].eq("pclass3_survival")],
            rules_cols,
        )
    )
    parts.append(
        table(
            "Pclass 1 Death Rule Candidates",
            rules[rules["scope"].eq("pclass1_death")],
            rules_cols,
        )
    )
    parts.append(
        table(
            "Current Best Exception Errors",
            errors,
            error_cols,
            n=40,
        )
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def _append_reason(
    reason_lists: list[list[str]],
    mask: pd.Series,
    reason: str,
) -> None:
    for idx in np.flatnonzero(mask.to_numpy()):
        reason_lists[idx].append(reason)


def add_watchlist_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    p3_reasons: list[list[str]] = [[] for _ in range(len(out))]
    p1_reasons: list[list[str]] = [[] for _ in range(len(out))]

    p3 = out["Pclass"].eq(3)
    _append_reason(
        p3_reasons,
        p3 & out["Sex"].eq("female"),
        "p3_female",
    )
    _append_reason(
        p3_reasons,
        p3 & out["Title"].eq("Master") & out["FamilySizeBin"].eq("SmallFamily"),
        "p3_master_small_family",
    )
    _append_reason(
        p3_reasons,
        p3 & out["AgeBin"].eq("Infant"),
        "p3_infant",
    )
    _append_reason(
        p3_reasons,
        p3
        & (
            out["TicketFateSignal"].eq("all_survived")
            | out["FamilyTicketFateSignal"].eq("all_survived")
            | out["PrimaryFateSignal"].eq("all_survived")
        ),
        "p3_group_fate_all_survived",
    )
    _append_reason(
        p3_reasons,
        p3
        & out["TicketChildFemalePattern"].isin(
            [
                "protected_group_with_child",
                "child_female_no_adult_male",
                "child_only_or_child_male",
            ]
        ),
        "p3_child_or_female_protected_group",
    )
    _append_reason(
        p3_reasons,
        p3 & out["Embarked"].isin(["C", "Q"]),
        "p3_embarked_c_or_q",
    )
    _append_reason(
        p3_reasons,
        p3 & out["FareBin"].isin(["MidFare", "HighFare", "LuxuryFare"]),
        "p3_mid_or_high_raw_fare",
    )

    p1 = out["Pclass"].eq(1)
    _append_reason(
        p1_reasons,
        p1 & out["Sex"].eq("male") & out["Title"].eq("Mr"),
        "p1_male_mr",
    )
    _append_reason(
        p1_reasons,
        p1 & out["TicketChildFemalePattern"].eq("adult_male_only"),
        "p1_adult_male_only_ticket_object",
    )
    _append_reason(
        p1_reasons,
        p1 & out["PrimaryCompanionScope"].eq("solo"),
        "p1_solo_primary_companion",
    )
    _append_reason(
        p1_reasons,
        p1 & out["CompanionFemaleCountBin"].eq("0"),
        "p1_no_female_companion",
    )
    _append_reason(
        p1_reasons,
        p1 & pd.to_numeric(out["Age"], errors="coerce").ge(45),
        "p1_age_45_plus",
    )
    _append_reason(
        p1_reasons,
        p1
        & out["Sex"].eq("female")
        & out["PrimaryCompanionScope"].eq("solo")
        & pd.to_numeric(out["Age"], errors="coerce").ge(45),
        "p1_older_solo_female",
    )
    _append_reason(
        p1_reasons,
        p1
        & out["FamilyTicketFateSignal"].eq("mixed")
        & out["FamilySizeBin"].eq("SmallFamily")
        & out["FareBin"].eq("LuxuryFare"),
        "p1_luxury_small_family_mixed_fate",
    )

    out["P3SurvivalEvidence"] = ["|".join(reasons) for reasons in p3_reasons]
    out["P3SurvivalEvidenceCount"] = [len(reasons) for reasons in p3_reasons]
    out["P1DeathEvidence"] = ["|".join(reasons) for reasons in p1_reasons]
    out["P1DeathEvidenceCount"] = [len(reasons) for reasons in p1_reasons]
    return out


def build_test_watchlist(
    builder: PassengerFeatureBuilder,
    test_path: Path,
    strategy: str,
) -> pd.DataFrame:
    if not test_path.exists():
        return pd.DataFrame()

    test_df = pd.read_csv(test_path)
    test_features = builder.transform(test_df)
    test_features = add_child_group_features(test_features)
    test_features = add_rule_columns(test_features)
    test_prediction = predict_strategy(test_features, strategy)
    test_features = add_prediction_columns(test_features, test_prediction)
    test_features = add_watchlist_scores(test_features)

    p3_watch = test_features[
        test_features["Pclass"].eq(3)
        & test_features["Predicted"].eq(0)
        & test_features["P3SurvivalEvidenceCount"].gt(0)
    ].copy()
    p3_watch["WatchlistSlice"] = "pclass3_predicted_dead_with_survival_evidence"

    p1_watch = test_features[
        test_features["Pclass"].eq(1)
        & test_features["Predicted"].eq(1)
        & test_features["P1DeathEvidenceCount"].gt(0)
    ].copy()
    p1_watch["WatchlistSlice"] = "pclass1_predicted_alive_with_death_evidence"

    watchlist = pd.concat([p3_watch, p1_watch], ignore_index=True)
    if watchlist.empty:
        return watchlist

    columns = safe_columns(
        watchlist,
        [
            "WatchlistSlice",
            "PassengerId",
            "Predicted",
            "Name",
            "Sex",
            "Pclass",
            "Title",
            "Age",
            "AgeBin",
            "Embarked",
            "Fare",
            "FareBin",
            "FarePerTicketMember",
            "FarePerTicketBin",
            "FamilySize",
            "FamilySizeBin",
            "Ticket",
            "TicketNormalized",
            "TicketGroupSize",
            "TicketChildFemalePattern",
            "PrimaryCompanionScope",
            "PrimaryGroupChildFemalePattern",
            "PrimaryFateScope",
            "PrimaryFateSignal",
            "PrimaryFateSupportBin",
            "Cabin",
            "CabinKnown",
            "Deck",
            "SpatialAccessTier",
            "P3SurvivalEvidenceCount",
            "P3SurvivalEvidence",
            "P1DeathEvidenceCount",
            "P1DeathEvidence",
        ],
    )
    return watchlist[columns].sort_values(
        [
            "WatchlistSlice",
            "P3SurvivalEvidenceCount",
            "P1DeathEvidenceCount",
            "PassengerId",
        ],
        ascending=[True, False, False, True],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    if "Survived" not in train_df.columns:
        raise ValueError(f"{args.train} must contain Survived.")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    builder = PassengerFeatureBuilder(feature_set="full")
    features = builder.fit_transform(X, y)
    features = add_child_group_features(features)
    features = add_rule_columns(features)
    features["Survived"] = y

    prediction = predict_strategy(features, args.strategy)
    features = add_prediction_columns(features, prediction)
    strategy_accuracy = accuracy_score(y, prediction)

    p3_survivor = features["Pclass"].eq(3) & features["Survived"].eq(1)
    p3_death = features["Pclass"].eq(3) & features["Survived"].eq(0)
    p1_death = features["Pclass"].eq(1) & features["Survived"].eq(0)
    p1_survivor = features["Pclass"].eq(1) & features["Survived"].eq(1)

    lift = pd.concat(
        [
            categorical_lift(
                features,
                p3_survivor,
                p3_death,
                "pclass3_survivors_vs_pclass3_deaths",
            ),
            categorical_lift(
                features,
                p1_death,
                p1_survivor,
                "pclass1_deaths_vs_pclass1_survivors",
            ),
        ],
        ignore_index=True,
    )
    lift.to_csv(args.output_dir / "exception_categorical_lift.csv", index=False)

    numeric = pd.concat(
        [
            numeric_profile(
                features,
                p3_survivor,
                p3_death,
                "pclass3_survivors_vs_pclass3_deaths",
            ),
            numeric_profile(
                features,
                p1_death,
                p1_survivor,
                "pclass1_deaths_vs_pclass1_survivors",
            ),
        ],
        ignore_index=True,
    )
    numeric.to_csv(args.output_dir / "exception_numeric_profile.csv", index=False)

    rules = candidate_rules(
        features,
        min_p3_count=args.min_p3_count,
        min_p1_count=args.min_p1_count,
    )
    rules.to_csv(args.output_dir / "exception_rule_candidates.csv", index=False)

    exception_errors = features[
        (
            features["Pclass"].eq(3)
            & features["Survived"].eq(1)
            & features["Predicted"].eq(0)
        )
        | (
            features["Pclass"].eq(1)
            & features["Survived"].eq(0)
            & features["Predicted"].eq(1)
        )
    ].copy()
    exception_errors["ExceptionSlice"] = np.where(
        exception_errors["Pclass"].eq(3),
        "pclass3_survivor_false_negative",
        "pclass1_death_false_positive",
    )
    exception_errors = exception_errors[safe_columns(exception_errors, EXAMPLE_COLUMNS + ["ExceptionSlice"])]
    exception_errors.to_csv(
        args.output_dir / "exception_current_best_errors.csv",
        index=False,
    )

    test_watchlist = build_test_watchlist(builder, args.test, args.strategy)
    test_watchlist.to_csv(
        args.output_dir / "exception_test_watchlist.csv",
        index=False,
    )

    write_markdown_summary(
        args.output_dir / "exception_profiles.md",
        features,
        args.strategy,
        strategy_accuracy,
        lift,
        rules,
        exception_errors,
    )

    print(f"Strategy: {args.strategy}")
    print(f"Train accuracy: {strategy_accuracy:.4f}")
    print(f"Exception profile saved to: {args.output_dir / 'exception_profiles.md'}")
    print(f"Categorical lift saved to: {args.output_dir / 'exception_categorical_lift.csv'}")
    print(f"Numeric profile saved to: {args.output_dir / 'exception_numeric_profile.csv'}")
    print(f"Rule candidates saved to: {args.output_dir / 'exception_rule_candidates.csv'}")
    print(
        f"Current-best exception errors saved to: "
        f"{args.output_dir / 'exception_current_best_errors.csv'}"
    )
    print(f"Test watchlist saved to: {args.output_dir / 'exception_test_watchlist.csv'}")


if __name__ == "__main__":
    main()
