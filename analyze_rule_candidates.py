from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from optimize_submission import STRATEGIES, predict_strategy
from src.features import PassengerFeatureBuilder
from src.travel_group import count_bin


TRAIN_PATH = Path("data/train.csv")
REPORTS_DIR = Path("reports")
REPEAT_RANDOM_STATES = [42, 7, 99, 2026, 31415]

RULE_FEATURES = [
    "Sex",
    "PclassRule",
    "Title",
    "AgeBin",
    "AgeLE8",
    "AgeLE12",
    "FamilySizeBin",
    "FamilySizeLE4",
    "FamilySizeGE5",
    "FamilySizeGE7",
    "IsAloneRule",
    "TicketGroupSizeBin",
    "FamilyTicketGroupSizeBin",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "ChildFemaleCompanionPattern",
    "CompanionGroupSizeBin",
    "CompanionChildCountBin",
    "CompanionFemaleCountBin",
    "CompanionAdultMaleCountBin",
    "CompanionAccompanyingChildCountBin",
    "CompanionAccompanyingAdultMaleCountBin",
    "TicketCompositionType",
    "TicketPrefix",
    "TicketNumberBand",
    "TicketPrefixEmbarked",
    "TicketNumberBandEmbarked",
    "TicketPrefixPclassEmbarked",
    "TicketFamilyPattern",
    "TicketChildFemalePattern",
    "TicketChildCountBin",
    "TicketFemaleCountBin",
    "TicketFateSignal",
    "TicketFateSupportBin",
    "FamilyTicketFateSignal",
    "FamilyTicketFateSupportBin",
    "PrimaryFateScope",
    "PrimaryFateSignal",
    "PrimaryFateSupportBin",
    "TravelPartyType",
    "FareObjectInterpretation",
    "FareBin",
    "FarePerTicketBin",
    "FareBinPclass",
    "FarePerTicketBinPclass",
    "Embarked",
]


@dataclass(frozen=True)
class Atom:
    feature: str
    value: str

    @property
    def label(self) -> str:
        return f"{self.feature} == {self.value}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine conservative Titanic rule candidates from entity/object features."
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument(
        "--baseline",
        choices=sorted(STRATEGIES),
        default="conservative_hybrid",
        help="Strategy to override. Default: conservative_hybrid",
    )
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-changed", type=int, default=3)
    parser.add_argument("--min-fold-changed", type=int, default=2)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument("--max-generated-conditions", type=int, default=30000)
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--death-max-survival-rate", type=float, default=0.35)
    parser.add_argument("--survival-min-survival-rate", type=float, default=0.65)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def yes_no(mask: pd.Series) -> pd.Series:
    return np.where(mask.fillna(False), "yes", "no")


def add_rule_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()

    out["PclassRule"] = "P" + out["Pclass"].fillna("Unknown").astype(str)
    out["AgeLE8"] = yes_no(out["Age"].le(8))
    out["AgeLE12"] = yes_no(out["Age"].le(12))
    out["AgeLE16"] = yes_no(out["Age"].le(16))
    out["AgeGE18"] = yes_no(out["Age"].ge(18))
    out["AgeKnownRule"] = yes_no(out["Age"].notna())
    out["FamilySizeLE4"] = yes_no(out["FamilySize"].le(4))
    out["FamilySizeGE5"] = yes_no(out["FamilySize"].ge(5))
    out["FamilySizeGE7"] = yes_no(out["FamilySize"].ge(7))
    out["IsAloneRule"] = yes_no(out["IsAlone"].eq(1))

    for col in [
        "TicketGroupSize",
        "FamilyGroupSize",
        "FamilyTicketGroupSize",
        "CompanionGroupSize",
    ]:
        if col in out.columns:
            out[f"{col}Bin"] = out[col].map(count_bin)

    for feature in RULE_FEATURES:
        if feature not in out.columns:
            out[feature] = "Unknown"
        out[feature] = out[feature].fillna("Unknown").astype(str)

    return out


def atom_mask(frame: pd.DataFrame, atom: Atom) -> pd.Series:
    if atom.feature not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[atom.feature].fillna("Unknown").astype(str).eq(atom.value)


def condition_mask(frame: pd.DataFrame, atoms: tuple[Atom, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for atom in atoms:
        mask &= atom_mask(frame, atom)
    return mask


def condition_label(atoms: tuple[Atom, ...]) -> str:
    return " AND ".join(atom.label for atom in atoms)


def generate_atoms(features: pd.DataFrame, min_support: int) -> list[Atom]:
    atoms = []
    for feature in RULE_FEATURES:
        if feature not in features.columns:
            continue
        counts_by_value = features[feature].fillna("Unknown").astype(str).value_counts()
        if len(counts_by_value) > 40:
            continue
        for value, support in counts_by_value.items():
            if value == "Unknown":
                continue
            if min_support <= support < len(features):
                atoms.append(Atom(feature=feature, value=value))
    return sorted(atoms, key=lambda atom: (atom.feature, atom.value))


def generate_conditions(
    features: pd.DataFrame,
    atoms: list[Atom],
    min_support: int,
    max_conditions: int,
    max_generated_conditions: int,
) -> list[tuple[Atom, ...]]:
    atom_masks = {atom: atom_mask(features, atom) for atom in atoms}
    atom_positions = {atom: idx for idx, atom in enumerate(atoms)}
    all_conditions: list[tuple[Atom, ...]] = []
    current: list[tuple[tuple[Atom, ...], pd.Series]] = [
        ((atom,), mask)
        for atom, mask in atom_masks.items()
        if int(mask.sum()) >= min_support
    ]
    all_conditions.extend(condition for condition, _ in current)

    for _depth in range(2, max_conditions + 1):
        next_level: list[tuple[tuple[Atom, ...], pd.Series]] = []
        seen = set()
        for condition, mask in current:
            used_features = {atom.feature for atom in condition}
            last_atom = condition[-1]
            last_index = atom_positions[last_atom]
            for atom in atoms[last_index + 1 :]:
                if atom.feature in used_features:
                    continue
                next_condition = condition + (atom,)
                key = tuple(sorted(next_condition, key=lambda item: item.label))
                if key in seen:
                    continue
                next_mask = mask & atom_masks[atom]
                if int(next_mask.sum()) < min_support:
                    continue
                seen.add(key)
                next_level.append((key, next_mask))
                if len(all_conditions) + len(next_level) >= max_generated_conditions:
                    break
            if len(all_conditions) + len(next_level) >= max_generated_conditions:
                break
        all_conditions.extend(condition for condition, _ in next_level)
        current = next_level
        if not current:
            break
        if len(all_conditions) >= max_generated_conditions:
            break

    return all_conditions


def apply_rule(
    baseline_prediction: pd.Series,
    mask: pd.Series,
    target_prediction: int,
) -> pd.Series:
    prediction = baseline_prediction.copy()
    prediction.loc[mask] = target_prediction
    return prediction.astype(int)


def candidate_full_stats(
    features: pd.DataFrame,
    y: pd.Series,
    baseline_strategy: str,
    conditions: list[tuple[Atom, ...]],
    min_changed: int,
    death_max_survival_rate: float,
    survival_min_survival_rate: float,
) -> pd.DataFrame:
    baseline_prediction = predict_strategy(features, baseline_strategy)
    baseline_accuracy = accuracy_score(y, baseline_prediction)

    rows = []
    rule_ids = count(1)
    for atoms in conditions:
        mask = condition_mask(features, atoms)
        support = int(mask.sum())
        if support == 0:
            continue

        for target_prediction in [0, 1]:
            changed = mask & baseline_prediction.ne(target_prediction)
            changed_count = int(changed.sum())
            if changed_count < min_changed:
                continue

            changed_survival_rate = float(y.loc[changed].mean())
            if target_prediction == 0 and changed_survival_rate > death_max_survival_rate:
                continue
            if (
                target_prediction == 1
                and changed_survival_rate < survival_min_survival_rate
            ):
                continue

            rule_prediction = apply_rule(baseline_prediction, mask, target_prediction)
            rule_accuracy = accuracy_score(y, rule_prediction)
            corrected = int(
                (baseline_prediction.ne(y) & rule_prediction.eq(y) & changed).sum()
            )
            introduced = int(
                (baseline_prediction.eq(y) & rule_prediction.ne(y) & changed).sum()
            )
            rows.append(
                {
                    "rule_id": next(rule_ids),
                    "condition": condition_label(atoms),
                    "target_prediction": target_prediction,
                    "condition_count": support,
                    "changed_count": changed_count,
                    "changed_survival_rate": changed_survival_rate,
                    "baseline_accuracy": baseline_accuracy,
                    "rule_accuracy": rule_accuracy,
                    "accuracy_delta": rule_accuracy - baseline_accuracy,
                    "corrected_errors": corrected,
                    "introduced_errors": introduced,
                    "net_corrections": corrected - introduced,
                    "atoms": atoms,
                }
            )

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)
    candidates = candidates[candidates["accuracy_delta"].gt(0)].copy()
    return candidates.sort_values(
        ["accuracy_delta", "net_corrections", "changed_count"],
        ascending=[False, False, False],
    )


def fold_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = PassengerFeatureBuilder(feature_set="full")
    train_features = add_rule_columns(builder.fit_transform(X_train, y_train))
    valid_features = add_rule_columns(builder.transform(X_valid))
    return train_features, valid_features


def cv_rule_stats(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_rows: pd.DataFrame,
    baseline_strategy: str,
    cv_folds: int,
    min_fold_changed: int,
    death_max_survival_rate: float,
    survival_min_survival_rate: float,
) -> pd.DataFrame:
    rows = []
    for random_state in REPEAT_RANDOM_STATES:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_valid = X.iloc[valid_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_valid = y.iloc[valid_idx].reset_index(drop=True)
            train_features, valid_features = fold_features(X_train, X_valid, y_train)
            train_base = predict_strategy(train_features, baseline_strategy)
            valid_base = predict_strategy(valid_features, baseline_strategy)
            valid_base_accuracy = accuracy_score(y_valid, valid_base)

            for _, candidate in candidate_rows.iterrows():
                atoms = candidate["atoms"]
                target_prediction = int(candidate["target_prediction"])

                train_mask = condition_mask(train_features, atoms)
                train_changed = train_mask & train_base.ne(target_prediction)
                train_changed_count = int(train_changed.sum())
                selected = train_changed_count >= min_fold_changed
                train_changed_survival_rate = np.nan
                if selected:
                    train_changed_survival_rate = float(y_train.loc[train_changed].mean())
                    if (
                        target_prediction == 0
                        and train_changed_survival_rate > death_max_survival_rate
                    ):
                        selected = False
                    if (
                        target_prediction == 1
                        and train_changed_survival_rate < survival_min_survival_rate
                    ):
                        selected = False

                valid_mask = condition_mask(valid_features, atoms)
                valid_changed = valid_mask & valid_base.ne(target_prediction)
                if selected:
                    valid_prediction = apply_rule(
                        valid_base,
                        valid_mask,
                        target_prediction,
                    )
                else:
                    valid_prediction = valid_base

                valid_rule_accuracy = accuracy_score(y_valid, valid_prediction)
                corrected = int(
                    (
                        valid_base.ne(y_valid)
                        & valid_prediction.eq(y_valid)
                        & valid_changed
                    ).sum()
                )
                introduced = int(
                    (
                        valid_base.eq(y_valid)
                        & valid_prediction.ne(y_valid)
                        & valid_changed
                    ).sum()
                )
                rows.append(
                    {
                        "rule_id": candidate["rule_id"],
                        "condition": candidate["condition"],
                        "target_prediction": target_prediction,
                        "random_state": random_state,
                        "fold": fold,
                        "selected": selected,
                        "train_changed_count": train_changed_count,
                        "train_changed_survival_rate": train_changed_survival_rate,
                        "valid_changed_count": int(valid_changed.sum()) if selected else 0,
                        "valid_baseline_accuracy": valid_base_accuracy,
                        "valid_rule_accuracy": valid_rule_accuracy,
                        "valid_accuracy_delta": valid_rule_accuracy
                        - valid_base_accuracy,
                        "valid_corrected_errors": corrected,
                        "valid_introduced_errors": introduced,
                        "valid_net_corrections": corrected - introduced,
                    }
                )

    return pd.DataFrame(rows)


def summarize_cv(
    candidates: pd.DataFrame,
    cv_scores: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        cv_scores.groupby(["rule_id", "condition", "target_prediction"])
        .agg(
            selected_folds=("selected", "sum"),
            cv_runs=("selected", "count"),
            valid_accuracy_delta_mean=("valid_accuracy_delta", "mean"),
            valid_accuracy_delta_std=("valid_accuracy_delta", "std"),
            valid_changed_count_sum=("valid_changed_count", "sum"),
            valid_corrected_errors_sum=("valid_corrected_errors", "sum"),
            valid_introduced_errors_sum=("valid_introduced_errors", "sum"),
            valid_net_corrections_sum=("valid_net_corrections", "sum"),
            train_changed_count_mean=("train_changed_count", "mean"),
            train_changed_survival_rate_mean=("train_changed_survival_rate", "mean"),
        )
        .reset_index()
    )
    summary["selected_rate"] = summary["selected_folds"] / summary["cv_runs"]
    full_cols = [
        "rule_id",
        "condition_count",
        "changed_count",
        "changed_survival_rate",
        "accuracy_delta",
        "corrected_errors",
        "introduced_errors",
        "net_corrections",
    ]
    merged = summary.merge(candidates[full_cols], on="rule_id", how="left")
    return merged.sort_values(
        [
            "valid_net_corrections_sum",
            "valid_accuracy_delta_mean",
            "selected_rate",
            "accuracy_delta",
        ],
        ascending=[False, False, False, False],
    )


def serializable_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    output = candidates.drop(columns=["atoms"], errors="ignore").copy()
    return output.sort_values(
        ["accuracy_delta", "net_corrections", "changed_count"],
        ascending=[False, False, False],
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
    features = add_rule_columns(builder.fit_transform(X, y))

    atoms = generate_atoms(features, min_support=args.min_support)
    conditions = generate_conditions(
        features,
        atoms,
        min_support=args.min_support,
        max_conditions=args.max_conditions,
        max_generated_conditions=args.max_generated_conditions,
    )
    print(f"Generated atoms: {len(atoms)}")
    print(f"Generated conditions: {len(conditions)}")

    candidates = candidate_full_stats(
        features,
        y,
        baseline_strategy=args.baseline,
        conditions=conditions,
        min_changed=args.min_changed,
        death_max_survival_rate=args.death_max_survival_rate,
        survival_min_survival_rate=args.survival_min_survival_rate,
    )
    if candidates.empty:
        print("No positive rule candidates found.")
        return

    candidates = candidates.head(args.top_n).copy()
    candidate_path = args.output_dir / f"rule_candidates_{args.baseline}.csv"
    serializable_candidates(candidates).to_csv(candidate_path, index=False)

    cv_scores = cv_rule_stats(
        X,
        y,
        candidates,
        baseline_strategy=args.baseline,
        cv_folds=args.cv_folds,
        min_fold_changed=args.min_fold_changed,
        death_max_survival_rate=args.death_max_survival_rate,
        survival_min_survival_rate=args.survival_min_survival_rate,
    )
    cv_path = args.output_dir / f"rule_cv_scores_{args.baseline}.csv"
    cv_scores.to_csv(cv_path, index=False)

    cv_summary = summarize_cv(candidates, cv_scores)
    summary_path = args.output_dir / f"rule_cv_summary_{args.baseline}.csv"
    cv_summary.to_csv(summary_path, index=False)

    printable = cv_summary.head(20).copy()
    for col in [
        "valid_accuracy_delta_mean",
        "valid_accuracy_delta_std",
        "train_changed_survival_rate_mean",
        "changed_survival_rate",
        "accuracy_delta",
        "selected_rate",
    ]:
        printable[col] = printable[col].round(5)

    print(printable.to_string(index=False))
    print(f"Rule candidates saved to: {candidate_path}")
    print(f"Rule CV scores saved to: {cv_path}")
    print(f"Rule CV summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
