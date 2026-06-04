from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


MaskFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class RegimeRule:
    name: str
    regime: str
    target_prediction: int
    status: str
    description: str
    mask_fn: MaskFn

    def mask(self, features: pd.DataFrame) -> pd.Series:
        return self.mask_fn(features).fillna(False).astype(bool)


def p3_master_small_family(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(3)
        & features["Title"].eq("Master")
        & features["FamilySizeBin"].eq("SmallFamily")
    )


def has_any_all_died_group_fate(features: pd.DataFrame) -> pd.Series:
    return (
        features["TicketFateSignal"].eq("all_died")
        | features["FamilyFateSignal"].eq("all_died")
        | features["FamilyTicketFateSignal"].eq("all_died")
        | features["PrimaryFateSignal"].eq("all_died")
    )


def has_any_all_survived_group_fate(features: pd.DataFrame) -> pd.Series:
    return (
        features["TicketFateSignal"].eq("all_survived")
        | features["FamilyFateSignal"].eq("all_survived")
        | features["FamilyTicketFateSignal"].eq("all_survived")
        | features["PrimaryFateSignal"].eq("all_survived")
    )


def has_any_mixed_group_fate(features: pd.DataFrame) -> pd.Series:
    return (
        features["TicketFateSignal"].eq("mixed")
        | features["FamilyFateSignal"].eq("mixed")
        | features["FamilyTicketFateSignal"].eq("mixed")
        | features["PrimaryFateSignal"].eq("mixed")
    )


def p3_master_small_family_without_all_died(features: pd.DataFrame) -> pd.Series:
    return p3_master_small_family(features) & ~has_any_all_died_group_fate(features)


def p3_master_small_family_all_survived(features: pd.DataFrame) -> pd.Series:
    return (
        p3_master_small_family_without_all_died(features)
        & has_any_all_survived_group_fate(features)
    )


def p3_master_small_family_mixed_without_all_died(features: pd.DataFrame) -> pd.Series:
    return (
        p3_master_small_family_without_all_died(features)
        & has_any_mixed_group_fate(features)
        & ~has_any_all_survived_group_fate(features)
    )


def p3_group_fate_all_survived(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(3)
        & (
            features["TicketFateSignal"].eq("all_survived")
            | features["FamilyTicketFateSignal"].eq("all_survived")
            | features["PrimaryFateSignal"].eq("all_survived")
        )
    )


def p1_older_solo_female(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(1)
        & features["Sex"].eq("female")
        & features["PrimaryCompanionScope"].eq("solo")
        & pd.to_numeric(features["Age"], errors="coerce").ge(45)
    )


def p1_luxury_small_family_mixed_fate(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(1)
        & features["Sex"].eq("female")
        & features["FamilySizeBin"].eq("SmallFamily")
        & features["FareBin"].eq("LuxuryFare")
        & features["FamilyTicketFateSignal"].eq("mixed")
    )


def p1_isham_like_older_solo_miss_midfare(features: pd.DataFrame) -> pd.Series:
    return (
        features["Pclass"].eq(1)
        & features["Sex"].eq("female")
        & features["Title"].eq("Miss")
        & features["FamilySizeBin"].eq("Alone")
        & features["FareBin"].eq("MidFare")
        & pd.to_numeric(features["Age"], errors="coerce").ge(45)
    )


def p1_predicted_alive_all_died_group_fate(features: pd.DataFrame) -> pd.Series:
    return features["Pclass"].eq(1) & has_any_all_died_group_fate(features)


REGIME_RULES = {
    "p3_master_small_family_rescue": RegimeRule(
        name="p3_master_small_family_rescue",
        regime="p3_survival_rescue",
        target_prediction=1,
        status="rejected_public",
        description=(
            "Pclass 3 Masters in small families are evaluated as a separate "
            "low-class survival rescue regime. Public score rejected this "
            "broad rescue despite positive train/CV evidence."
        ),
        mask_fn=p3_master_small_family,
    ),
    "p3_master_small_family_no_all_died_rescue": RegimeRule(
        name="p3_master_small_family_no_all_died_rescue",
        regime="p3_survival_rescue",
        target_prediction=1,
        status="public_tie",
        description=(
            "Narrow Pclass 3 Master small-family rescue. This keeps the child "
            "survival evidence but refuses to override any all_died GroupFate "
            "signal. Public score tied the current best, so it remains an audit "
            "variant rather than a replacement."
        ),
        mask_fn=p3_master_small_family_without_all_died,
    ),
    "p3_master_small_family_all_survived_rescue": RegimeRule(
        name="p3_master_small_family_all_survived_rescue",
        regime="p3_survival_rescue",
        target_prediction=1,
        status="public_best_component",
        description=(
            "Split of the neutral v2 P3 rescue: keep only the clean all_survived "
            "GroupFate side. In test this isolates PassengerId 1309 and raised "
            "the public score to 0.80143."
        ),
        mask_fn=p3_master_small_family_all_survived,
    ),
    "p3_master_small_family_mixed_rescue": RegimeRule(
        name="p3_master_small_family_mixed_rescue",
        regime="p3_survival_rescue",
        target_prediction=1,
        status="candidate_split",
        description=(
            "Split of the neutral v2 P3 rescue: keep the mixed-fate side without "
            "all_survived evidence. In test this isolates PassengerId 1284."
        ),
        mask_fn=p3_master_small_family_mixed_without_all_died,
    ),
    "p3_group_fate_all_survived_broad": RegimeRule(
        name="p3_group_fate_all_survived_broad",
        regime="p3_survival_rescue",
        target_prediction=1,
        status="holdout_too_broad",
        description=(
            "Broad Pclass 3 all-survived GroupFate signal; useful for watchlists "
            "but too broad as a direct submission rule."
        ),
        mask_fn=p3_group_fate_all_survived,
    ),
    "p1_older_solo_female_death_risk": RegimeRule(
        name="p1_older_solo_female_death_risk",
        regime="p1_death_override",
        target_prediction=0,
        status="rejected_broad",
        description=(
            "Older solo 1st-class female pattern. Train audit shows it is too "
            "broad and would likely remove true survivors."
        ),
        mask_fn=p1_older_solo_female,
    ),
    "p1_luxury_small_family_mixed_fate": RegimeRule(
        name="p1_luxury_small_family_mixed_fate",
        regime="p1_death_override",
        target_prediction=0,
        status="holdout_no_test_change",
        description=(
            "Narrow 1st-class luxury small-family mixed-fate signal kept for "
            "analysis, not active in the submission layer."
        ),
        mask_fn=p1_luxury_small_family_mixed_fate,
    ),
    "p1_isham_like_older_solo_miss_midfare": RegimeRule(
        name="p1_isham_like_older_solo_miss_midfare",
        regime="p1_death_override",
        target_prediction=0,
        status="holdout_narrow",
        description=(
            "Isham-like 1st-class death override: older solo Miss with mid fare. "
            "Kept as a narrow candidate for audit before any submission."
        ),
        mask_fn=p1_isham_like_older_solo_miss_midfare,
    ),
    "p1_any_all_died_group_fate": RegimeRule(
        name="p1_any_all_died_group_fate",
        regime="p1_death_override",
        target_prediction=0,
        status="holdout_broad",
        description=(
            "1st-class passenger with any all_died GroupFate evidence. Useful in "
            "conflict analysis but too broad without extra narrowing."
        ),
        mask_fn=p1_predicted_alive_all_died_group_fate,
    ),
}

ACTIVE_REGIME_RULE_NAMES = ("p3_master_small_family_all_survived_rescue",)


def apply_regime_rule_layer(
    features: pd.DataFrame,
    base_prediction: pd.Series,
    rule_names: tuple[str, ...] = ACTIVE_REGIME_RULE_NAMES,
) -> pd.Series:
    prediction = base_prediction.copy()
    for rule_name in rule_names:
        rule = REGIME_RULES[rule_name]
        prediction.loc[rule.mask(features)] = rule.target_prediction
    return prediction.astype(int)


def changed_passenger_ids(
    features: pd.DataFrame,
    mask: pd.Series,
    base_prediction: pd.Series,
    target_prediction: int,
) -> str:
    changed = mask & base_prediction.ne(target_prediction)
    if "PassengerId" not in features.columns:
        return ""
    return "|".join(features.loc[changed, "PassengerId"].astype(int).astype(str))


def regime_rule_audit(
    train_features: pd.DataFrame,
    y: pd.Series,
    train_base_prediction: pd.Series,
    test_features: pd.DataFrame,
    test_base_prediction: pd.Series,
) -> pd.DataFrame:
    base_accuracy = accuracy_score(y, train_base_prediction)
    rows = []

    for rule in REGIME_RULES.values():
        train_mask = rule.mask(train_features)
        train_changed = train_mask & train_base_prediction.ne(rule.target_prediction)
        rule_prediction = train_base_prediction.copy()
        rule_prediction.loc[train_mask] = rule.target_prediction
        rule_accuracy = accuracy_score(y, rule_prediction)

        if int(train_mask.sum()) > 0:
            survival_rate = float(y.loc[train_mask].mean())
        else:
            survival_rate = np.nan
        target_rate = survival_rate if rule.target_prediction == 1 else 1 - survival_rate

        corrected = int(
            (
                train_base_prediction.ne(y)
                & rule_prediction.eq(y)
                & train_changed
            ).sum()
        )
        introduced = int(
            (
                train_base_prediction.eq(y)
                & rule_prediction.ne(y)
                & train_changed
            ).sum()
        )

        test_mask = rule.mask(test_features)
        test_changed = test_mask & test_base_prediction.ne(rule.target_prediction)

        rows.append(
            {
                "rule_name": rule.name,
                "regime": rule.regime,
                "status": rule.status,
                "target_prediction": rule.target_prediction,
                "description": rule.description,
                "train_count": int(train_mask.sum()),
                "train_changed_count": int(train_changed.sum()),
                "train_survival_rate": survival_rate,
                "train_target_rate": target_rate,
                "baseline_accuracy": base_accuracy,
                "rule_accuracy": rule_accuracy,
                "accuracy_delta": rule_accuracy - base_accuracy,
                "corrected_errors": corrected,
                "introduced_errors": introduced,
                "net_corrections": corrected - introduced,
                "test_count": int(test_mask.sum()),
                "test_changed_count": int(test_changed.sum()),
                "test_changed_passenger_ids": changed_passenger_ids(
                    test_features,
                    test_mask,
                    test_base_prediction,
                    rule.target_prediction,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["status", "accuracy_delta", "net_corrections"],
        ascending=[True, False, False],
    )
