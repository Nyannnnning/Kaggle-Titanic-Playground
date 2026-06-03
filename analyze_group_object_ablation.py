from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    COMPACT_TRAVEL_GROUP_CATEGORICAL_FEATURES,
    COMPACT_TRAVEL_GROUP_NUMERIC_FEATURES,
    TICKET_GROUP_COMPOSITION_CATEGORICAL_FEATURES,
    TICKET_GROUP_COMPOSITION_NUMERIC_FEATURES,
    PassengerFeatureBuilder,
    feature_columns_for_set,
)


TRAIN_PATH = Path("data/train.csv")
REPORTS_DIR = Path("reports")
REPEAT_RANDOM_STATES = [42, 7, 99, 2026, 31415]

FARE_OBJECT_CONTEXT_NUMERIC = [
    "FarePerTicketMember",
    "FarePerFamilyMember",
    "FarePerFamilyTicketMember",
    "TicketFamilyMismatch",
    "TicketFamilyDelta",
    "HighRawFareLargeFamilyFlag",
    "ZeroFareFlag",
]

FARE_OBJECT_CONTEXT_CATEGORICAL = [
    "FarePerTicketBin",
    "TravelPartyType",
    "FareObjectInterpretation",
    "FarePerTicketBinPclass",
]

PRIMARY_COMPANION_CATEGORICAL = [
    "SelfGroupRole",
    "PrimaryCompanionScope",
    "PrimaryGroupChildFemalePattern",
    "ChildFemaleCompanionPattern",
]

FAMILY_TICKET_NUMERIC = [
    "FamilyTicketChildCount",
    "FamilyTicketFemaleCount",
    "FamilyTicketAdultMaleCount",
    "FamilyTicketAccompanyingChildCount",
    "FamilyTicketAccompanyingFemaleCount",
    "FamilyTicketAccompanyingAdultMaleCount",
]

FAMILY_TICKET_CATEGORICAL = [
    "FamilyTicketChildFemalePattern",
    "FamilyTicketChildCountBin",
    "FamilyTicketFemaleCountBin",
    "FamilyTicketAdultMaleCountBin",
]

FAMILY_NUMERIC = [
    "FamilyChildCount",
    "FamilyFemaleCount",
    "FamilyAdultMaleCount",
    "FamilyAccompanyingChildCount",
    "FamilyAccompanyingFemaleCount",
    "FamilyAccompanyingAdultMaleCount",
]

FAMILY_CATEGORICAL = [
    "FamilyChildFemalePattern",
    "FamilyChildCountBin",
    "FamilyFemaleCountBin",
    "FamilyAdultMaleCountBin",
]

ABLATION_VARIANTS = {
    "clean_baseline": {
        "add_numeric": [],
        "add_categorical": [],
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "no_fare_object_context": {
        "add_numeric": [],
        "add_categorical": [],
        "remove_numeric": FARE_OBJECT_CONTEXT_NUMERIC,
        "remove_categorical": FARE_OBJECT_CONTEXT_CATEGORICAL,
    },
    "primary_companion_pattern_only": {
        "add_numeric": [],
        "add_categorical": PRIMARY_COMPANION_CATEGORICAL,
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "family_ticket_object_only": {
        "add_numeric": FAMILY_TICKET_NUMERIC,
        "add_categorical": FAMILY_TICKET_CATEGORICAL,
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "family_object_only": {
        "add_numeric": FAMILY_NUMERIC,
        "add_categorical": FAMILY_CATEGORICAL,
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "compact_travel_group_object": {
        "add_numeric": COMPACT_TRAVEL_GROUP_NUMERIC_FEATURES,
        "add_categorical": COMPACT_TRAVEL_GROUP_CATEGORICAL_FEATURES,
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "ticket_composition_object": {
        "add_numeric": sorted(TICKET_GROUP_COMPOSITION_NUMERIC_FEATURES),
        "add_categorical": sorted(TICKET_GROUP_COMPOSITION_CATEGORICAL_FEATURES),
        "remove_numeric": [],
        "remove_categorical": [],
    },
    "ticket_plus_companion_object": {
        "add_numeric": sorted(TICKET_GROUP_COMPOSITION_NUMERIC_FEATURES)
        + COMPACT_TRAVEL_GROUP_NUMERIC_FEATURES,
        "add_categorical": sorted(TICKET_GROUP_COMPOSITION_CATEGORICAL_FEATURES)
        + COMPACT_TRAVEL_GROUP_CATEGORICAL_FEATURES,
        "remove_numeric": [],
        "remove_categorical": [],
    },
}


def merge_features(base: list[str], add: list[str], remove: list[str]) -> list[str]:
    remove_set = set(remove)
    merged = [feature for feature in base if feature not in remove_set]
    for feature in add:
        if feature not in merged:
            merged.append(feature)
    return merged


def columns_for_variant(variant: str) -> tuple[list[str], list[str]]:
    base_numeric, base_categorical = feature_columns_for_set("clean")
    config = ABLATION_VARIANTS[variant]
    numeric = merge_features(
        base_numeric,
        config["add_numeric"],
        config["remove_numeric"],
    )
    categorical = merge_features(
        base_categorical,
        config["add_categorical"],
        config["remove_categorical"],
    )
    return numeric, categorical


def build_logistic_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("feature_builder", PassengerFeatureBuilder(feature_set="full")),
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def run_group_object_ablation(cv_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    rows = []
    for random_state in REPEAT_RANDOM_STATES:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        for variant in ABLATION_VARIANTS:
            numeric_features, categorical_features = columns_for_variant(variant)
            for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
                model = build_logistic_model(numeric_features, categorical_features)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
                rows.append(
                    {
                        "variant": variant,
                        "random_state": random_state,
                        "fold": fold,
                        "accuracy": accuracy_score(y.iloc[valid_idx], predictions),
                        "roc_auc": roc_auc_score(y.iloc[valid_idx], probabilities),
                        "log_loss": log_loss(y.iloc[valid_idx], probabilities),
                        "numeric_feature_count": len(numeric_features),
                        "categorical_feature_count": len(categorical_features),
                        "added_numeric_features": "|".join(
                            ABLATION_VARIANTS[variant]["add_numeric"]
                        ),
                        "added_categorical_features": "|".join(
                            ABLATION_VARIANTS[variant]["add_categorical"]
                        ),
                        "removed_numeric_features": "|".join(
                            ABLATION_VARIANTS[variant]["remove_numeric"]
                        ),
                        "removed_categorical_features": "|".join(
                            ABLATION_VARIANTS[variant]["remove_categorical"]
                        ),
                    }
                )

    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby("variant")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            cv_runs=("accuracy", "count"),
            numeric_feature_count=("numeric_feature_count", "first"),
            categorical_feature_count=("categorical_feature_count", "first"),
            added_numeric_features=("added_numeric_features", "first"),
            added_categorical_features=("added_categorical_features", "first"),
            removed_numeric_features=("removed_numeric_features", "first"),
            removed_categorical_features=("removed_categorical_features", "first"),
        )
        .reset_index()
        .sort_values(["accuracy_mean", "log_loss_mean"], ascending=[False, True])
    )
    return scores, summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores, summary = run_group_object_ablation()
    scores.to_csv(REPORTS_DIR / "group_object_ablation_scores.csv", index=False)
    summary.to_csv(REPORTS_DIR / "group_object_ablation_summary.csv", index=False)

    printable = summary.copy()
    for col in [
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "log_loss_mean",
        "log_loss_std",
    ]:
        printable[col] = printable[col].round(5)

    print(printable.to_string(index=False))
    print(
        f"Group-object ablation scores saved to: "
        f"{REPORTS_DIR / 'group_object_ablation_scores.csv'}"
    )
    print(
        f"Group-object ablation summary saved to: "
        f"{REPORTS_DIR / 'group_object_ablation_summary.csv'}"
    )


if __name__ == "__main__":
    main()
