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

from src.features import PassengerFeatureBuilder, feature_columns_for_set


TRAIN_PATH = Path("data/train.csv")
REPORTS_DIR = Path("reports")
REPEAT_RANDOM_STATES = [42, 7, 99, 2026, 31415]

NAME_NUMERIC_FEATURES = [
    "FormalGivenTokenCount",
    "FormalNameHasInitial",
    "FormalNameHasQuote",
    "HasParentheticalName",
]

NAME_CATEGORICAL_FEATURES = [
    "FormalNamePattern",
    "FormalSurnameShape",
    "FormalGivenTokenCountBin",
]

ABLATION_VARIANTS = {
    "clean_baseline": {
        "numeric": [],
        "categorical": [],
    },
    "formal_pattern_only": {
        "numeric": [],
        "categorical": ["FormalNamePattern"],
    },
    "formal_structure_no_parentheses": {
        "numeric": [
            "FormalGivenTokenCount",
            "FormalNameHasInitial",
            "FormalNameHasQuote",
        ],
        "categorical": [
            "FormalNamePattern",
            "FormalSurnameShape",
            "FormalGivenTokenCountBin",
        ],
    },
    "formal_structure_with_parenthetical_flag": {
        "numeric": [
            "FormalGivenTokenCount",
            "FormalNameHasInitial",
            "FormalNameHasQuote",
            "HasParentheticalName",
        ],
        "categorical": [
            "FormalNamePattern",
            "FormalSurnameShape",
            "FormalGivenTokenCountBin",
        ],
    },
    "surname_shape_only": {
        "numeric": [],
        "categorical": ["FormalSurnameShape"],
    },
    "token_count_only": {
        "numeric": ["FormalGivenTokenCount"],
        "categorical": ["FormalGivenTokenCountBin"],
    },
}


def columns_for_variant(variant: str) -> tuple[list[str], list[str]]:
    base_numeric, base_categorical = feature_columns_for_set("clean")
    numeric = [col for col in base_numeric if col not in NAME_NUMERIC_FEATURES]
    categorical = [col for col in base_categorical if col not in NAME_CATEGORICAL_FEATURES]

    config = ABLATION_VARIANTS[variant]
    numeric.extend(config["numeric"])
    categorical.extend(config["categorical"])
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
            ("feature_builder", PassengerFeatureBuilder(feature_set="clean")),
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def run_name_ablation(cv_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                        "name_numeric_features": "|".join(
                            ABLATION_VARIANTS[variant]["numeric"]
                        ),
                        "name_categorical_features": "|".join(
                            ABLATION_VARIANTS[variant]["categorical"]
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
            name_numeric_features=("name_numeric_features", "first"),
            name_categorical_features=("name_categorical_features", "first"),
        )
        .reset_index()
        .sort_values(["accuracy_mean", "log_loss_mean"], ascending=[False, True])
    )
    return scores, summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores, summary = run_name_ablation()
    scores.to_csv(REPORTS_DIR / "name_ablation_scores.csv", index=False)
    summary.to_csv(REPORTS_DIR / "name_ablation_summary.csv", index=False)

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
    print(f"Name ablation scores saved to: {REPORTS_DIR / 'name_ablation_scores.csv'}")
    print(f"Name ablation summary saved to: {REPORTS_DIR / 'name_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
