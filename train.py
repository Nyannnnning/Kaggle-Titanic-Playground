from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.explain import write_logistic_coefficients
from src.features import FEATURE_SETS, WEAK_SIGNAL_FEATURES, feature_columns_for_set
from src.geometry import (
    centroid_distance_frame,
    error_type,
    write_passenger_clusters,
    write_prototype_profiles,
    write_survival_geometry_validation,
)
from src.model import MODEL_TYPES, build_model, predict_survival_probability, survival_scores
from src.passenger_report import write_passenger_reports
from src.shap_explain import write_shap_reports


DEFAULT_TRAIN_PATH = Path("data/train.csv")
DEFAULT_MODEL_PATH = Path("models/titanic_model.pkl")
REPORTS_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Titanic survival model.")
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train.csv. Default: {DEFAULT_TRAIN_PATH}",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the trained model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_TYPES),
        default="logistic_score",
        help="Model type to train. Default: logistic_score",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="clean",
        help="Feature set to use. Default: clean",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds. Use 0 to skip. Default: 5",
    )
    return parser.parse_args()


def write_validation_scores(
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    predictions,
    probabilities,
    scores,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = y_valid.astype(int).to_numpy()
    y_pred = predictions.astype(int)
    validation_scores = pd.DataFrame(
        {
            "PassengerId": X_valid["PassengerId"].to_numpy(),
            "Survived": y_true,
            "Predicted": y_pred,
            "SurvivalProbability": probabilities,
            "SurvivalScore": scores,
            "Correct": y_true == y_pred,
            "ErrorType": error_type(y_true, y_pred),
        }
    )
    validation_scores.to_csv(output_path, index=False)
    return validation_scores


def write_feature_set_report(feature_set: str, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_features, categorical_features = feature_columns_for_set(feature_set)
    included_features = set(numeric_features + categorical_features)
    rows = []
    for feature in numeric_features:
        rows.append(
            {
                "feature": feature,
                "kind": "numeric",
                "feature_set": feature_set,
                "included": True,
                "filtered_reason": "",
            }
        )
    for feature in categorical_features:
        rows.append(
            {
                "feature": feature,
                "kind": "categorical",
                "feature_set": feature_set,
                "included": True,
                "filtered_reason": "",
            }
        )
    for feature in WEAK_SIGNAL_FEATURES:
        if feature not in included_features:
            rows.append(
                {
                    "feature": feature,
                    "kind": "weak_signal",
                    "feature_set": feature_set,
                    "included": False,
                    "filtered_reason": "uncertain proxy removed from this feature set",
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    return report


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    feature_set: str,
    cv_folds: int,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv_folds <= 1:
        empty = pd.DataFrame(
            columns=[
                "fold",
                "model_type",
                "feature_set",
                "validation_accuracy",
                "train_size",
                "validation_size",
            ]
        )
        empty.to_csv(output_path, index=False)
        return empty

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model = build_model(model_type, feature_set=feature_set)
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid).astype(int)
        rows.append(
            {
                "fold": fold,
                "model_type": model_type,
                "feature_set": feature_set,
                "validation_accuracy": accuracy_score(y_valid, predictions),
                "train_size": len(X_train),
                "validation_size": len(X_valid),
            }
        )
    scores = pd.DataFrame(rows)
    scores.to_csv(output_path, index=False)
    return scores


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    if "Survived" not in train_df.columns:
        raise ValueError(f"{args.train} must contain a Survived column.")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    write_feature_set_report(args.feature_set, REPORTS_DIR / "model_feature_list.csv")

    cv_scores = run_cross_validation(
        X,
        y,
        args.model_type,
        args.feature_set,
        args.cv_folds,
        REPORTS_DIR / "cross_validation_scores.csv",
    )

    validation_model = build_model(args.model_type, feature_set=args.feature_set)
    validation_model.fit(X_train, y_train)

    valid_predictions = validation_model.predict(X_valid).astype(int)
    valid_probabilities = predict_survival_probability(validation_model, X_valid)
    valid_scores = survival_scores(validation_model, X_valid)
    accuracy = accuracy_score(y_valid, valid_predictions)

    write_validation_scores(
        X_valid,
        y_valid,
        valid_predictions,
        valid_probabilities,
        valid_scores,
        REPORTS_DIR / "validation_scores.csv",
    )
    write_survival_geometry_validation(
        validation_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        valid_predictions,
        valid_probabilities,
        REPORTS_DIR / "survival_geometry_validation.csv",
    )

    final_model = build_model(args.model_type, feature_set=args.feature_set)
    final_model.fit(X, y)
    full_predictions = final_model.predict(X).astype(int)
    full_probabilities = predict_survival_probability(final_model, X)
    full_geometry = centroid_distance_frame(
        final_model,
        X,
        y,
        X,
        y,
        full_predictions,
        full_probabilities,
    )
    full_geometry.to_csv(REPORTS_DIR / "survival_geometry_train.csv", index=False)

    if args.model_type == "logistic_score":
        write_logistic_coefficients(final_model, REPORTS_DIR / "logistic_coefficients.csv")

    shap_force_report = None
    if args.model_type == "xgboost":
        shap_paths = write_shap_reports(
            final_model,
            X,
            full_probabilities,
            REPORTS_DIR,
        )
        shap_force_report = pd.read_csv(shap_paths["passenger_shap_force_report"])

    write_passenger_clusters(
        final_model,
        X,
        y,
        full_probabilities,
        REPORTS_DIR / "passenger_clusters.csv",
    )
    write_prototype_profiles(
        X,
        y,
        full_predictions,
        full_probabilities,
        REPORTS_DIR / "prototype_profiles.json",
    )
    passenger_report_paths = write_passenger_reports(
        final_model,
        X,
        full_probabilities,
        full_geometry,
        REPORTS_DIR,
        shap_force_report=shap_force_report,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_output)

    print(f"Model type: {args.model_type}")
    print(f"Feature set: {args.feature_set}")
    if not cv_scores.empty:
        print(
            "Cross-validation accuracy: "
            f"{cv_scores['validation_accuracy'].mean():.4f} "
            f"+/- {cv_scores['validation_accuracy'].std(ddof=0):.4f}"
        )
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Model saved to: {args.model_output}")
    print(f"Validation scores saved to: {REPORTS_DIR / 'validation_scores.csv'}")
    print(f"Geometry report saved to: {REPORTS_DIR / 'survival_geometry_validation.csv'}")
    print(f"Passenger force report saved to: {passenger_report_paths['passenger_force_report']}")
    print(f"Passenger profiles saved to: {passenger_report_paths['passenger_profiles']}")


if __name__ == "__main__":
    main()
