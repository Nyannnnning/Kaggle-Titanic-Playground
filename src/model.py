from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURE_SETS, PassengerFeatureBuilder, feature_columns_for_set


MODEL_TYPES = {
    "logistic_score",
    "hist_gradient_boosting",
    "random_forest_optional",
    "xgboost",
}


def build_preprocessor(model_type: str, feature_set: str = "clean") -> ColumnTransformer:
    numeric_features, categorical_features = feature_columns_for_set(feature_set)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "logistic_score":
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )


def build_classifier(model_type: str):
    if model_type == "logistic_score":
        return LogisticRegression(max_iter=2000, random_state=42)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=42,
        )
    if model_type == "random_forest_optional":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=140,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=15.0,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model(model_type: str = "logistic_score", feature_set: str = "clean") -> Pipeline:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    return Pipeline(
        steps=[
            (
                "feature_builder",
                PassengerFeatureBuilder(feature_set=feature_set),
            ),
            ("preprocessing", build_preprocessor(model_type, feature_set=feature_set)),
            ("classifier", build_classifier(model_type)),
        ]
    )


def get_classifier(model: Pipeline):
    return model.named_steps["classifier"]


def get_feature_names(model: Pipeline) -> np.ndarray:
    return model.named_steps["preprocessing"].get_feature_names_out()


def predict_survival_probability(model: Pipeline, X) -> np.ndarray:
    classifier = get_classifier(model)
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def survival_scores(model: Pipeline, X) -> np.ndarray:
    classifier = get_classifier(model)
    if hasattr(classifier, "decision_function"):
        return model.decision_function(X)
    return predict_survival_probability(model, X)
