from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame


MODEL_TYPES = {
    "logistic_score",
    "hist_gradient_boosting",
    "random_forest_optional",
}


def build_preprocessor(model_type: str) -> ColumnTransformer:
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
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
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
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model(model_type: str = "logistic_score") -> Pipeline:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        steps=[
            (
                "feature_builder",
                FunctionTransformer(build_feature_frame, validate=False),
            ),
            ("preprocessing", build_preprocessor(model_type)),
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
