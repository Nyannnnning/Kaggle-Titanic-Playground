from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_frame


def error_type(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    labels = []
    for truth, pred in zip(y_true, y_pred, strict=False):
        if truth == 1 and pred == 1:
            labels.append("true_positive")
        elif truth == 0 and pred == 0:
            labels.append("true_negative")
        elif truth == 0 and pred == 1:
            labels.append("false_positive")
        else:
            labels.append("false_negative")
    return labels


def preprocessed_matrix(model, X) -> np.ndarray:
    feature_frame = model.named_steps["feature_builder"].transform(X)
    matrix = model.named_steps["preprocessing"].transform(feature_frame)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def scaled_feature_spaces(model, X_fit, X_transform):
    fit_matrix = preprocessed_matrix(model, X_fit)
    transform_matrix = preprocessed_matrix(model, X_transform)
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_matrix)
    transform_scaled = scaler.transform(transform_matrix)
    return fit_scaled, transform_scaled, scaler


def centroid_distance_frame(
    model,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    n_clusters: int = 4,
) -> pd.DataFrame:
    fit_scaled, eval_scaled, _ = scaled_feature_spaces(model, X_fit, X_eval)
    y_fit_values = np.asarray(y_fit)

    survivor_centroid = fit_scaled[y_fit_values == 1].mean(axis=0)
    non_survivor_centroid = fit_scaled[y_fit_values == 0].mean(axis=0)

    distance_to_survivor = pairwise_distances(
        eval_scaled,
        survivor_centroid.reshape(1, -1),
    ).ravel()
    distance_to_non_survivor = pairwise_distances(
        eval_scaled,
        non_survivor_centroid.reshape(1, -1),
    ).ravel()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(fit_scaled)
    clusters = kmeans.predict(eval_scaled)

    passenger_id = (
        X_eval["PassengerId"].to_numpy()
        if "PassengerId" in X_eval.columns
        else np.arange(len(X_eval))
    )
    y_eval_values = np.asarray(y_eval).astype(int)
    predictions = np.asarray(predictions).astype(int)

    return pd.DataFrame(
        {
            "PassengerId": passenger_id,
            "Survived": y_eval_values,
            "Predicted": predictions,
            "SurvivalProbability": probabilities,
            "DistanceToSurvivorCentroid": distance_to_survivor,
            "DistanceToNonSurvivorCentroid": distance_to_non_survivor,
            "SurvivalDistanceDelta": distance_to_non_survivor - distance_to_survivor,
            "ClusterId": clusters,
            "ErrorType": error_type(y_eval_values, predictions),
        }
    )


def write_survival_geometry_validation(
    model,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_df = centroid_distance_frame(
        model,
        X_fit,
        y_fit,
        X_eval,
        y_eval,
        predictions,
        probabilities,
    )
    geometry_df.to_csv(output_path, index=False)
    return geometry_df


def passenger_clusters_frame(
    model,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    probabilities: np.ndarray | None = None,
    n_clusters: int = 4,
) -> pd.DataFrame:
    matrix = preprocessed_matrix(model, X)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        matrix_scaled
    )

    passenger_id = X["PassengerId"].to_numpy() if "PassengerId" in X.columns else np.arange(len(X))
    clusters_df = pd.DataFrame(
        {
            "PassengerId": passenger_id,
            "ClusterId": clusters,
        }
    )
    if y is not None:
        clusters_df["Survived"] = np.asarray(y).astype(int)
        rates = clusters_df.groupby("ClusterId")["Survived"].mean().rename(
            "ClusterSurvivalRate"
        )
        clusters_df = clusters_df.merge(rates, on="ClusterId", how="left")
    if probabilities is not None:
        clusters_df["SurvivalProbability"] = probabilities
    if "ClusterSurvivalRate" not in clusters_df.columns:
        clusters_df["ClusterSurvivalRate"] = np.nan
    return clusters_df


def write_passenger_clusters(
    model,
    X: pd.DataFrame,
    y: pd.Series | None,
    probabilities: np.ndarray | None,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_df = passenger_clusters_frame(model, X, y, probabilities)
    clusters_df.to_csv(output_path, index=False)
    return clusters_df


def _mode(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return None
    return clean.mode().iloc[0]


def _records_for_examples(df: pd.DataFrame) -> list[dict[str, object]]:
    columns = [
        "PassengerId",
        "Name",
        "Sex",
        "Age",
        "Pclass",
        "Title",
        "Fare",
        "Deck",
        "Survived",
        "Predicted",
        "SurvivalProbability",
        "ErrorType",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].to_dict(orient="records")


def prototype_profiles(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    frame = build_feature_frame(X).reset_index(drop=True)
    frame["Survived"] = np.asarray(y).astype(int)
    frame["Predicted"] = np.asarray(predictions).astype(int)
    frame["SurvivalProbability"] = probabilities
    frame["ErrorType"] = error_type(frame["Survived"].to_numpy(), frame["Predicted"].to_numpy())

    numeric_existing = [col for col in NUMERIC_FEATURES if col in frame.columns]
    categorical_existing = [col for col in CATEGORICAL_FEATURES if col in frame.columns]

    def profile_for(value: int) -> dict[str, object]:
        subset = frame[frame["Survived"] == value]
        return {
            "numeric_feature_means": subset[numeric_existing].mean(numeric_only=True).to_dict(),
            "categorical_feature_modes": {
                col: _mode(subset[col]) for col in categorical_existing
            },
        }

    high_survivor = frame.sort_values("SurvivalProbability", ascending=False).head(10)
    high_non_survivor = frame.sort_values("SurvivalProbability", ascending=True).head(10)
    boundary = frame.assign(
        BoundaryDistance=(frame["SurvivalProbability"] - 0.5).abs()
    ).sort_values("BoundaryDistance").head(20)

    return {
        "survivor_prototype": profile_for(1),
        "non_survivor_prototype": profile_for(0),
        "high_confidence_survivor_examples": _records_for_examples(high_survivor),
        "high_confidence_non_survivor_examples": _records_for_examples(high_non_survivor),
        "boundary_cases": _records_for_examples(boundary),
    }


def write_prototype_profiles(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles = prototype_profiles(X, y, predictions, probabilities)
    output_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=True), encoding="utf-8")
    return profiles
