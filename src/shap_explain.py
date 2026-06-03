from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import get_feature_names
from .vectorizer import feature_frame, human_force_label, transformed_matrix


def _coerce_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def shap_value_matrix(model, X: pd.DataFrame) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is not installed. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    classifier = model.named_steps["classifier"]
    matrix = transformed_matrix(model, X)
    explainer = shap.TreeExplainer(classifier)
    return _coerce_shap_values(explainer.shap_values(matrix))


def shap_force_label(
    feature_name: str,
    feature_row: pd.Series,
    vector_value: float,
) -> str:
    label = human_force_label(feature_name, feature_row)
    if feature_name.startswith("cat__") and abs(vector_value) <= 1e-12:
        return f"Not {label}"
    if feature_name == "num__CabinKnown" and abs(vector_value) <= 1e-12:
        return "Cabin Missing"
    if feature_name == "num__CabinMissing" and abs(vector_value) <= 1e-12:
        return "Cabin Present"
    if feature_name == "num__DeckKnown" and abs(vector_value) <= 1e-12:
        return "Deck Unknown"
    if feature_name == "num__DeckMissing" and abs(vector_value) <= 1e-12:
        return "Deck Present"
    if feature_name == "num__IsAlone" and abs(vector_value) <= 1e-12:
        return "Traveling With Family"
    return label


def shap_long_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    values = shap_value_matrix(model, X)
    matrix = transformed_matrix(model, X)
    feature_names = get_feature_names(model)
    features = feature_frame(model, X)
    passenger_ids = (
        X["PassengerId"].to_numpy()
        if "PassengerId" in X.columns
        else np.arange(len(X))
    )

    rows: list[dict[str, object]] = []
    for passenger_idx, passenger_id in enumerate(passenger_ids):
        feature_row = features.iloc[passenger_idx]
        for feature_idx, feature_name in enumerate(feature_names):
            shap_value = float(values[passenger_idx, feature_idx])
            rows.append(
                {
                    "PassengerId": passenger_id,
                    "VectorFeature": feature_name,
                    "VectorValue": float(matrix[passenger_idx, feature_idx]),
                    "SHAPValue": shap_value,
                    "AbsSHAPValue": abs(shap_value),
                    "Direction": "positive" if shap_value > 0 else "negative",
                    "ForceLabel": shap_force_label(
                        feature_name,
                        feature_row,
                        float(matrix[passenger_idx, feature_idx]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def shap_summary_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    long_df = shap_long_frame(model, X)
    summary = (
        long_df.groupby("VectorFeature", as_index=False)
        .agg(
            MeanAbsSHAP=("AbsSHAPValue", "mean"),
            MeanSHAP=("SHAPValue", "mean"),
            PositiveRate=("SHAPValue", lambda s: float((s > 0).mean())),
            ExampleLabel=("ForceLabel", "first"),
        )
        .sort_values("MeanAbsSHAP", ascending=False)
    )
    summary["TypicalDirection"] = np.select(
        [summary["MeanSHAP"] > 1e-9, summary["MeanSHAP"] < -1e-9],
        ["positive", "negative"],
        default="mixed_or_neutral",
    )
    return summary


def shap_force_summary_frame(
    model,
    X: pd.DataFrame,
    probabilities: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    long_df = shap_long_frame(model, X)
    passenger_ids = (
        X["PassengerId"].to_numpy()
        if "PassengerId" in X.columns
        else np.arange(len(X))
    )

    rows = []
    for passenger_id, probability in zip(passenger_ids, probabilities, strict=False):
        passenger_values = long_df[long_df["PassengerId"] == passenger_id].copy()
        positive = (
            passenger_values[passenger_values["SHAPValue"] > 0]
            .sort_values("AbsSHAPValue", ascending=False)
            .head(top_n)
        )
        negative = (
            passenger_values[passenger_values["SHAPValue"] < 0]
            .sort_values("AbsSHAPValue", ascending=False)
            .head(top_n)
        )
        rows.append(
            {
                "PassengerId": passenger_id,
                "SurvivalProbability": probability,
                "SHAPPositiveForces": " | ".join(
                    f"+ {row.ForceLabel} ({row.SHAPValue:+.3f})"
                    for row in positive.itertuples(index=False)
                ),
                "SHAPNegativeForces": " | ".join(
                    f"- {row.ForceLabel} ({row.SHAPValue:+.3f})"
                    for row in negative.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_shap_reports(
    model,
    X: pd.DataFrame,
    probabilities: np.ndarray,
    reports_dir: Path,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "shap_summary.csv"
    values_path = reports_dir / "passenger_shap_values.csv"
    force_path = reports_dir / "passenger_shap_force_report.csv"

    long_df = shap_long_frame(model, X)
    long_df.to_csv(values_path, index=False)

    summary = (
        long_df.groupby("VectorFeature", as_index=False)
        .agg(
            MeanAbsSHAP=("AbsSHAPValue", "mean"),
            MeanSHAP=("SHAPValue", "mean"),
            PositiveRate=("SHAPValue", lambda s: float((s > 0).mean())),
            ExampleLabel=("ForceLabel", "first"),
        )
        .sort_values("MeanAbsSHAP", ascending=False)
    )
    summary["TypicalDirection"] = np.select(
        [summary["MeanSHAP"] > 1e-9, summary["MeanSHAP"] < -1e-9],
        ["positive", "negative"],
        default="mixed_or_neutral",
    )
    summary.to_csv(summary_path, index=False)

    shap_force_summary_frame(model, X, probabilities).to_csv(force_path, index=False)

    return {
        "shap_summary": summary_path,
        "passenger_shap_values": values_path,
        "passenger_shap_force_report": force_path,
    }
