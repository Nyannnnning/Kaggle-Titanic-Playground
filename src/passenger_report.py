from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .entity import passenger_entities
from .vectorizer import (
    feature_frame,
    logistic_contribution_frame,
    passenger_force_summary_frame,
    passenger_vector_features_frame,
    passenger_vector_frame,
)


def write_passenger_reports(
    model,
    X: pd.DataFrame,
    probabilities: np.ndarray,
    geometry: pd.DataFrame | None,
    reports_dir: Path,
    shap_force_report: pd.DataFrame | None = None,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = reports_dir / "passenger_vectors.csv"
    vector_features_path = reports_dir / "passenger_vector_features.csv"
    contributions_path = reports_dir / "passenger_feature_contributions.csv"
    force_report_path = reports_dir / "passenger_force_report.csv"
    profiles_dir = reports_dir / "passenger_profiles"

    vectors = passenger_vector_frame(model, X)
    vectors.to_csv(vectors_path, index=False)

    vector_features = passenger_vector_features_frame(model, X)
    vector_features.to_csv(vector_features_path, index=False)

    contributions = logistic_contribution_frame(model, X)
    contributions.to_csv(contributions_path, index=False)

    force_report = passenger_force_summary_frame(model, X, probabilities, geometry)
    if shap_force_report is not None and not shap_force_report.empty:
        shap_cols = [
            "PassengerId",
            "SHAPPositiveForces",
            "SHAPNegativeForces",
        ]
        shap_cols = [col for col in shap_cols if col in shap_force_report.columns]
        force_report = force_report.merge(
            shap_force_report[shap_cols],
            on="PassengerId",
            how="left",
        )
        if "SHAPPositiveForces" in force_report.columns:
            force_report["PositiveForces"] = force_report["SHAPPositiveForces"].fillna(
                force_report["PositiveForces"]
            )
            force_report = force_report.drop(columns=["SHAPPositiveForces"])
        if "SHAPNegativeForces" in force_report.columns:
            force_report["NegativeForces"] = force_report["SHAPNegativeForces"].fillna(
                force_report["NegativeForces"]
            )
            force_report = force_report.drop(columns=["SHAPNegativeForces"])
    force_report.to_csv(force_report_path, index=False)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    model_feature_frame = feature_frame(model, X)
    for entity in passenger_entities(X, force_report, feature_df=model_feature_frame):
        passenger_id = entity["entity_id"]
        profile_path = profiles_dir / f"passenger_{passenger_id}.json"
        profile_path.write_text(
            json.dumps(entity, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return {
        "passenger_vectors": vectors_path,
        "passenger_vector_features": vector_features_path,
        "passenger_feature_contributions": contributions_path,
        "passenger_force_report": force_report_path,
        "passenger_profiles": profiles_dir,
    }
