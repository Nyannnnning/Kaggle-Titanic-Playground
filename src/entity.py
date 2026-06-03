from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_frame


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    return value


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: to_builtin(value) for key, value in values.items()}


def force_list(force_string: object) -> list[str]:
    if pd.isna(force_string) or not str(force_string).strip():
        return []
    return [part.strip() for part in str(force_string).split("|") if part.strip()]


def passenger_entity(
    raw_row: pd.Series,
    feature_row: pd.Series,
    force_row: pd.Series | None = None,
) -> dict[str, Any]:
    model_evidence: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    forces: dict[str, Any] = {"positive": [], "negative": []}

    if force_row is not None:
        model_evidence = compact_dict(
            {
                "SurvivalProbability": force_row.get("SurvivalProbability"),
                "SurvivalScore": force_row.get("SurvivalScore"),
                "IsBoundaryCase": force_row.get("IsBoundaryCase"),
                "BoundaryDistance": force_row.get("BoundaryDistance"),
                "ErrorType": force_row.get("ErrorType"),
            }
        )
        geometry = compact_dict(
            {
                "DistanceToSurvivorPrototype": force_row.get("DistanceToSurvivorCentroid"),
                "DistanceToNonSurvivorPrototype": force_row.get("DistanceToNonSurvivorCentroid"),
                "SurvivalDelta": force_row.get("SurvivalDistanceDelta"),
                "ClusterId": force_row.get("ClusterId"),
            }
        )
        forces = {
            "positive": force_list(force_row.get("PositiveForces")),
            "negative": force_list(force_row.get("NegativeForces")),
        }

    return {
        "entity_type": "Passenger",
        "entity_id": to_builtin(raw_row.get("PassengerId")),
        "identity": compact_dict(
            {
                "PassengerId": raw_row.get("PassengerId"),
                "Name": raw_row.get("Name"),
                "Sex": feature_row.get("Sex"),
                "Age": feature_row.get("Age"),
                "AgeBin": feature_row.get("AgeBin"),
                "Title": feature_row.get("Title"),
            }
        ),
        "family": compact_dict(
            {
                "SibSp": feature_row.get("SibSp"),
                "Parch": feature_row.get("Parch"),
                "FamilySize": feature_row.get("FamilySize"),
                "FamilySizeBin": feature_row.get("FamilySizeBin"),
                "FamilyGroupSize": feature_row.get("FamilyGroupSize"),
                "IsAlone": feature_row.get("IsAlone"),
            }
        ),
        "ticket": compact_dict(
            {
                "Pclass": feature_row.get("Pclass"),
                "Ticket": raw_row.get("Ticket"),
                "TicketPrefix": feature_row.get("TicketPrefix"),
                "TicketGroupSize": feature_row.get("TicketGroupSize"),
                "Fare": feature_row.get("Fare"),
                "FareBin": feature_row.get("FareBin"),
                "FarePerFamilyMember": feature_row.get("FarePerFamilyMember"),
                "Embarked": feature_row.get("Embarked"),
            }
        ),
        "interactions": compact_dict(
            {
                "SexPclass": feature_row.get("SexPclass"),
                "TitlePclass": feature_row.get("TitlePclass"),
                "AgeBinSex": feature_row.get("AgeBinSex"),
                "FareBinPclass": feature_row.get("FareBinPclass"),
                "PclassAgeInteraction": feature_row.get("PclassAgeInteraction"),
                "PclassFareInteraction": feature_row.get("PclassFareInteraction"),
                "IsChild": feature_row.get("IsChild"),
                "IsAdultMale": feature_row.get("IsAdultMale"),
                "IsMother": feature_row.get("IsMother"),
                "IsFirstClassFemale": feature_row.get("IsFirstClassFemale"),
                "IsThirdClassMale": feature_row.get("IsThirdClassMale"),
            }
        ),
        "spatial": compact_dict(
            {
                "Cabin": feature_row.get("Cabin"),
                "CabinKnown": feature_row.get("CabinKnown"),
                "Deck": feature_row.get("Deck"),
                "DeckOrdinal": feature_row.get("DeckOrdinal"),
                "ApproxVerticalDistanceToBoatDeck": feature_row.get(
                    "ApproxVerticalDistanceToBoatDeck"
                ),
                "SpatialAccessTier": feature_row.get("SpatialAccessTier"),
                "ClassDeckConsistency": feature_row.get("ClassDeckConsistency"),
                "CabinZone": feature_row.get("CabinZone"),
                "ClassArea": feature_row.get("ClassArea"),
                "LifeboatAccessScore": feature_row.get("LifeboatAccessScore"),
                "StaircaseAccessScore": feature_row.get("StaircaseAccessScore"),
            }
        ),
        "language_origin": compact_dict(
            {
                "InferredOriginRegion": feature_row.get("InferredOriginRegion"),
                "InferredPrimaryLanguage": feature_row.get("InferredPrimaryLanguage"),
                "EnglishComprehensionProxy": feature_row.get("EnglishComprehensionProxy"),
                "LanguageBarrierRisk": feature_row.get("LanguageBarrierRisk"),
                "OriginInferenceConfidence": feature_row.get("OriginInferenceConfidence"),
            }
        ),
        "model_evidence": model_evidence,
        "geometry": geometry,
        "forces": forces,
    }


def passenger_entities(
    raw_df: pd.DataFrame,
    force_report: pd.DataFrame | None = None,
    feature_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    features = (
        feature_df.reset_index(drop=True)
        if feature_df is not None
        else build_feature_frame(raw_df).reset_index(drop=True)
    )
    force_by_id: dict[Any, pd.Series] = {}
    if force_report is not None and not force_report.empty:
        force_by_id = {
            row["PassengerId"]: row
            for _, row in force_report.iterrows()
        }

    entities = []
    for idx, raw_row in raw_df.reset_index(drop=True).iterrows():
        passenger_id = raw_row.get("PassengerId")
        entities.append(
            passenger_entity(
                raw_row,
                features.iloc[idx],
                force_by_id.get(passenger_id),
            )
        )
    return entities
