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
                "Title": feature_row.get("Title"),
            }
        ),
        "name": compact_dict(
            {
                "FormalName": feature_row.get("FormalName"),
                "ParentheticalName": feature_row.get("ParentheticalName"),
                "HasParentheticalName": feature_row.get("HasParentheticalName"),
                "FormalSurname": feature_row.get("FormalSurname"),
                "FormalTitle": feature_row.get("FormalTitle"),
                "FormalGivenNames": feature_row.get("FormalGivenNames"),
                "FormalGivenFirstToken": feature_row.get("FormalGivenFirstToken"),
                "FormalGivenTokenCount": feature_row.get("FormalGivenTokenCount"),
                "FormalGivenTokenCountBin": feature_row.get(
                    "FormalGivenTokenCountBin"
                ),
                "FormalNameHasInitial": feature_row.get("FormalNameHasInitial"),
                "FormalNameHasQuote": feature_row.get("FormalNameHasQuote"),
                "FormalSurnameShape": feature_row.get("FormalSurnameShape"),
                "FormalNamePattern": feature_row.get("FormalNamePattern"),
                "FormalNameInterpretation": feature_row.get(
                    "FormalNameInterpretation"
                ),
            }
        ),
        "age": compact_dict(
            {
                "Age": feature_row.get("Age"),
                "AgeKnown": feature_row.get("AgeKnown"),
                "AgeMissing": feature_row.get("AgeMissing"),
                "AgeBin": feature_row.get("AgeBin"),
                "AgeZWithinPclass": feature_row.get("AgeZWithinPclass"),
                "AgeQuartileWithinPclass": feature_row.get("AgeQuartileWithinPclass"),
                "AgeRelativeBucketWithinPclass": feature_row.get(
                    "AgeRelativeBucketWithinPclass"
                ),
                "AgeRelativeBucketWithinPclassSex": feature_row.get(
                    "AgeRelativeBucketWithinPclassSex"
                ),
                "AgeMissingPatternByPclassTitle": feature_row.get(
                    "AgeMissingPatternByPclassTitle"
                ),
                "IsYoungWithinPclass": feature_row.get("IsYoungWithinPclass"),
                "IsOldWithinPclass": feature_row.get("IsOldWithinPclass"),
                "IsOldMaleWithinPclass": feature_row.get("IsOldMaleWithinPclass"),
                "IsYoungMasterWithinPclass": feature_row.get(
                    "IsYoungMasterWithinPclass"
                ),
            }
        ),
        "family": compact_dict(
            {
                "SibSp": feature_row.get("SibSp"),
                "Parch": feature_row.get("Parch"),
                "FamilySize": feature_row.get("FamilySize"),
                "FamilySizeBin": feature_row.get("FamilySizeBin"),
                "FamilyGroupSize": feature_row.get("FamilyGroupSize"),
                "FamilyTicketGroupSize": feature_row.get("FamilyTicketGroupSize"),
                "IsAlone": feature_row.get("IsAlone"),
            }
        ),
        "family_group": compact_dict(
            {
                "FamilyKey": feature_row.get("FamilyKey"),
                "FamilyGroupSize": feature_row.get("FamilyGroupSize"),
                "FamilyChildCount": feature_row.get("FamilyChildCount"),
                "FamilyFemaleCount": feature_row.get("FamilyFemaleCount"),
                "FamilyAdultMaleCount": feature_row.get("FamilyAdultMaleCount"),
                "FamilyChildRatio": feature_row.get("FamilyChildRatio"),
                "FamilyFemaleRatio": feature_row.get("FamilyFemaleRatio"),
                "FamilyAdultMaleRatio": feature_row.get("FamilyAdultMaleRatio"),
                "FamilyAccompanyingChildCount": feature_row.get(
                    "FamilyAccompanyingChildCount"
                ),
                "FamilyAccompanyingFemaleCount": feature_row.get(
                    "FamilyAccompanyingFemaleCount"
                ),
                "FamilyAccompanyingAdultMaleCount": feature_row.get(
                    "FamilyAccompanyingAdultMaleCount"
                ),
                "FamilyChildFemalePattern": feature_row.get(
                    "FamilyChildFemalePattern"
                ),
            }
        ),
        "family_ticket_group": compact_dict(
            {
                "FamilyTicketKey": feature_row.get("FamilyTicketKey"),
                "FamilyTicketGroupSize": feature_row.get("FamilyTicketGroupSize"),
                "FamilyTicketChildCount": feature_row.get("FamilyTicketChildCount"),
                "FamilyTicketFemaleCount": feature_row.get("FamilyTicketFemaleCount"),
                "FamilyTicketAdultMaleCount": feature_row.get(
                    "FamilyTicketAdultMaleCount"
                ),
                "FamilyTicketChildRatio": feature_row.get("FamilyTicketChildRatio"),
                "FamilyTicketFemaleRatio": feature_row.get("FamilyTicketFemaleRatio"),
                "FamilyTicketAdultMaleRatio": feature_row.get(
                    "FamilyTicketAdultMaleRatio"
                ),
                "FamilyTicketAccompanyingChildCount": feature_row.get(
                    "FamilyTicketAccompanyingChildCount"
                ),
                "FamilyTicketAccompanyingFemaleCount": feature_row.get(
                    "FamilyTicketAccompanyingFemaleCount"
                ),
                "FamilyTicketAccompanyingAdultMaleCount": feature_row.get(
                    "FamilyTicketAccompanyingAdultMaleCount"
                ),
                "FamilyTicketChildFemalePattern": feature_row.get(
                    "FamilyTicketChildFemalePattern"
                ),
            }
        ),
        "companion_group": compact_dict(
            {
                "SelfGroupRole": feature_row.get("SelfGroupRole"),
                "PrimaryCompanionScope": feature_row.get("PrimaryCompanionScope"),
                "CompanionGroupSize": feature_row.get("CompanionGroupSize"),
                "CompanionChildCount": feature_row.get("CompanionChildCount"),
                "CompanionFemaleCount": feature_row.get("CompanionFemaleCount"),
                "CompanionAdultMaleCount": feature_row.get("CompanionAdultMaleCount"),
                "CompanionAccompanyingChildCount": feature_row.get(
                    "CompanionAccompanyingChildCount"
                ),
                "CompanionAccompanyingFemaleCount": feature_row.get(
                    "CompanionAccompanyingFemaleCount"
                ),
                "CompanionAccompanyingAdultMaleCount": feature_row.get(
                    "CompanionAccompanyingAdultMaleCount"
                ),
                "PrimaryGroupChildFemalePattern": feature_row.get(
                    "PrimaryGroupChildFemalePattern"
                ),
                "ChildFemaleCompanionPattern": feature_row.get(
                    "ChildFemaleCompanionPattern"
                ),
            }
        ),
        "group_fate": compact_dict(
            {
                "TicketFateKnown": feature_row.get("TicketFateKnown"),
                "TicketFateKnownCount": feature_row.get("TicketFateKnownCount"),
                "TicketFateSurvivalRate": feature_row.get("TicketFateSurvivalRate"),
                "TicketFateSignal": feature_row.get("TicketFateSignal"),
                "FamilyFateKnown": feature_row.get("FamilyFateKnown"),
                "FamilyFateKnownCount": feature_row.get("FamilyFateKnownCount"),
                "FamilyFateSurvivalRate": feature_row.get("FamilyFateSurvivalRate"),
                "FamilyFateSignal": feature_row.get("FamilyFateSignal"),
                "FamilyTicketFateKnown": feature_row.get("FamilyTicketFateKnown"),
                "FamilyTicketFateKnownCount": feature_row.get(
                    "FamilyTicketFateKnownCount"
                ),
                "FamilyTicketFateSurvivalRate": feature_row.get(
                    "FamilyTicketFateSurvivalRate"
                ),
                "FamilyTicketFateSignal": feature_row.get(
                    "FamilyTicketFateSignal"
                ),
                "PrimaryFateScope": feature_row.get("PrimaryFateScope"),
                "PrimaryFateKnownCount": feature_row.get("PrimaryFateKnownCount"),
                "PrimaryFateSurvivalRate": feature_row.get(
                    "PrimaryFateSurvivalRate"
                ),
                "PrimaryFateSignal": feature_row.get("PrimaryFateSignal"),
            }
        ),
        "ticket": compact_dict(
            {
                "Pclass": feature_row.get("Pclass"),
                "Ticket": raw_row.get("Ticket"),
                "TicketPrefix": feature_row.get("TicketPrefix"),
                "TicketNumber": feature_row.get("TicketNumber"),
                "TicketNumberBand": feature_row.get("TicketNumberBand"),
                "TicketGroupSize": feature_row.get("TicketGroupSize"),
                "Embarked": feature_row.get("Embarked"),
                "TicketPrefixEmbarked": feature_row.get("TicketPrefixEmbarked"),
                "TicketNumberBandEmbarked": feature_row.get(
                    "TicketNumberBandEmbarked"
                ),
                "TicketPrefixPclassEmbarked": feature_row.get(
                    "TicketPrefixPclassEmbarked"
                ),
            }
        ),
        "ticket_group": compact_dict(
            {
                "TicketGroupSize": feature_row.get("TicketGroupSize"),
                "TicketChildCount": feature_row.get("TicketChildCount"),
                "TicketFemaleCount": feature_row.get("TicketFemaleCount"),
                "TicketMaleCount": feature_row.get("TicketMaleCount"),
                "TicketAdultMaleCount": feature_row.get("TicketAdultMaleCount"),
                "TicketMasterCount": feature_row.get("TicketMasterCount"),
                "TicketDistinctFamilyCount": feature_row.get("TicketDistinctFamilyCount"),
                "TicketMaxFamilySize": feature_row.get("TicketMaxFamilySize"),
                "TicketChildRatio": feature_row.get("TicketChildRatio"),
                "TicketFemaleRatio": feature_row.get("TicketFemaleRatio"),
                "TicketAdultMaleRatio": feature_row.get("TicketAdultMaleRatio"),
                "TicketHasChild": feature_row.get("TicketHasChild"),
                "TicketHasFemale": feature_row.get("TicketHasFemale"),
                "TicketHasAdultMale": feature_row.get("TicketHasAdultMale"),
                "TicketHasFamily": feature_row.get("TicketHasFamily"),
                "TicketIsMixedSex": feature_row.get("TicketIsMixedSex"),
                "TicketCompositionType": feature_row.get("TicketCompositionType"),
                "TicketFamilyPattern": feature_row.get("TicketFamilyPattern"),
                "TicketChildFemalePattern": feature_row.get("TicketChildFemalePattern"),
                "TravelPartyType": feature_row.get("TravelPartyType"),
                "TicketEqualsFamilySize": feature_row.get("TicketEqualsFamilySize"),
                "TicketFamilyMismatch": feature_row.get("TicketFamilyMismatch"),
                "TicketFamilyDelta": feature_row.get("TicketFamilyDelta"),
            }
        ),
        "fare": compact_dict(
            {
                "Fare": feature_row.get("Fare"),
                "FareMissing": feature_row.get("FareMissing"),
                "ZeroFareFlag": feature_row.get("ZeroFareFlag"),
                "FareTotalPence": feature_row.get("FareTotalPence"),
                "FarePounds": feature_row.get("FarePounds"),
                "FareShillings": feature_row.get("FareShillings"),
                "FarePence": feature_row.get("FarePence"),
                "FareBin": feature_row.get("FareBin"),
                "FarePerTicketMember": feature_row.get("FarePerTicketMember"),
                "FarePerFamilyMember": feature_row.get("FarePerFamilyMember"),
                "FarePerFamilyTicketMember": feature_row.get(
                    "FarePerFamilyTicketMember"
                ),
                "FarePerTicketBin": feature_row.get("FarePerTicketBin"),
                "FareObjectInterpretation": feature_row.get("FareObjectInterpretation"),
                "HighRawFareLargeFamilyFlag": feature_row.get(
                    "HighRawFareLargeFamilyFlag"
                ),
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
