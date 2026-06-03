from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .model import get_classifier, get_feature_names, survival_scores


def feature_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    return model.named_steps["feature_builder"].transform(X).reset_index(drop=True)


def transformed_matrix(model, X: pd.DataFrame) -> np.ndarray:
    features = feature_frame(model, X)
    matrix = model.named_steps["preprocessing"].transform(features)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def passenger_vector_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    matrix = transformed_matrix(model, X)
    feature_names = get_feature_names(model)
    vectors = pd.DataFrame(matrix, columns=feature_names)
    if "PassengerId" in X.columns:
        vectors.insert(0, "PassengerId", X["PassengerId"].to_numpy())
    else:
        vectors.insert(0, "PassengerId", np.arange(len(vectors)))
    return vectors


def passenger_vector_features_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    vectors = passenger_vector_frame(model, X)
    long_df = vectors.melt(
        id_vars=["PassengerId"],
        var_name="VectorFeature",
        value_name="VectorValue",
    )
    return long_df[long_df["VectorValue"].abs() > 1e-12].reset_index(drop=True)


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"


def _class_label(value: object) -> str:
    mapping = {1: "First Class", 2: "Second Class", 3: "Third Class"}
    try:
        return mapping.get(int(value), f"Pclass {_format_number(value)}")
    except (TypeError, ValueError):
        return "Unknown Class"


def _cat_value(feature_name: str, prefix: str) -> str | None:
    marker = f"cat__{prefix}_"
    if not feature_name.startswith(marker):
        return None
    return feature_name.removeprefix(marker)


def human_force_label(feature_name: str, row: pd.Series) -> str:
    if feature_name == "num__Pclass":
        return _class_label(row.get("Pclass"))
    if feature_name == "num__Age":
        return f"Age {_format_number(row.get('Age'))}"
    if feature_name == "num__SibSp":
        return f"Siblings/Spouses {_format_number(row.get('SibSp'))}"
    if feature_name == "num__Parch":
        return f"Parents/Children {_format_number(row.get('Parch'))}"
    if feature_name == "num__Fare":
        return f"Fare {_format_number(row.get('Fare'))}"
    if feature_name == "num__FamilySize":
        return f"Family Size {_format_number(row.get('FamilySize'))}"
    if feature_name == "num__IsAlone":
        return "Traveling Alone" if row.get("IsAlone") == 1 else "Traveling With Family"
    if feature_name == "num__FarePerFamilyMember":
        return f"Fare Per Family Member {_format_number(row.get('FarePerFamilyMember'))}"
    if feature_name == "num__TicketGroupSize":
        return f"Ticket Group Size {_format_number(row.get('TicketGroupSize'))}"
    if feature_name == "num__FamilyGroupSize":
        return f"Family Group Size {_format_number(row.get('FamilyGroupSize'))}"
    if feature_name == "num__PclassAgeInteraction":
        return "Pclass x Age"
    if feature_name == "num__PclassFareInteraction":
        return "Pclass x Fare"
    if feature_name == "num__IsChild":
        return "Child" if row.get("IsChild") == 1 else "Not Child"
    if feature_name == "num__IsAdultMale":
        return "Adult Male" if row.get("IsAdultMale") == 1 else "Not Adult Male"
    if feature_name == "num__IsMother":
        return "Mother Proxy" if row.get("IsMother") == 1 else "Not Mother Proxy"
    if feature_name == "num__IsFirstClassFemale":
        return (
            "First Class Female"
            if row.get("IsFirstClassFemale") == 1
            else "Not First Class Female"
        )
    if feature_name == "num__IsThirdClassMale":
        return "Third Class Male" if row.get("IsThirdClassMale") == 1 else "Not Third Class Male"
    if feature_name == "num__DeckOrdinal":
        return f"Deck Vertical Proxy {_format_number(row.get('DeckOrdinal'))}"
    if feature_name == "num__ApproxVerticalDistanceToBoatDeck":
        return "Approx Distance To Boat Deck"
    if feature_name == "num__LifeboatAccessScore":
        return "Lifeboat Access Proxy"
    if feature_name == "num__StaircaseAccessScore":
        return "Staircase Access Proxy"
    if feature_name == "num__CabinKnown":
        return "Cabin Known"
    if feature_name == "num__CabinMissing":
        return "Cabin Missing"
    if feature_name == "num__DeckKnown":
        return "Deck Known"
    if feature_name == "num__DeckMissing":
        return "Deck Missing"
    if feature_name == "num__OriginInferenceConfidence":
        return "Origin Inference Confidence"

    sex = _cat_value(feature_name, "Sex")
    if sex:
        return sex.capitalize()

    embarked = _cat_value(feature_name, "Embarked")
    if embarked:
        ports = {"S": "Embarked Southampton", "C": "Embarked Cherbourg", "Q": "Embarked Queenstown"}
        return ports.get(embarked, f"Embarked {embarked}")

    title = _cat_value(feature_name, "Title")
    if title:
        return f"Title {title}"

    age_bucket = _cat_value(feature_name, "AgeBin")
    if age_bucket:
        return f"Age Bin {age_bucket}"

    fare_bucket = _cat_value(feature_name, "FareBin")
    if fare_bucket:
        return f"Fare Bin {fare_bucket}"

    family_bucket = _cat_value(feature_name, "FamilySizeBin")
    if family_bucket:
        return f"Family Size Bin {family_bucket}"

    ticket_prefix = _cat_value(feature_name, "TicketPrefix")
    if ticket_prefix:
        return f"Ticket Prefix {ticket_prefix}"

    sex_pclass = _cat_value(feature_name, "SexPclass")
    if sex_pclass:
        return f"Sex x Class {sex_pclass}"

    title_pclass = _cat_value(feature_name, "TitlePclass")
    if title_pclass:
        return f"Title x Class {title_pclass}"

    age_bin_sex = _cat_value(feature_name, "AgeBinSex")
    if age_bin_sex:
        return f"Age Bin x Sex {age_bin_sex}"

    fare_bin_pclass = _cat_value(feature_name, "FareBinPclass")
    if fare_bin_pclass:
        return f"Fare Bin x Class {fare_bin_pclass}"

    deck = _cat_value(feature_name, "Deck")
    if deck:
        return f"Deck {deck}"

    tier = _cat_value(feature_name, "SpatialAccessTier")
    if tier:
        labels = {
            "near_boat_deck": "Near Boat Deck",
            "upper_deck": "Upper Deck",
            "middle_deck": "Middle Deck",
            "lower_deck": "Lower Deck",
            "unknown": "Unknown Spatial Tier",
        }
        return labels.get(tier, tier.replace("_", " ").title())

    consistency = _cat_value(feature_name, "ClassDeckConsistency")
    if consistency:
        return consistency.replace("_", " ").title()

    cabin_zone = _cat_value(feature_name, "CabinZone")
    if cabin_zone:
        return f"Cabin Zone {cabin_zone.replace('_', ' ').title()}"

    class_area = _cat_value(feature_name, "ClassArea")
    if class_area:
        return f"Class Area {class_area.replace('_', ' ').title()}"

    origin = _cat_value(feature_name, "InferredOriginRegion")
    if origin:
        return f"Origin Region {origin.replace('_', ' ').title()}"

    language = _cat_value(feature_name, "InferredPrimaryLanguage")
    if language:
        return f"Primary Language {language.replace('_', ' ').title()}"

    english = _cat_value(feature_name, "EnglishComprehensionProxy")
    if english:
        return f"English Proxy {english.replace('_', ' ').title()}"

    barrier = _cat_value(feature_name, "LanguageBarrierRisk")
    if barrier:
        return f"Language Barrier {barrier.replace('_', ' ').title()}"

    return re.sub(r"^(num|cat)__", "", feature_name).replace("_", " ")


def logistic_contribution_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    classifier = get_classifier(model)
    if not isinstance(classifier, LogisticRegression):
        return pd.DataFrame(
            columns=[
                "PassengerId",
                "VectorFeature",
                "VectorValue",
                "Coefficient",
                "Contribution",
                "Direction",
                "ForceLabel",
            ]
        )

    matrix = transformed_matrix(model, X)
    feature_names = get_feature_names(model)
    coefficients = classifier.coef_[0]
    contributions = matrix * coefficients
    features = feature_frame(model, X)

    rows: list[dict[str, object]] = []
    passenger_ids = (
        X["PassengerId"].to_numpy()
        if "PassengerId" in X.columns
        else np.arange(len(X))
    )
    for passenger_idx, passenger_id in enumerate(passenger_ids):
        feature_row = features.iloc[passenger_idx]
        non_zero = np.where(np.abs(contributions[passenger_idx]) > 1e-12)[0]
        for feature_idx in non_zero:
            contribution = float(contributions[passenger_idx, feature_idx])
            rows.append(
                {
                    "PassengerId": passenger_id,
                    "VectorFeature": feature_names[feature_idx],
                    "VectorValue": float(matrix[passenger_idx, feature_idx]),
                    "Coefficient": float(coefficients[feature_idx]),
                    "Contribution": contribution,
                    "Direction": "positive" if contribution > 0 else "negative",
                    "ForceLabel": human_force_label(feature_names[feature_idx], feature_row),
                }
            )
    return pd.DataFrame(rows)


def _top_force_strings(
    contributions: pd.DataFrame,
    passenger_id: object,
    direction: str,
    top_n: int,
) -> list[str]:
    subset = contributions[
        (contributions["PassengerId"] == passenger_id)
        & (contributions["Direction"] == direction)
    ].copy()
    if subset.empty:
        return []
    subset["AbsContribution"] = subset["Contribution"].abs()
    subset = subset.sort_values("AbsContribution", ascending=False).head(top_n)
    sign = "+" if direction == "positive" else "-"
    return [
        f"{sign} {row.ForceLabel} ({row.Contribution:+.3f})"
        for row in subset.itertuples(index=False)
    ]


def passenger_force_summary_frame(
    model,
    X: pd.DataFrame,
    probabilities: np.ndarray,
    geometry: pd.DataFrame | None,
    top_n: int = 5,
) -> pd.DataFrame:
    passenger_ids = (
        X["PassengerId"].to_numpy()
        if "PassengerId" in X.columns
        else np.arange(len(X))
    )
    scores = survival_scores(model, X)
    contributions = logistic_contribution_frame(model, X)

    summary = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "SurvivalProbability": probabilities,
            "SurvivalScore": scores,
            "BoundaryDistance": np.abs(probabilities - 0.5),
            "IsBoundaryCase": np.abs(probabilities - 0.5) <= 0.1,
        }
    )

    if geometry is not None and not geometry.empty:
        keep_cols = [
            col
            for col in [
                "PassengerId",
                "DistanceToSurvivorCentroid",
                "DistanceToNonSurvivorCentroid",
                "SurvivalDistanceDelta",
                "ClusterId",
                "ErrorType",
            ]
            if col in geometry.columns
        ]
        summary = summary.merge(geometry[keep_cols], on="PassengerId", how="left")

    summary["PositiveForces"] = [
        " | ".join(_top_force_strings(contributions, passenger_id, "positive", top_n))
        for passenger_id in passenger_ids
    ]
    summary["NegativeForces"] = [
        " | ".join(_top_force_strings(contributions, passenger_id, "negative", top_n))
        for passenger_id in passenger_ids
    ]
    return summary
