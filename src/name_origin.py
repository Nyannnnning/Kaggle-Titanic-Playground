from __future__ import annotations

import re

import pandas as pd


NAME_ORIGIN_COLUMNS = [
    "PassengerId",
    "InferredOriginRegion",
    "InferredPrimaryLanguage",
    "EnglishComprehensionProxy",
    "LanguageBarrierRisk",
    "OriginInferenceConfidence",
]


def _surname(name: object) -> str:
    if pd.isna(name):
        return ""
    return str(name).split(",", maxsplit=1)[0].strip().lower()


def _title(name: object) -> str:
    if pd.isna(name):
        return ""
    match = re.search(r" ([A-Za-z]+)\.", str(name))
    return match.group(1) if match else ""


def infer_name_origin(row: pd.Series) -> dict[str, object]:
    surname = _surname(row.get("Name"))
    title = _title(row.get("Name"))
    embarked = row.get("Embarked")

    region = "unknown"
    language = "unknown"
    english_proxy = "uncertain"
    barrier_risk = "medium"
    confidence = 0.2

    if surname.startswith(("mc", "mac", "o'")) or embarked in {"S", "Q"}:
        region = "british_irish_or_transatlantic"
        language = "english"
        english_proxy = "likely_high"
        barrier_risk = "low"
        confidence = 0.45

    if surname.endswith(("sson", "sen", "strom", "berg", "lund")):
        region = "northern_europe"
        language = "scandinavian_or_germanic"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.55

    if surname.endswith(("ski", "sky", "wicz", "vich")):
        region = "eastern_europe"
        language = "slavic"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.55

    if surname.endswith(("ez", "es")) or title in {"Don", "Dona"}:
        region = "iberian_or_latin_america"
        language = "spanish_or_portuguese"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.55

    if surname.endswith(("ini", "elli", "etti", "otti", "ari")):
        region = "italian"
        language = "italian"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.55

    if title in {"Mme", "Mlle"} or surname.startswith(("de ", "du ", "le ", "la ")):
        region = "western_europe"
        language = "french_or_western_european"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.55

    if embarked == "C" and confidence < 0.5:
        region = "continental_europe_or_transatlantic"
        language = "mixed_european"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.4

    if not surname:
        region = "unknown"
        language = "unknown"
        english_proxy = "uncertain"
        barrier_risk = "medium"
        confidence = 0.0

    return {
        "InferredOriginRegion": region,
        "InferredPrimaryLanguage": language,
        "EnglishComprehensionProxy": english_proxy,
        "LanguageBarrierRisk": barrier_risk,
        "OriginInferenceConfidence": confidence,
    }


def add_name_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inferred = pd.DataFrame([infer_name_origin(row) for _, row in out.iterrows()])
    if "PassengerId" in out.columns:
        inferred.insert(0, "PassengerId", out["PassengerId"].to_numpy())
    else:
        inferred.insert(0, "PassengerId", range(len(out)))
    return inferred[NAME_ORIGIN_COLUMNS]
