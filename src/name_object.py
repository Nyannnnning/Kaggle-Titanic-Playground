from __future__ import annotations

import re

import pandas as pd


TITLE_REPLACEMENTS = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def strip_parenthetical_name(name: object) -> str:
    if pd.isna(name):
        return ""
    formal = re.sub(r"\s*\([^)]*\)", "", str(name)).strip()
    return re.sub(r"\s+", " ", formal)


def extract_parenthetical_name(name: object) -> str:
    if pd.isna(name):
        return ""
    matches = re.findall(r"\(([^)]*)\)", str(name))
    return " | ".join(match.strip() for match in matches if match.strip())


def surname_from_formal_name(formal_name: object) -> str:
    if pd.isna(formal_name):
        return "Unknown"
    surname = str(formal_name).split(",", maxsplit=1)[0].strip()
    return surname if surname else "Unknown"


def formal_remainder(formal_name: object) -> str:
    if pd.isna(formal_name):
        return ""
    pieces = str(formal_name).split(",", maxsplit=1)
    if len(pieces) < 2:
        return ""
    return pieces[1].strip()


def formal_title(remainder: object) -> str:
    if pd.isna(remainder):
        return "Unknown"
    match = re.match(r"([A-Za-z]+)\.", str(remainder).strip())
    if not match:
        return "Unknown"
    title = match.group(1)
    return TITLE_REPLACEMENTS.get(title, title)


def formal_given_names(remainder: object) -> str:
    if pd.isna(remainder):
        return ""
    given = re.sub(r"^[A-Za-z]+\.\s*", "", str(remainder).strip())
    return given.strip()


def _tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    clean = re.sub(r'["“”]', "", str(value))
    return [token for token in re.split(r"\s+", clean.strip()) if token]


def token_count_bin(count: object) -> str:
    if pd.isna(count):
        return "Unknown"
    value = int(count)
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def surname_shape(surname: object) -> str:
    if pd.isna(surname) or not str(surname).strip():
        return "unknown"
    value = str(surname).strip()
    has_hyphen = "-" in value
    has_apostrophe = "'" in value or "’" in value
    has_space = bool(re.search(r"\s+", value))
    if has_hyphen:
        return "hyphenated"
    if has_apostrophe:
        return "apostrophe"
    if has_space:
        return "multiword"
    return "single_token"


def formal_name_pattern(row: pd.Series) -> str:
    title = row.get("FormalTitle", "Unknown")
    token_count = int(row.get("FormalGivenTokenCount", 0))
    has_initial = bool(row.get("FormalNameHasInitial", 0))

    if title == "Mrs" and token_count <= 0:
        return "mrs_without_spouse_formal_given"
    if title == "Mrs":
        return "mrs_spouse_style_formal_name"
    if title in {"Mr", "Miss", "Master"} and has_initial:
        return f"{title.lower()}_with_initial"
    if title in {"Mr", "Miss", "Master"} and token_count >= 2:
        return f"{title.lower()}_multi_given"
    if title in {"Mr", "Miss", "Master"}:
        return f"{title.lower()}_single_given"
    return "rare_or_unknown_title"


def formal_name_interpretation(row: pd.Series) -> str:
    title = row.get("FormalTitle", "Unknown")
    if title == "Mrs":
        return "married_female_formal_spouse_name"
    if title in {"Mr", "Miss", "Master"}:
        return "formal_personal_name"
    return "rare_title_formal_name"


def add_name_object_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Name" not in out.columns:
        out["Name"] = ""

    out["FormalName"] = out["Name"].map(strip_parenthetical_name)
    out["ParentheticalName"] = out["Name"].map(extract_parenthetical_name)
    out["HasParentheticalName"] = out["ParentheticalName"].astype(bool).astype(int)
    out["FormalSurname"] = out["FormalName"].map(surname_from_formal_name)
    out["FormalRemainder"] = out["FormalName"].map(formal_remainder)
    out["FormalTitle"] = out["FormalRemainder"].map(formal_title)
    out["FormalGivenNames"] = out["FormalRemainder"].map(formal_given_names)
    out["FormalGivenFirstToken"] = out["FormalGivenNames"].map(
        lambda value: _tokens(value)[0] if _tokens(value) else "Unknown"
    )
    out["FormalGivenTokenCount"] = out["FormalGivenNames"].map(lambda value: len(_tokens(value)))
    out["FormalGivenTokenCountBin"] = out["FormalGivenTokenCount"].map(token_count_bin)
    out["FormalNameHasInitial"] = out["FormalGivenNames"].map(
        lambda value: int(any(re.fullmatch(r"[A-Z]\.?", token) for token in _tokens(value)))
    )
    out["FormalNameHasQuote"] = out["FormalName"].str.contains('"', regex=False).fillna(False).astype(int)
    out["FormalSurnameShape"] = out["FormalSurname"].map(surname_shape)
    out["FormalNamePattern"] = out.apply(formal_name_pattern, axis=1)
    out["FormalNameInterpretation"] = out.apply(formal_name_interpretation, axis=1)

    return out
