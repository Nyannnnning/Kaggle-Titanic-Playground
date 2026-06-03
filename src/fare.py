from __future__ import annotations

import numpy as np
import pandas as pd


PENCE_PER_POUND = 240
PENCE_PER_SHILLING = 12


def fare_amount_bin(fare: object) -> str:
    if pd.isna(fare):
        return "FareMissing"
    value = float(fare)
    if value <= 0:
        return "FreeOrUnknown"
    if value <= 8:
        return "VeryLowFare"
    if value <= 15:
        return "LowFare"
    if value <= 31:
        return "MidFare"
    if value <= 100:
        return "HighFare"
    return "LuxuryFare"


def family_ticket_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["FamilyKey"].fillna("Unknown").astype(str)
        + "_"
        + frame["TicketNormalized"].fillna("UNKNOWN").astype(str)
    )


def _fare_total_pence(fare: pd.Series) -> pd.Series:
    return (fare.astype(float) * PENCE_PER_POUND).round()


def _travel_party_type(row: pd.Series) -> str:
    if pd.isna(row.get("Fare")):
        return "unknown"

    ticket_size = row.get("TicketGroupSize")
    family_size = row.get("FamilySize")
    if pd.isna(ticket_size) or pd.isna(family_size):
        return "unknown"

    ticket_size = int(ticket_size)
    family_size = int(family_size)

    if ticket_size <= 1 and family_size <= 1:
        return "solo_ticket_solo_family"
    if ticket_size == family_size and family_size > 1:
        return "family_ticket_aligned"
    if ticket_size > family_size and family_size <= 1:
        return "shared_ticket_nonfamily"
    if ticket_size > family_size:
        return "ticket_larger_than_family"
    if ticket_size < family_size:
        return "family_split_across_tickets"
    return "unknown"


def _fare_interpretation(row: pd.Series) -> str:
    party_type = row.get("TravelPartyType")
    if party_type == "solo_ticket_solo_family":
        return "direct_passenger_fare"
    if party_type == "family_ticket_aligned":
        return "family_group_fare"
    if party_type in {"shared_ticket_nonfamily", "ticket_larger_than_family"}:
        return "shared_ticket_fare"
    if party_type == "family_split_across_tickets":
        return "partial_family_fare"
    return "unknown"


def add_fare_object_features(
    df: pd.DataFrame,
    ticket_group_size_map: dict[str, int] | None = None,
    family_ticket_group_size_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Create FareObject features from ticket and family context.

    Fare is treated as a ticket-level price field. The passenger-level signal is
    derived from how that fare relates to ticket group size and family size.
    """
    out = df.copy()

    required_defaults = {
        "Fare": np.nan,
        "FamilySize": np.nan,
        "FamilyKey": "Unknown",
        "TicketNormalized": "UNKNOWN",
        "Pclass": np.nan,
    }
    for col, default in required_defaults.items():
        if col not in out.columns:
            out[col] = default

    out["FamilyTicketKey"] = family_ticket_key(out)

    if "TicketGroupSize" not in out.columns:
        if ticket_group_size_map is None:
            ticket_group_size_map = out["TicketNormalized"].value_counts().to_dict()
        out["TicketGroupSize"] = (
            out["TicketNormalized"].map(ticket_group_size_map).fillna(1)
        )

    if family_ticket_group_size_map is None:
        family_ticket_group_size_map = out["FamilyTicketKey"].value_counts().to_dict()
    out["FamilyTicketGroupSize"] = (
        out["FamilyTicketKey"].map(family_ticket_group_size_map).fillna(1)
    )

    fare = out["Fare"]
    out["FareMissing"] = fare.isna().astype(int)
    out["ZeroFareFlag"] = fare.eq(0).fillna(False).astype(int)
    out["FareTotalPence"] = _fare_total_pence(fare)
    out.loc[fare.isna(), "FareTotalPence"] = np.nan

    total_pence = out["FareTotalPence"]
    out["FarePounds"] = np.floor(total_pence / PENCE_PER_POUND)
    out["FareShillings"] = np.floor(
        (total_pence % PENCE_PER_POUND) / PENCE_PER_SHILLING
    )
    out["FarePence"] = total_pence % PENCE_PER_SHILLING
    for col in ["FarePounds", "FareShillings", "FarePence"]:
        out.loc[fare.isna(), col] = np.nan

    out["FarePerTicketMember"] = fare / out["TicketGroupSize"].replace(0, np.nan)
    out["FarePerFamilyMember"] = fare / out["FamilySize"].replace(0, np.nan)
    out["FarePerFamilyTicketMember"] = fare / out["FamilyTicketGroupSize"].replace(
        0, np.nan
    )

    out["TicketEqualsFamilySize"] = (
        out["TicketGroupSize"].eq(out["FamilySize"]).fillna(False).astype(int)
    )
    out["TicketFamilyMismatch"] = (1 - out["TicketEqualsFamilySize"]).astype(int)
    out["TicketFamilyDelta"] = out["TicketGroupSize"] - out["FamilySize"]

    out["TravelPartyType"] = out.apply(_travel_party_type, axis=1)
    out["FareObjectInterpretation"] = out.apply(_fare_interpretation, axis=1)
    out["FarePerTicketBin"] = out["FarePerTicketMember"].map(fare_amount_bin)
    out["HighRawFareLargeFamilyFlag"] = (
        fare.ge(31).fillna(False) & out["FamilySize"].ge(5).fillna(False)
    ).astype(int)

    out["FareBinPclass"] = (
        out["FareBin"].astype(str) + "_P" + out["Pclass"].astype(str)
        if "FareBin" in out.columns
        else out["Fare"].map(fare_amount_bin).astype(str) + "_P" + out["Pclass"].astype(str)
    )
    out["FarePerTicketBinPclass"] = (
        out["FarePerTicketBin"].astype(str) + "_P" + out["Pclass"].astype(str)
    )

    return out
