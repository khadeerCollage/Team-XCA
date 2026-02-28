"""
schemas/vendor_risk.py
"""

from pydantic import BaseModel, Field
from typing import Optional


class VendorRiskScore(BaseModel):
    gstin:              str
    name:               str
    state:              Optional[str] = None

    # Raw score components (0 to 1 each)
    mismatch_rate:         float = Field(ge=0, le=1)  # % invoices mismatched
    avg_filing_delay:      float = Field(ge=0, le=1)  # normalized delay days
    itc_overclaim_ratio:   float = Field(ge=0, le=1)  # claimed vs eligible
    circular_txn_flag:     float = Field(ge=0, le=1)  # 0 or 1
    new_vendor_flag:       float = Field(ge=0, le=1)  # < 6 months = 1

    # Final weighted score
    risk_score:            float = Field(ge=0, le=1)  # 0=safe 1=fraud
    risk_level:            str                         # LOW / MEDIUM / HIGH

    # Stats
    total_transactions:    int   = 0
    total_itc_claimed:     float = 0.0
    total_itc_eligible:    float = 0.0
    total_itc_at_risk:     float = 0.0

    # Decision
    recommendation:        str   = ""   # "AUTO-APPROVE" / "REVIEW" / "BLOCK"
