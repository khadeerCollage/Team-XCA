"""
schemas/taxpayer.py
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class BusinessType(str, Enum):
    MANUFACTURER     = "Manufacturer"
    TRADER           = "Trader"
    SERVICE_PROVIDER = "Service Provider"
    COMPOSITE        = "Composite"


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Request: Create a Taxpayer ─────────────────────────────────
class TaxpayerCreate(BaseModel):
    gstin:             str          = Field(..., min_length=15, max_length=15,
                                           description="15-char GSTIN eg. 27AABCT1332L1ZX")
    name:              str          = Field(..., min_length=2)
    state:             str
    turnover:          float        = Field(..., gt=0, description="Annual turnover in INR")
    business_type:     BusinessType = BusinessType.TRADER
    registration_date: Optional[str] = None
    risk_score:        Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("gstin")
    @classmethod
    def gstin_must_be_uppercase(cls, v: str) -> str:
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "gstin": "27AABCT1332L1ZX",
                "name": "Vikram Fabrics Pvt Ltd",
                "state": "Maharashtra",
                "turnover": 5000000.0,
                "business_type": "Manufacturer",
                "registration_date": "2020-01-15",
                "risk_score": 0.12
            }
        }
    }


# ── Response: Taxpayer from DB ─────────────────────────────────
class TaxpayerResponse(BaseModel):
    gstin:             str
    name:              str
    state:             str
    turnover:          float
    business_type:     str
    registration_date: Optional[str]   = None
    risk_score:        float            = 0.0
    risk_level:        Optional[str]    = None   # computed field
    total_invoices:    Optional[int]    = None   # computed from graph
    mismatch_count:    Optional[int]    = None   # computed from graph


# ── Response: Summary list item ────────────────────────────────
class TaxpayerSummary(BaseModel):
    gstin:          str
    name:           str
    state:          str
    risk_score:     float
    risk_level:     str
    mismatch_count: int = 0
    itc_at_risk:    float = 0.0
