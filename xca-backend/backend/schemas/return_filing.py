"""
schemas/return_filing.py
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ReturnType(str, Enum):
    GSTR1  = "GSTR-1"
    GSTR2B = "GSTR-2B"
    GSTR3B = "GSTR-3B"


class ReturnCreate(BaseModel):
    return_type:     ReturnType
    gstin:           str = Field(..., min_length=15, max_length=15)
    period:          str
    filing_date:     Optional[str] = None
    total_itc_claimed: float = 0.0
    status:          str = "Filed"

    model_config = {
        "json_schema_extra": {
            "example": {
                "return_type": "GSTR-3B",
                "gstin": "27ABCDE5678G2H6",
                "period": "Oct-2024",
                "filing_date": "2024-11-20",
                "total_itc_claimed": 60000.0,
                "status": "Filed"
            }
        }
    }


class ReturnResponse(BaseModel):
    return_type:     str
    gstin:           str
    period:          str
    filing_date:     Optional[str] = None
    total_itc_claimed: float       = 0.0
    status:          str
    days_delayed:    Optional[int] = None   # computed
