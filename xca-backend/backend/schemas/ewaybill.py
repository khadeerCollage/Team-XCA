"""
schemas/ewaybill.py
"""

from pydantic import BaseModel, Field
from typing import Optional


class EWayBillCreate(BaseModel):
    ewb_no:      str
    invoice_no:  str
    seller_gstin: str
    buyer_gstin:  str
    vehicle_no:  Optional[str] = None
    from_state:  str
    to_state:    str
    goods_value: float = Field(gt=0)
    valid:       bool  = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "ewb_no": "EWB-33112299",
                "invoice_no": "INV-2024-0882",
                "seller_gstin": "29XYZPQ1234F1Z5",
                "buyer_gstin": "27ABCDE5678G2H6",
                "vehicle_no": "KA-01-AB-1234",
                "from_state": "29",
                "to_state": "27",
                "goods_value": 500000.0,
                "valid": True
            }
        }
    }


class EWayBillResponse(BaseModel):
    ewb_no:      str
    invoice_no:  str
    seller_gstin: str
    buyer_gstin:  str
    vehicle_no:  Optional[str] = None
    from_state:  str
    to_state:    str
    goods_value: float
    valid:       bool
