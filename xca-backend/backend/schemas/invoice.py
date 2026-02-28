"""
schemas/invoice.py
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class InvoiceStatus(str, Enum):
    FILED    = "Filed"
    AMENDED  = "Amended"
    CANCELLED = "Cancelled"
    PENDING  = "Pending"


class GSTRate(float, Enum):
    ZERO     = 0
    FIVE     = 5
    TWELVE   = 12
    EIGHTEEN = 18
    TWENTY_EIGHT = 28


# ── Request: Create Invoice ────────────────────────────────────
class InvoiceCreate(BaseModel):
    invoice_no:     str   = Field(..., description="Unique invoice number eg. INV-2024-0001")
    irn:            Optional[str] = Field(None, description="Invoice Reference Number from NIC")
    seller_gstin:   str   = Field(..., min_length=15, max_length=15)
    buyer_gstin:    str   = Field(..., min_length=15, max_length=15)
    invoice_date:   str   = Field(..., description="Date as YYYY-MM-DD")
    taxable_value:  float = Field(..., gt=0)
    gst_rate:       float = Field(..., ge=0, le=28)
    igst:           float = Field(default=0.0, ge=0)
    cgst:           float = Field(default=0.0, ge=0)
    sgst:           float = Field(default=0.0, ge=0)
    period:         str   = Field(..., description="Filing period eg. Oct-2024")
    status:         InvoiceStatus = InvoiceStatus.FILED

    @field_validator("seller_gstin", "buyer_gstin")
    @classmethod
    def uppercase_gstin(cls, v: str) -> str:
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "invoice_no": "INV-2024-0882",
                "irn": "IRN-ABCD-1234-EFGH",
                "seller_gstin": "29XYZPQ1234F1Z5",
                "buyer_gstin": "27ABCDE5678G2H6",
                "invoice_date": "2024-10-15",
                "taxable_value": 500000.0,
                "gst_rate": 12,
                "igst": 60000.0,
                "cgst": 0.0,
                "sgst": 0.0,
                "period": "Oct-2024",
                "status": "Filed"
            }
        }
    }


# ── Response: Invoice from DB ──────────────────────────────────
class InvoiceResponse(BaseModel):
    invoice_no:     str
    irn:            Optional[str]  = None
    seller_gstin:   str
    buyer_gstin:    str
    invoice_date:   str
    taxable_value:  float
    gst_rate:       float
    igst:           float = 0.0
    cgst:           float = 0.0
    sgst:           float = 0.0
    total_gst:      float = 0.0       # computed: igst + cgst + sgst
    period:         str
    status:         str
    has_mismatch:   Optional[bool]  = False
    missing_ewb:    Optional[bool]  = False


# ── Response: Invoice with full chain ─────────────────────────
class InvoiceChain(BaseModel):
    invoice:       InvoiceResponse
    irn_valid:     bool = False
    ewaybill_no:   Optional[str]  = None
    ewaybill_valid: bool = False
    gstr1_filed:   bool = False
    gstr2b_match:  bool = False
    payment_done:  bool = False
    chain_broken_at: Optional[str] = None   # "GSTR-2B" / "IRN" / "e-Way Bill"

