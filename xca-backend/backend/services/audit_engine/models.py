"""
Pydantic models for audit engine input validation and structured data.
All fields are mandatory as per PRD specification.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    APPROVED = "ITC APPROVED"
    REJECTED = "ITC REJECTED"


class CheckpointStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class ValidationInput(BaseModel):
    """Input schema — all fields mandatory."""

    invoice_number: str = Field(..., min_length=1, description="Invoice identifier")
    buyer_name: str = Field(..., min_length=1, description="Buyer entity name")
    buyer_gstin: str = Field(..., min_length=15, max_length=15, description="Buyer GSTIN (15 chars)")
    seller_name: str = Field(..., min_length=1, description="Seller entity name")
    seller_gstin: str = Field(..., min_length=15, max_length=15, description="Seller GSTIN (15 chars)")
    itc_amount: float = Field(..., gt=0, description="ITC amount claimed (must be > 0)")
    gstr1_filed: bool = Field(..., description="Whether GSTR-1 has been filed by seller")
    gstr3b_filed: bool = Field(..., description="Whether GSTR-3B has been filed by seller")
    tax_paid: bool = Field(..., description="Whether tax has been paid by seller")
    irn_valid: bool = Field(..., description="Whether Invoice Registration Number is valid")
    eway_bill_verified: bool = Field(..., description="Whether e-Way Bill is verified")
    route: Optional[str] = Field(None, description="Transport route (e.g., BLR-MUM). If None, e-Way check skipped.")

    @field_validator("buyer_gstin", "seller_gstin")
    @classmethod
    def validate_gstin_format(cls, v: str) -> str:
        """Basic GSTIN format validation: 15 alphanumeric characters."""
        if not v.isalnum():
            raise ValueError("GSTIN must be alphanumeric (15 characters)")
        return v.upper()

    model_config = {"extra": "forbid"}


class Checkpoint(BaseModel):
    """Individual checkpoint result."""

    step: int
    name: str
    label: str
    status: CheckpointStatus
    detail: str


class AuditResult(BaseModel):
    """Complete audit result returned by the engine."""

    invoice_number: str
    buyer_name: str
    buyer_gstin: str
    seller_name: str
    seller_gstin: str
    itc_amount: float
    chain_valid: bool
    checkpoints: list[Checkpoint]
    break_point: Optional[str] = None
    break_step: Optional[int] = None
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    decision: Decision
    explanation: str
    chain_summary: str
    report: str
