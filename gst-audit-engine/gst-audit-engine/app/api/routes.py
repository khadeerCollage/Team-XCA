"""
FastAPI Router — REST API endpoints for the GST Audit Report Engine.
"""

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.core.models import ValidationInput, AuditResult
from app.core.engine import generate_audit_report

router = APIRouter(prefix="/api/v1", tags=["Audit"])


@router.post(
    "/audit",
    response_model=AuditResult,
    summary="Generate GST Audit Report",
    description=(
        "Accepts a validated invoice/compliance JSON payload and returns "
        "a structured audit report with risk classification and ITC decision."
    ),
)
async def audit_invoice(payload: ValidationInput) -> AuditResult:
    """
    Generate a structured GST audit report.

    - Evaluates the 5-step compliance chain
    - Classifies risk level
    - Returns formatted report with decision
    """
    try:
        result = generate_audit_report(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit engine error: {str(e)}")


@router.post(
    "/audit/report-only",
    response_model=dict,
    summary="Get Report Text Only",
    description="Returns only the formatted report string for display or PDF generation.",
)
async def audit_report_text(payload: ValidationInput) -> dict:
    """Return just the report text and decision — lightweight endpoint."""
    try:
        result = generate_audit_report(payload)
        return {
            "invoice_number": result.invoice_number,
            "decision": result.decision.value,
            "risk_level": result.risk_level.value,
            "risk_score": result.risk_score,
            "report": result.report,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit engine error: {str(e)}")


@router.get(
    "/health",
    summary="Health Check",
    description="Returns service health status.",
)
async def health_check() -> dict:
    return {"status": "healthy", "service": "GST Audit Report Engine", "version": "1.0.0"}
