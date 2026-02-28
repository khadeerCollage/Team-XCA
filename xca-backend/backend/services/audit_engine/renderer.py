"""
Report Renderer — builds the final structured audit report string.
Uses centralized templates. Pure functions, no side effects.
"""

from .models import Checkpoint, CheckpointStatus, RiskLevel
from .report_templates import (
    REPORT_VALID,
    REPORT_INVALID,
    REPORT_CRITICAL,
    SYMBOL_PASS,
    SYMBOL_FAIL,
    SYMBOL_SKIP,
    FAILURE_EXPLANATIONS,
    CHECKPOINT_DISPLAY,
)


def build_chain_summary(checkpoints: list[Checkpoint]) -> str:
    """
    Build a visual chain summary string.
    Example: GSTR1 PASS -> GSTR3B FAIL -> TAX FAIL -> IRN PASS -> EWB SKIP
    """
    parts: list[str] = []
    for cp in checkpoints:
        symbol = {
            CheckpointStatus.PASS: SYMBOL_PASS,
            CheckpointStatus.FAIL: SYMBOL_FAIL,
            CheckpointStatus.SKIPPED: SYMBOL_SKIP,
        }[cp.status]
        display = CHECKPOINT_DISPLAY.get(cp.name, cp.name)
        parts.append(f"{display} {symbol}")
    return " -> ".join(parts)


def build_explanation(checkpoints: list[Checkpoint]) -> str:
    """
    Build explanation text from all failed checkpoints.
    Shows the first major break prominently, lists all failures.
    """
    failed = [cp for cp in checkpoints if cp.status == CheckpointStatus.FAIL]
    if not failed:
        return ""

    lines: list[str] = []
    for cp in failed:
        explanation = FAILURE_EXPLANATIONS.get(cp.name, f"{cp.label} verification failed.")
        lines.append(explanation)
    return " ".join(lines)


def format_amount(amount: float) -> str:
    """Format amount with comma separation and no unnecessary decimals."""
    if amount == int(amount):
        return f"{int(amount):,}"
    return f"{amount:,.2f}"


def render_report(
    invoice_number: str,
    buyer_name: str,
    buyer_gstin: str,
    seller_name: str,
    itc_amount: float,
    chain_valid: bool,
    checkpoints: list[Checkpoint],
    break_point: str | None,
    break_step: int | None,
    risk_level: RiskLevel,
    risk_score: int,
    decision: str,
    route: str | None,
) -> str:
    """
    Render the final audit report string using the appropriate template.
    """
    buyer_gstin_short = buyer_gstin[:7] + "..."
    formatted_amount = format_amount(itc_amount)
    chain_summary = build_chain_summary(checkpoints)

    passed_count = sum(1 for cp in checkpoints if cp.status == CheckpointStatus.PASS)

    if chain_valid:
        irn_line = "IRN confirmed. " if any(
            cp.name == "IRN" and cp.status == CheckpointStatus.PASS for cp in checkpoints
        ) else ""

        eway_line = ""
        eway_cp = next((cp for cp in checkpoints if cp.name == "EWB"), None)
        if eway_cp and eway_cp.status == CheckpointStatus.PASS and route:
            eway_line = f"e-Way Bill verified for route {route}. "
        elif eway_cp and eway_cp.status == CheckpointStatus.SKIPPED:
            eway_line = "e-Way Bill not required (no transport route). "

        return REPORT_VALID.format(
            invoice_number=invoice_number,
            buyer_name=buyer_name,
            buyer_gstin_short=buyer_gstin_short,
            itc_amount=formatted_amount,
            seller_name=seller_name,
            passed_count=passed_count,
            irn_line=irn_line,
            eway_line=eway_line,
            chain_summary=chain_summary,
            risk_level=risk_level.value,
            risk_score=risk_score,
            decision=decision,
        )

    # INVALID chain
    explanation = build_explanation(checkpoints)

    if risk_level == RiskLevel.CRITICAL:
        return REPORT_CRITICAL.format(
            invoice_number=invoice_number,
            buyer_name=buyer_name,
            buyer_gstin_short=buyer_gstin_short,
            itc_amount=formatted_amount,
            seller_name=seller_name,
            explanation=explanation,
            chain_summary=chain_summary,
            risk_score=risk_score,
            decision=decision,
        )

    return REPORT_INVALID.format(
        invoice_number=invoice_number,
        buyer_name=buyer_name,
        buyer_gstin_short=buyer_gstin_short,
        itc_amount=formatted_amount,
        seller_name=seller_name,
        explanation=explanation,
        break_point=break_point,
        break_step=break_step,
        chain_summary=chain_summary,
        risk_level=risk_level.value,
        risk_score=risk_score,
        decision=decision,
    )
