"""
Audit Engine — main orchestrator.

This is the single entry point: generate_audit_report(validation_input)
It chains together: validate → classify risk → render report.
"""

from app.core.models import (
    ValidationInput,
    AuditResult,
)
from app.core.validator import evaluate_chain, find_break_point, is_chain_valid
from app.core.risk import determine_risk_level, calculate_risk_score, generate_decision
from app.core.renderer import render_report, build_chain_summary, build_explanation


def generate_audit_report(data: ValidationInput) -> AuditResult:
    """
    Main entry point. Takes validated input, returns a complete AuditResult.

    Pipeline:
        1. evaluate_chain()        → list of Checkpoint
        2. find_break_point()      → first failure name + step
        3. is_chain_valid()        → bool
        4. determine_risk_level()  → RiskLevel enum
        5. calculate_risk_score()  → 0–100
        6. generate_decision()     → APPROVED / REJECTED
        7. render_report()         → formatted string
    """
    # Step 1: Evaluate all checkpoints
    checkpoints = evaluate_chain(data)

    # Step 2: Find first break point
    break_name, break_step = find_break_point(checkpoints)

    # Step 3: Is chain intact?
    chain_valid = is_chain_valid(checkpoints)

    # Step 4: Risk classification
    risk_level = determine_risk_level(checkpoints)

    # Step 5: Risk score
    risk_score = calculate_risk_score(checkpoints)

    # Step 6: Decision
    decision = generate_decision(risk_level)

    # Step 7: Build explanation and chain summary
    explanation = build_explanation(checkpoints)
    chain_summary = build_chain_summary(checkpoints)

    # Step 8: Render full report
    report = render_report(
        invoice_number=data.invoice_number,
        buyer_name=data.buyer_name,
        buyer_gstin=data.buyer_gstin,
        seller_name=data.seller_name,
        itc_amount=data.itc_amount,
        chain_valid=chain_valid,
        checkpoints=checkpoints,
        break_point=break_name,
        break_step=break_step,
        risk_level=risk_level,
        risk_score=risk_score,
        decision=decision.value,
        route=data.route,
    )

    return AuditResult(
        invoice_number=data.invoice_number,
        buyer_name=data.buyer_name,
        buyer_gstin=data.buyer_gstin,
        seller_name=data.seller_name,
        seller_gstin=data.seller_gstin,
        itc_amount=data.itc_amount,
        chain_valid=chain_valid,
        checkpoints=checkpoints,
        break_point=break_name,
        break_step=break_step,
        risk_level=risk_level,
        risk_score=risk_score,
        decision=decision,
        explanation=explanation,
        chain_summary=chain_summary,
        report=report,
    )
