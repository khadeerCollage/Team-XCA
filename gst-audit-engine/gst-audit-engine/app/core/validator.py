"""
Chain Validator — evaluates GST compliance checkpoints in strict sequence.

Validation Order (per PRD):
    Step 1 → GSTR-1
    Step 2 → GSTR-3B
    Step 3 → Tax Payment
    Step 4 → IRN
    Step 5 → e-Way Bill
"""

from app.core.models import (
    ValidationInput,
    Checkpoint,
    CheckpointStatus,
)

# Strict evaluation order
CHECKPOINT_DEFINITIONS = [
    {
        "step": 1,
        "name": "GSTR1",
        "label": "GSTR-1 Filing",
        "field": "gstr1_filed",
        "pass_detail": "Seller has filed GSTR-1. Invoice reported to GST portal.",
        "fail_detail": "Seller has NOT filed GSTR-1. Invoice not reported to GST portal.",
    },
    {
        "step": 2,
        "name": "GSTR3B",
        "label": "GSTR-3B Filing",
        "field": "gstr3b_filed",
        "pass_detail": "Seller has filed GSTR-3B. Tax liability acknowledged.",
        "fail_detail": "Seller has NOT filed GSTR-3B. Tax liability not acknowledged.",
    },
    {
        "step": 3,
        "name": "TAX",
        "label": "Tax Payment",
        "field": "tax_paid",
        "pass_detail": "Tax has been paid by seller to the government.",
        "fail_detail": "Tax has NOT been paid by seller. Government has not received dues.",
    },
    {
        "step": 4,
        "name": "IRN",
        "label": "IRN Validation",
        "field": "irn_valid",
        "pass_detail": "Invoice Registration Number (IRN) is valid on IRP.",
        "fail_detail": "Invoice Registration Number (IRN) is INVALID or not found on IRP.",
    },
    {
        "step": 5,
        "name": "EWB",
        "label": "e-Way Bill Verification",
        "field": "eway_bill_verified",
        "pass_detail": "e-Way Bill verified for transport route.",
        "fail_detail": "e-Way Bill verification FAILED or not generated.",
    },
]


def evaluate_chain(data: ValidationInput) -> list[Checkpoint]:
    """
    Evaluate all 5 checkpoints in strict sequence.
    Returns a list of Checkpoint objects.

    If route is None → e-Way Bill checkpoint is SKIPPED (not required).
    """
    checkpoints: list[Checkpoint] = []

    for defn in CHECKPOINT_DEFINITIONS:
        field_name: str = defn["field"]

        # Special handling: skip e-Way Bill if no route provided
        if defn["name"] == "EWB" and data.route is None:
            checkpoints.append(
                Checkpoint(
                    step=defn["step"],
                    name=defn["name"],
                    label=defn["label"],
                    status=CheckpointStatus.SKIPPED,
                    detail="e-Way Bill verification skipped — no transport route provided.",
                )
            )
            continue

        value = getattr(data, field_name)

        if value:
            checkpoints.append(
                Checkpoint(
                    step=defn["step"],
                    name=defn["name"],
                    label=defn["label"],
                    status=CheckpointStatus.PASS,
                    detail=defn["pass_detail"],
                )
            )
        else:
            checkpoints.append(
                Checkpoint(
                    step=defn["step"],
                    name=defn["name"],
                    label=defn["label"],
                    status=CheckpointStatus.FAIL,
                    detail=defn["fail_detail"],
                )
            )

    return checkpoints


def find_break_point(checkpoints: list[Checkpoint]) -> tuple[str | None, int | None]:
    """
    Find the FIRST failed checkpoint (break point).
    Returns (checkpoint_name, step_number) or (None, None) if chain is intact.
    """
    for cp in checkpoints:
        if cp.status == CheckpointStatus.FAIL:
            return cp.name, cp.step
    return None, None


def is_chain_valid(checkpoints: list[Checkpoint]) -> bool:
    """Chain is valid only if no checkpoint has FAIL status."""
    return all(cp.status != CheckpointStatus.FAIL for cp in checkpoints)
