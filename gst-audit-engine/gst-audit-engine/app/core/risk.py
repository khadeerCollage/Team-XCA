"""
Risk Classifier — determines risk level and risk score based on checkpoint results.

Risk Classification Logic (per PRD):
    - All checkpoints pass       → LOW RISK    → ITC APPROVED
    - Minor issue (e.g. late filing but tax paid) → MEDIUM RISK
    - Major break (GSTR-3B missing / tax unpaid / IRN invalid) → HIGH RISK → ITC REJECTED
    - All checkpoints fail       → CRITICAL    → ITC REJECTED
"""

from app.core.models import (
    Checkpoint,
    CheckpointStatus,
    RiskLevel,
    Decision,
)

# Checkpoints considered "major" failures
MAJOR_FAIL_CHECKPOINTS = {"GSTR3B", "TAX", "IRN"}

# Risk score deductions per checkpoint failure
RISK_DEDUCTIONS = {
    "GSTR1": 15,
    "GSTR3B": 25,
    "TAX": 30,
    "IRN": 20,
    "EWB": 10,
}


def determine_risk_level(checkpoints: list[Checkpoint]) -> RiskLevel:
    """Classify risk based on which checkpoints failed."""
    failed = {cp.name for cp in checkpoints if cp.status == CheckpointStatus.FAIL}

    if not failed:
        return RiskLevel.LOW

    # All evaluated (non-skipped) checkpoints failed → CRITICAL
    evaluated = [cp for cp in checkpoints if cp.status != CheckpointStatus.SKIPPED]
    if all(cp.status == CheckpointStatus.FAIL for cp in evaluated):
        return RiskLevel.CRITICAL

    # Any major checkpoint failed → HIGH
    if failed & MAJOR_FAIL_CHECKPOINTS:
        return RiskLevel.HIGH

    # Only minor checkpoints failed (GSTR1 or EWB only) → MEDIUM
    return RiskLevel.MEDIUM


def calculate_risk_score(checkpoints: list[Checkpoint]) -> int:
    """
    Calculate a risk score from 0 (highest risk) to 100 (no risk).
    Starts at 100 and deducts per failed checkpoint.
    """
    score = 100
    for cp in checkpoints:
        if cp.status == CheckpointStatus.FAIL:
            score -= RISK_DEDUCTIONS.get(cp.name, 10)
    return max(score, 0)


def generate_decision(risk_level: RiskLevel) -> Decision:
    """
    Generate ITC decision based on risk level.
    LOW / MEDIUM → APPROVED (MEDIUM = warning, but still claimable)
    HIGH / CRITICAL → REJECTED
    """
    if risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        return Decision.APPROVED
    return Decision.REJECTED
