"""
Rule-Based Risk Scoring Engine
===============================
Applies GST business rules to flag deterministic fraud patterns
(e.g., Missing GSTR-1, Fake IRN).
"""

from enum import Enum
from typing import List, Tuple


class MismatchType(str, Enum):
    TYPE_A_MISSING_GSTR1 = "Missing GSTR-1"
    TYPE_B_ITC_OVERCLAIM = "ITC Overclaim"
    TYPE_C_FAKE_IRN = "Invalid/Cancelled IRN"
    TYPE_D_NO_EWAY = "Missing e-Way Bill"
    TYPE_E_CIRCULAR = "Circular Trading Suspected"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleEngine:
    def __init__(self):
        # Base penalties for different mismatch types
        self.penalties = {
            MismatchType.TYPE_A_MISSING_GSTR1: 0.70,
            MismatchType.TYPE_B_ITC_OVERCLAIM: 0.50,
            MismatchType.TYPE_C_FAKE_IRN: 0.90,
            MismatchType.TYPE_D_NO_EWAY: 0.60,
            MismatchType.TYPE_E_CIRCULAR: 0.85,
        }

    def evaluate(self, mismatches: List[MismatchType]) -> Tuple[RiskLevel, float]:
        """
        Calculates deterministic rule-based risk score and level.
        """
        return self._resolve_risk(mismatches)

    def _resolve_risk(self, mismatches: List[MismatchType]) -> Tuple[RiskLevel, float]:
        if not mismatches:
            return RiskLevel.LOW, 0.05

        score = 0.0
        for m in mismatches:
            score += self.penalties.get(m, 0.0)

        # Cap deterministic score at 0.99
        score = min(score, 0.99)

        if score >= 0.90:
            level = RiskLevel.CRITICAL
        elif score >= 0.70:
            level = RiskLevel.HIGH
        elif score >= 0.40:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        return level, score