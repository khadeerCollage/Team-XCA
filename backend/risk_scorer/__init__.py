"""
Vendor Risk Scorer — GST CIBIL Score Engine
=============================================
Assigns every GSTIN a compliance score from 0 to 100.

Components:
    1. Filing Frequency Score    (25%)
    2. Dispute / Mismatch Score  (25%)
    3. Network Connections Score (25%)
    4. Physical Existence Score  (25%)

Score Bands:
    80–100 → SAFE
    50–79  → MODERATE
    30–49  → HIGH RISK
    0–29   → FRAUD

Downstream Actions:
    Score ≥ 80  → Auto ITC Approval
    50 ≤ Score < 80 → Manual Review
    Score < 50  → Block ITC
"""

from .vendor_risk_engine import VendorRiskEngine, compute_all_vendor_scores
from .itc_decision import ITCDecisionEngine, decide_itc_all
from .explainer import generate_vendor_explanation

__all__ = [
    "VendorRiskEngine",
    "compute_all_vendor_scores",
    "ITCDecisionEngine",
    "decide_itc_all",
    "generate_vendor_explanation",
]