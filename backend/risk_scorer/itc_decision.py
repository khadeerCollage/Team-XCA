"""
Auto ITC Decision Engine
==========================
Uses vendor risk scores to make automated ITC approval decisions.

Decision Logic:
    VendorScore ≥ 80  → AUTO APPROVE ITC
    50 ≤ Score < 80   → MANUAL REVIEW required
    Score < 50        → BLOCK ITC
    Score < 30        → BLOCK + FLAG FOR AUDIT

Also considers invoice-level mismatch results for fine-grained decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"


class ITCDecision:
    """Enum-like constants for ITC decisions."""
    AUTO_APPROVE = "AUTO_APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    BLOCK = "BLOCK"
    BLOCK_AND_AUDIT = "BLOCK_AND_AUDIT"


# Decision thresholds
AUTO_APPROVE_THRESHOLD = 80
MANUAL_REVIEW_THRESHOLD = 50
BLOCK_THRESHOLD = 30

# Decision descriptions
DECISION_DESCRIPTIONS = {
    ITCDecision.AUTO_APPROVE: "ITC automatically approved — vendor is compliant",
    ITCDecision.MANUAL_REVIEW: "ITC requires manual review — moderate risk detected",
    ITCDecision.BLOCK: "ITC BLOCKED — vendor risk score below threshold",
    ITCDecision.BLOCK_AND_AUDIT: "ITC BLOCKED — immediate audit recommended for suspected fraud",
}


@dataclass
class ITCDecisionResult:
    """Result of ITC decision for a buyer-vendor pair."""
    buyer_gstin: str
    vendor_gstin: str
    vendor_name: str = ""
    vendor_score: float = 0.0
    vendor_category: str = "UNKNOWN"
    itc_decision: str = ITCDecision.MANUAL_REVIEW
    itc_amount: float = 0.0
    eligible_amount: float = 0.0
    blocked_amount: float = 0.0
    decision_reason: str = ""
    invoice_count: int = 0
    invoice_details: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "buyer_gstin": self.buyer_gstin,
            "vendor_gstin": self.vendor_gstin,
            "vendor_name": self.vendor_name,
            "vendor_score": round(self.vendor_score, 2),
            "vendor_category": self.vendor_category,
            "itc_decision": self.itc_decision,
            "itc_amount": round(self.itc_amount, 2),
            "eligible_amount": round(self.eligible_amount, 2),
            "blocked_amount": round(self.blocked_amount, 2),
            "decision_reason": self.decision_reason,
            "invoice_count": self.invoice_count,
            "invoice_details": self.invoice_details,
        }


class ITCDecisionEngine:
    """
    Automated ITC approval/blocking engine powered by vendor risk scores.
    """

    def __init__(self, vendor_scores: List[dict] = None, session=None):
        """
        Parameters
        ----------
        vendor_scores : list[dict]
            Output from VendorRiskEngine.score_all_vendors().
            Each dict must have: gstin, final_score, risk_category, business_name
        session : sqlalchemy Session, optional
        """
        self._session = session
        self._owns_session = False

        if self._session is None:
            engine = create_engine(POSTGRES_URL)
            Session = sessionmaker(bind=engine)
            self._session = Session()
            self._owns_session = True

        # Build score lookup
        self._score_map: Dict[str, dict] = {}
        if vendor_scores:
            for vs in vendor_scores:
                self._score_map[vs["gstin"]] = vs

    def close(self):
        if self._owns_session and self._session:
            self._session.close()

    def _make_decision(self, vendor_score: float, vendor_category: str) -> str:
        """Core decision logic based on vendor score."""
        if vendor_score >= AUTO_APPROVE_THRESHOLD:
            return ITCDecision.AUTO_APPROVE
        elif vendor_score >= MANUAL_REVIEW_THRESHOLD:
            return ITCDecision.MANUAL_REVIEW
        elif vendor_score >= BLOCK_THRESHOLD:
            return ITCDecision.BLOCK
        else:
            return ITCDecision.BLOCK_AND_AUDIT

    def _get_invoices_between(self, buyer: str, seller: str) -> List[dict]:
        """Get all invoices between a buyer and seller."""
        rows = self._session.execute(
            text("""
                SELECT invoice_number, taxable_value, cgst, sgst, igst, total_value
                FROM invoices
                WHERE buyer_gstin = :buyer AND seller_gstin = :seller
                ORDER BY invoice_number
            """),
            {"buyer": buyer, "seller": seller},
        ).fetchall()

        results = []
        for row in rows:
            tax = float(row[2] or 0) + float(row[3] or 0) + float(row[4] or 0)
            results.append({
                "invoice_number": row[0],
                "taxable_value": float(row[1] or 0),
                "cgst": float(row[2] or 0),
                "sgst": float(row[3] or 0),
                "igst": float(row[4] or 0),
                "total_value": float(row[5] or 0),
                "itc_amount": tax,
            })
        return results

    def decide_for_pair(self, buyer_gstin: str, vendor_gstin: str) -> ITCDecisionResult:
        """
        Make ITC decision for a specific buyer-vendor pair.
        """
        result = ITCDecisionResult(
            buyer_gstin=buyer_gstin,
            vendor_gstin=vendor_gstin,
        )

        # Get vendor score
        vs = self._score_map.get(vendor_gstin, {})
        result.vendor_score = vs.get("final_score", 50.0)
        result.vendor_category = vs.get("risk_category", "UNKNOWN")
        result.vendor_name = vs.get("business_name", "Unknown")

        # Get invoices
        invoices = self._get_invoices_between(buyer_gstin, vendor_gstin)
        result.invoice_count = len(invoices)
        result.invoice_details = invoices

        total_itc = sum(inv["itc_amount"] for inv in invoices)
        result.itc_amount = total_itc

        # Make decision
        decision = self._make_decision(result.vendor_score, result.vendor_category)
        result.itc_decision = decision
        result.decision_reason = DECISION_DESCRIPTIONS[decision]

        # Compute eligible / blocked amounts
        if decision == ITCDecision.AUTO_APPROVE:
            result.eligible_amount = total_itc
            result.blocked_amount = 0.0
        elif decision == ITCDecision.MANUAL_REVIEW:
            result.eligible_amount = total_itc * 0.5  # provisional 50%
            result.blocked_amount = total_itc * 0.5
        elif decision == ITCDecision.BLOCK:
            result.eligible_amount = 0.0
            result.blocked_amount = total_itc
        else:  # BLOCK_AND_AUDIT
            result.eligible_amount = 0.0
            result.blocked_amount = total_itc

        return result

    def decide_for_buyer(self, buyer_gstin: str) -> List[ITCDecisionResult]:
        """
        Make ITC decisions for all vendors supplying to a buyer.
        """
        # Find all sellers for this buyer
        rows = self._session.execute(
            text("""
                SELECT DISTINCT seller_gstin
                FROM invoices
                WHERE buyer_gstin = :buyer
                ORDER BY seller_gstin
            """),
            {"buyer": buyer_gstin},
        ).fetchall()

        results = []
        for row in rows:
            seller = row[0]
            decision = self.decide_for_pair(buyer_gstin, seller)
            results.append(decision)

        return results

    def decide_all(self) -> List[ITCDecisionResult]:
        """
        Make ITC decisions for ALL buyer-vendor pairs in the database.
        """
        rows = self._session.execute(
            text("""
                SELECT DISTINCT buyer_gstin, seller_gstin
                FROM invoices
                ORDER BY buyer_gstin, seller_gstin
            """)
        ).fetchall()

        results = []
        for row in rows:
            decision = self.decide_for_pair(row[0], row[1])
            results.append(decision)

        # Print summary
        summary = {
            ITCDecision.AUTO_APPROVE: 0,
            ITCDecision.MANUAL_REVIEW: 0,
            ITCDecision.BLOCK: 0,
            ITCDecision.BLOCK_AND_AUDIT: 0,
        }
        total_itc = 0.0
        blocked_itc = 0.0

        for r in results:
            summary[r.itc_decision] += 1
            total_itc += r.itc_amount
            blocked_itc += r.blocked_amount

        print("\n" + "=" * 56)
        print("  Auto ITC Decision Summary")
        print("=" * 56)
        print(f"  Total buyer-vendor pairs : {len(results)}")
        print(f"  AUTO APPROVE             : {summary[ITCDecision.AUTO_APPROVE]}")
        print(f"  MANUAL REVIEW            : {summary[ITCDecision.MANUAL_REVIEW]}")
        print(f"  BLOCK                    : {summary[ITCDecision.BLOCK]}")
        print(f"  BLOCK + AUDIT            : {summary[ITCDecision.BLOCK_AND_AUDIT]}")
        print(f"  Total ITC value          : ₹{total_itc:,.0f}")
        print(f"  Blocked ITC value        : ₹{blocked_itc:,.0f}")
        print(f"  Block percentage         : {100*blocked_itc/max(total_itc,1):.1f}%")
        print("=" * 56 + "\n")

        return results


def decide_itc_all(vendor_scores: List[dict], session=None) -> List[dict]:
    """Convenience function: run ITC decisions for all pairs."""
    engine = ITCDecisionEngine(vendor_scores=vendor_scores, session=session)
    try:
        results = engine.decide_all()
        return [r.to_dict() for r in results]
    finally:
        engine.close()