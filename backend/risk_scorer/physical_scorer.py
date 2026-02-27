"""
Physical Existence Scorer
===========================
Evaluates whether a vendor has demonstrable physical operations
(real goods movement, valid invoices, authenticated documents).

Detects ghost / shell companies that exist only on paper.

Sub-scores:
    A. IRN validity rate        (35% of component)
    B. e-Way bill consistency   (35% of component)
    C. Payment status rate      (20% of component)
    D. Registration age score   (10% of component)

Output: 0–100 (higher = more likely to be a real business)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"


@dataclass
class PhysicalScoreBreakdown:
    """Detailed breakdown of physical existence score."""
    gstin: str
    final_score: float = 100.0
    irn_validity_score: float = 100.0
    eway_consistency_score: float = 100.0
    payment_status_score: float = 100.0
    registration_age_score: float = 100.0
    total_invoices_issued: int = 0
    irn_valid_count: int = 0
    irn_missing_count: int = 0
    irn_cancelled_count: int = 0
    irn_rate: float = 1.0
    eway_required_count: int = 0
    eway_present_count: int = 0
    eway_rate: float = 1.0
    paid_invoices: int = 0
    payment_rate: float = 1.0
    registration_age_months: int = 0
    business_type: str = ""
    is_active: bool = True
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gstin": self.gstin,
            "final_score": round(self.final_score, 2),
            "irn_validity_score": round(self.irn_validity_score, 2),
            "eway_consistency_score": round(self.eway_consistency_score, 2),
            "payment_status_score": round(self.payment_status_score, 2),
            "registration_age_score": round(self.registration_age_score, 2),
            "total_invoices_issued": self.total_invoices_issued,
            "irn_valid_count": self.irn_valid_count,
            "irn_missing_count": self.irn_missing_count,
            "irn_cancelled_count": self.irn_cancelled_count,
            "irn_rate": round(self.irn_rate, 4),
            "eway_required_count": self.eway_required_count,
            "eway_present_count": self.eway_present_count,
            "eway_rate": round(self.eway_rate, 4),
            "paid_invoices": self.paid_invoices,
            "payment_rate": round(self.payment_rate, 4),
            "registration_age_months": self.registration_age_months,
            "business_type": self.business_type,
            "is_active": self.is_active,
            "details": self.details,
        }


class PhysicalScorer:
    """
    Scores vendor based on physical / documentary evidence of real operations.

    Score = 0.35 × IRNValidityRate × 100
          + 0.35 × eWayConsistencyRate × 100
          + 0.20 × PaymentStatusRate × 100
          + 0.10 × RegistrationAgeScore
    """

    def __init__(self, driver=None, session=None):
        self._driver = driver
        self._session = session
        self._owns_driver = False
        self._owns_session = False

        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

        if self._session is None:
            engine = create_engine(POSTGRES_URL)
            Session = sessionmaker(bind=engine)
            self._session = Session()
            self._owns_session = True

    def close(self):
        if self._owns_session and self._session:
            self._session.close()
        if self._owns_driver and self._driver:
            self._driver.close()

    def _get_irn_stats(self, gstin: str) -> Dict:
        """IRN coverage for invoices issued by this vendor."""
        row = self._session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT inv.invoice_number) AS total,
                    COUNT(DISTINCT CASE WHEN irn.status = 'Active' THEN inv.invoice_number END) AS valid,
                    COUNT(DISTINCT CASE WHEN irn.irn_number IS NULL THEN inv.invoice_number END) AS missing,
                    COUNT(DISTINCT CASE WHEN irn.status = 'Cancelled' THEN inv.invoice_number END) AS cancelled
                FROM invoices inv
                LEFT JOIN irn_records irn ON inv.invoice_number = irn.invoice_number
                WHERE inv.seller_gstin = :g
            """),
            {"g": gstin},
        ).fetchone()

        return {
            "total": int(row[0]) if row else 0,
            "valid": int(row[1]) if row else 0,
            "missing": int(row[2]) if row else 0,
            "cancelled": int(row[3]) if row else 0,
        }

    def _get_eway_stats(self, gstin: str) -> Dict:
        """e-Way bill coverage for high-value invoices."""
        row = self._session.execute(
            text("""
                SELECT
                    COUNT(DISTINCT inv.invoice_number) AS required,
                    COUNT(DISTINCT CASE
                        WHEN ew.status IN ('Active', 'Valid')
                        THEN inv.invoice_number END) AS present
                FROM invoices inv
                LEFT JOIN eway_bills ew ON inv.invoice_number = ew.invoice_number
                WHERE inv.seller_gstin = :g
                  AND inv.total_value > 50000
            """),
            {"g": gstin},
        ).fetchone()

        return {
            "required": int(row[0]) if row else 0,
            "present": int(row[1]) if row else 0,
        }

    def _get_payment_stats(self, gstin: str) -> Dict:
        """Payment status of invoices issued."""
        row = self._session.execute(
            text("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(CASE WHEN payment_status = 'Paid' THEN 1 END) AS paid
                FROM invoices
                WHERE seller_gstin = :g
            """),
            {"g": gstin},
        ).fetchone()

        return {
            "total": int(row[0]) if row else 0,
            "paid": int(row[1]) if row else 0,
        }

    def _get_registration_info(self, gstin: str) -> Dict:
        """Registration age and metadata from Neo4j."""
        with self._driver.session() as neo_sess:
            rec = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})
                RETURN t.registration_date AS reg_date,
                       t.business_type     AS btype,
                       t.is_active         AS active
                """,
                gstin=gstin,
            ).single()

        if rec is None:
            return {"age_months": 0, "business_type": "Unknown", "is_active": True}

        # Parse registration date
        age_months = 0
        reg_date = rec["reg_date"]
        if reg_date:
            try:
                if isinstance(reg_date, str):
                    rd = datetime.strptime(reg_date[:10], "%Y-%m-%d")
                else:
                    rd = reg_date
                delta = datetime.now() - rd
                age_months = max(int(delta.days / 30), 0)
            except (ValueError, TypeError):
                age_months = 12  # default

        return {
            "age_months": age_months,
            "business_type": rec["btype"] or "Unknown",
            "is_active": rec["active"] if rec["active"] is not None else True,
        }

    def score_vendor(self, gstin: str) -> PhysicalScoreBreakdown:
        """Compute physical existence score for one vendor."""
        breakdown = PhysicalScoreBreakdown(gstin=gstin)

        # ── IRN stats ──
        irn = self._get_irn_stats(gstin)
        breakdown.total_invoices_issued = irn["total"]
        breakdown.irn_valid_count = irn["valid"]
        breakdown.irn_missing_count = irn["missing"]
        breakdown.irn_cancelled_count = irn["cancelled"]

        if irn["total"] > 0:
            breakdown.irn_rate = irn["valid"] / irn["total"]
        else:
            breakdown.irn_rate = 1.0  # no invoices, neutral

        breakdown.irn_validity_score = breakdown.irn_rate * 100

        # Extra penalty for cancelled IRNs (fraud indicator)
        if irn["cancelled"] > 0:
            cancel_penalty = min((irn["cancelled"] / max(irn["total"], 1)) * 50, 30)
            breakdown.irn_validity_score = max(
                0, breakdown.irn_validity_score - cancel_penalty
            )

        # ── e-Way stats ──
        eway = self._get_eway_stats(gstin)
        breakdown.eway_required_count = eway["required"]
        breakdown.eway_present_count = eway["present"]

        if eway["required"] > 0:
            breakdown.eway_rate = eway["present"] / eway["required"]
        else:
            breakdown.eway_rate = 1.0  # no high-value invoices

        breakdown.eway_consistency_score = breakdown.eway_rate * 100

        # ── Payment stats ──
        pay = self._get_payment_stats(gstin)
        if pay["total"] > 0:
            breakdown.payment_rate = pay["paid"] / pay["total"]
            breakdown.paid_invoices = pay["paid"]
        else:
            breakdown.payment_rate = 1.0

        breakdown.payment_status_score = breakdown.payment_rate * 100

        # ── Registration age ──
        reg = self._get_registration_info(gstin)
        breakdown.registration_age_months = reg["age_months"]
        breakdown.business_type = reg["business_type"]
        breakdown.is_active = reg["is_active"]

        # Score: newer = riskier, older = safer
        # <3 months: 20, 3-6: 40, 6-12: 60, 12-24: 80, 24+: 100
        age = reg["age_months"]
        if age < 3:
            age_score = 20.0
        elif age < 6:
            age_score = 40.0
        elif age < 12:
            age_score = 60.0
        elif age < 24:
            age_score = 80.0
        else:
            age_score = 100.0

        # Inactive penalty
        if not reg["is_active"]:
            age_score = max(age_score - 40, 0)

        # Trader type has slightly higher base risk
        if reg["business_type"] == "Trader":
            age_score = max(age_score - 10, 0)

        breakdown.registration_age_score = age_score

        # ── Final score ──
        breakdown.final_score = (
            0.35 * breakdown.irn_validity_score
            + 0.35 * breakdown.eway_consistency_score
            + 0.20 * breakdown.payment_status_score
            + 0.10 * breakdown.registration_age_score
        )

        # ── Details ──
        if irn["missing"] > 0:
            breakdown.details.append(
                f"{irn['missing']} invoices missing IRN — "
                f"document authentication concern"
            )
        if irn["cancelled"] > 0:
            breakdown.details.append(
                f"{irn['cancelled']} invoices have CANCELLED IRN — "
                f"possible fake invoice indicator"
            )
        if eway["required"] > 0 and eway["present"] == 0:
            breakdown.details.append(
                f"No e-Way bills for {eway['required']} high-value invoices — "
                f"no evidence of goods movement"
            )
        elif eway["required"] > 0 and breakdown.eway_rate < 0.5:
            breakdown.details.append(
                f"Only {breakdown.eway_rate*100:.0f}% e-Way bill coverage — "
                f"weak goods movement evidence"
            )
        if age < 6:
            breakdown.details.append(
                f"Recently registered ({age} months) — "
                f"elevated risk for new entities"
            )
        if not reg["is_active"]:
            breakdown.details.append("Entity marked as INACTIVE")
        if irn["total"] == 0:
            breakdown.details.append(
                "No invoices issued — limited operational evidence"
            )
        if not breakdown.details:
            breakdown.details.append(
                "Strong physical evidence — valid IRNs, e-Way bills, payments"
            )

        return breakdown

    def score_all_vendors(self) -> List[PhysicalScoreBreakdown]:
        """Score all vendors."""
        with self._driver.session() as neo_sess:
            result = neo_sess.run(
                "MATCH (t:Taxpayer) RETURN t.gstin AS gstin ORDER BY t.gstin"
            )
            gstins = [r["gstin"] for r in result]

        results = []
        for i, gstin in enumerate(gstins):
            score = self.score_vendor(gstin)
            results.append(score)
            if (i + 1) % 20 == 0:
                logger.info("Physical scores computed: %d/%d", i + 1, len(gstins))

        return results
        