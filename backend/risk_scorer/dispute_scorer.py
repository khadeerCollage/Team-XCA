"""
Dispute / Mismatch Rate Scorer
================================
Evaluates the proportion and severity of invoice mismatches
associated with each vendor.

Sub-scores:
    A. Mismatch rate         (40% of component)
    B. Severity-weighted     (30% of component)
    C. Trend improvement     (20% of component)
    D. ITC overclaim ratio   (10% of component)

Output: 0–100 score (higher = fewer disputes = safer)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"

# Severity weights for mismatch types
MISMATCH_SEVERITY = {
    "Fake/Missing IRN": 1.0,      # CRITICAL → highest penalty
    "Missing GSTR-1": 0.75,       # HIGH
    "No e-Way Bill": 0.70,        # HIGH
    "Amount Mismatch": 0.50,      # MEDIUM
    "No Mismatch": 0.0,           # clean
}

EXPECTED_FILING_PERIODS = 6


@dataclass
class DisputeScoreBreakdown:
    """Detailed breakdown of dispute/mismatch score."""
    gstin: str
    final_score: float = 100.0
    mismatch_rate_score: float = 100.0
    severity_weighted_score: float = 100.0
    trend_score: float = 100.0
    itc_overclaim_score: float = 100.0
    total_invoices: int = 0
    mismatched_invoices: int = 0
    mismatch_rate: float = 0.0
    mismatch_type_counts: Dict[str, int] = field(default_factory=dict)
    avg_severity: float = 0.0
    itc_overclaim_ratio: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gstin": self.gstin,
            "final_score": round(self.final_score, 2),
            "mismatch_rate_score": round(self.mismatch_rate_score, 2),
            "severity_weighted_score": round(self.severity_weighted_score, 2),
            "trend_score": round(self.trend_score, 2),
            "itc_overclaim_score": round(self.itc_overclaim_score, 2),
            "total_invoices": self.total_invoices,
            "mismatched_invoices": self.mismatched_invoices,
            "mismatch_rate": round(self.mismatch_rate, 4),
            "mismatch_type_counts": self.mismatch_type_counts,
            "avg_severity": round(self.avg_severity, 4),
            "itc_overclaim_ratio": round(self.itc_overclaim_ratio, 4),
            "details": self.details,
        }


class DisputeScorer:
    """
    Scores vendor based on invoice dispute / mismatch history.

    Score = 0.40 × (100 - MismatchRate×100)
          + 0.30 × (100 - AvgSeverity×100)
          + 0.20 × TrendScore
          + 0.10 × (100 - OverclaimPenalty)
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

    def _get_invoice_mismatch_data(self, gstin: str) -> Dict:
        """
        Gather mismatch information for all invoices involving this GSTIN
        (as seller or buyer).
        """
        # Total invoices
        row = self._session.execute(
            text("""
                SELECT COUNT(*) FROM invoices
                WHERE seller_gstin = :g OR buyer_gstin = :g
            """),
            {"g": gstin},
        ).fetchone()
        total_invoices = int(row[0]) if row else 0

        # Invoices with missing IRN
        row_irn = self._session.execute(
            text("""
                SELECT COUNT(DISTINCT inv.invoice_number)
                FROM invoices inv
                LEFT JOIN irn_records irn
                    ON inv.invoice_number = irn.invoice_number
                WHERE (inv.seller_gstin = :g OR inv.buyer_gstin = :g)
                  AND (irn.irn_number IS NULL OR irn.status = 'Cancelled')
            """),
            {"g": gstin},
        ).fetchone()
        irn_missing = int(row_irn[0]) if row_irn else 0

        # Invoices with missing GSTR-1
        row_g1 = self._session.execute(
            text("""
                SELECT COUNT(DISTINCT inv.invoice_number)
                FROM invoices inv
                LEFT JOIN gstr1_records g1
                    ON inv.invoice_number = g1.invoice_number
                   AND inv.seller_gstin = g1.seller_gstin
                WHERE inv.seller_gstin = :g
                  AND (g1.invoice_number IS NULL OR g1.return_status != 'Filed')
            """),
            {"g": gstin},
        ).fetchone()
        gstr1_missing = int(row_g1[0]) if row_g1 else 0

        # Invoices > 50K without eway bill
        row_eway = self._session.execute(
            text("""
                SELECT COUNT(DISTINCT inv.invoice_number)
                FROM invoices inv
                LEFT JOIN eway_bills ew
                    ON inv.invoice_number = ew.invoice_number
                WHERE (inv.seller_gstin = :g OR inv.buyer_gstin = :g)
                  AND inv.total_value > 50000
                  AND (ew.eway_bill_number IS NULL
                       OR ew.status NOT IN ('Active', 'Valid'))
            """),
            {"g": gstin},
        ).fetchone()
        eway_missing = int(row_eway[0]) if row_eway else 0

        # Amount mismatches (GSTR-1 vs invoice total differs > 2%)
        row_amt = self._session.execute(
            text("""
                SELECT COUNT(DISTINCT inv.invoice_number)
                FROM invoices inv
                JOIN gstr1_records g1
                    ON inv.invoice_number = g1.invoice_number
                   AND inv.seller_gstin = g1.seller_gstin
                WHERE inv.seller_gstin = :g
                  AND ABS(inv.total_value - g1.total_invoice_value) >
                      GREATEST(inv.total_value * 0.02, 100)
            """),
            {"g": gstin},
        ).fetchone()
        amount_mismatch = int(row_amt[0]) if row_amt else 0

        return {
            "total_invoices": total_invoices,
            "irn_missing": irn_missing,
            "gstr1_missing": gstr1_missing,
            "eway_missing": eway_missing,
            "amount_mismatch": amount_mismatch,
        }

    def _get_itc_overclaim_ratio(self, gstin: str) -> float:
        """
        ITC overclaim = (claimed - available) / available.
        """
        row_avail = self._session.execute(
            text("""
                SELECT COALESCE(SUM(cgst_amount + sgst_amount + igst_amount), 0)
                FROM gstr1_records
                WHERE buyer_gstin = :g AND return_status = 'Filed'
            """),
            {"g": gstin},
        ).fetchone()
        available = float(row_avail[0]) if row_avail else 0.0

        row_claimed = self._session.execute(
            text("""
                SELECT COALESCE(SUM(total_itc_claimed), 0)
                FROM gstr3b_records
                WHERE gstin = :g
            """),
            {"g": gstin},
        ).fetchone()
        claimed = float(row_claimed[0]) if row_claimed else 0.0

        if available > 0:
            return max((claimed - available) / available, 0.0)
        elif claimed > 0:
            return 1.0
        return 0.0

    def score_vendor(self, gstin: str) -> DisputeScoreBreakdown:
        """Compute dispute/mismatch score for one vendor."""
        breakdown = DisputeScoreBreakdown(gstin=gstin)

        data = self._get_invoice_mismatch_data(gstin)
        total = data["total_invoices"]
        breakdown.total_invoices = total

        if total == 0:
            breakdown.final_score = 80.0  # neutral — no data
            breakdown.details.append("No invoice history — neutral score assigned")
            return breakdown

        # Count unique mismatched invoices (union, not sum — some overlap)
        # Approximation: take max of individual counts as lower bound
        unique_mismatched = max(
            data["irn_missing"],
            data["gstr1_missing"],
            data["eway_missing"],
            data["amount_mismatch"],
        )
        # Better estimate: use a heuristic sum with overlap discount
        raw_sum = (
            data["irn_missing"]
            + data["gstr1_missing"]
            + data["eway_missing"]
            + data["amount_mismatch"]
        )
        estimated_unique = min(raw_sum, total)  # can't exceed total
        breakdown.mismatched_invoices = estimated_unique

        breakdown.mismatch_type_counts = {
            "Fake/Missing IRN": data["irn_missing"],
            "Missing GSTR-1": data["gstr1_missing"],
            "No e-Way Bill": data["eway_missing"],
            "Amount Mismatch": data["amount_mismatch"],
        }

        # ── Sub-score A: Mismatch rate (40%) ──
        mismatch_rate = estimated_unique / total
        breakdown.mismatch_rate = mismatch_rate
        breakdown.mismatch_rate_score = max(0.0, (1.0 - mismatch_rate) * 100)

        # ── Sub-score B: Severity-weighted (30%) ──
        severity_sum = 0.0
        severity_count = 0
        for mtype, count in breakdown.mismatch_type_counts.items():
            if count > 0:
                weight = MISMATCH_SEVERITY.get(mtype, 0.5)
                severity_sum += weight * count
                severity_count += count

        if severity_count > 0:
            avg_severity = severity_sum / severity_count
        else:
            avg_severity = 0.0
        breakdown.avg_severity = avg_severity
        breakdown.severity_weighted_score = max(0.0, (1.0 - avg_severity) * 100)

        # ── Sub-score C: Trend score (20%) ──
        # For hackathon: if mismatch rate < 30%, treat as improving
        if mismatch_rate < 0.15:
            trend = 100.0
        elif mismatch_rate < 0.30:
            trend = 75.0
        elif mismatch_rate < 0.50:
            trend = 50.0
        elif mismatch_rate < 0.70:
            trend = 25.0
        else:
            trend = 0.0
        breakdown.trend_score = trend

        # ── Sub-score D: ITC overclaim (10%) ──
        overclaim = self._get_itc_overclaim_ratio(gstin)
        breakdown.itc_overclaim_ratio = overclaim
        overclaim_penalty = min(overclaim * 100, 100)
        breakdown.itc_overclaim_score = max(0.0, 100.0 - overclaim_penalty)

        # ── Final score ──
        breakdown.final_score = (
            0.40 * breakdown.mismatch_rate_score
            + 0.30 * breakdown.severity_weighted_score
            + 0.20 * breakdown.trend_score
            + 0.10 * breakdown.itc_overclaim_score
        )

        # ── Details ──
        if data["irn_missing"] > 0:
            breakdown.details.append(
                f"{data['irn_missing']} invoices missing/cancelled IRN"
            )
        if data["gstr1_missing"] > 0:
            breakdown.details.append(
                f"{data['gstr1_missing']} invoices not reported in GSTR-1"
            )
        if data["eway_missing"] > 0:
            breakdown.details.append(
                f"{data['eway_missing']} high-value invoices without e-Way bill"
            )
        if data["amount_mismatch"] > 0:
            breakdown.details.append(
                f"{data['amount_mismatch']} invoices with amount discrepancy"
            )
        if overclaim > 0.1:
            breakdown.details.append(
                f"ITC overclaim ratio: {overclaim*100:.0f}% above available credit"
            )
        if mismatch_rate > 0.5:
            breakdown.details.append(
                f"CRITICAL: {mismatch_rate*100:.0f}% invoice mismatch rate"
            )
        if not breakdown.details:
            breakdown.details.append("Clean dispute history — no significant mismatches")

        return breakdown

    def score_all_vendors(self) -> List[DisputeScoreBreakdown]:
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
                logger.info("Dispute scores computed: %d/%d", i + 1, len(gstins))

        return results