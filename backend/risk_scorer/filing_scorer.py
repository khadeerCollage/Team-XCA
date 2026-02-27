"""
Filing Frequency Scorer
========================
Evaluates how consistently and punctually a vendor files
GSTR-1 and GSTR-3B returns.

Sub-scores:
    A. On-time filing rate         (40% of component)
    B. GSTR-3B coverage rate       (30% of component)
    C. GSTR-1 coverage rate        (20% of component)
    D. Filing gap penalty          (10% of component)

Output: 0–100 score (higher = more compliant)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"

EXPECTED_FILING_PERIODS = 6
LATE_FILING_THRESHOLD_DAYS = 20  # returns due by 20th of following month


@dataclass
class FilingScoreBreakdown:
    """Detailed breakdown of filing frequency score."""
    gstin: str
    final_score: float = 0.0
    ontime_rate_score: float = 0.0
    gstr3b_coverage_score: float = 0.0
    gstr1_coverage_score: float = 0.0
    filing_gap_penalty: float = 0.0
    gstr1_filed: int = 0
    gstr1_expected: int = EXPECTED_FILING_PERIODS
    gstr3b_filed: int = 0
    gstr3b_expected: int = EXPECTED_FILING_PERIODS
    late_filings: int = 0
    missing_filings: int = 0
    consecutive_misses: int = 0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gstin": self.gstin,
            "final_score": round(self.final_score, 2),
            "ontime_rate_score": round(self.ontime_rate_score, 2),
            "gstr3b_coverage_score": round(self.gstr3b_coverage_score, 2),
            "gstr1_coverage_score": round(self.gstr1_coverage_score, 2),
            "filing_gap_penalty": round(self.filing_gap_penalty, 2),
            "gstr1_filed": self.gstr1_filed,
            "gstr1_expected": self.gstr1_expected,
            "gstr3b_filed": self.gstr3b_filed,
            "gstr3b_expected": self.gstr3b_expected,
            "late_filings": self.late_filings,
            "missing_filings": self.missing_filings,
            "consecutive_misses": self.consecutive_misses,
            "details": self.details,
        }


class FilingScorer:
    """
    Computes filing frequency compliance score for a GSTIN.

    Score = 0.40 × OnTimeRate
          + 0.30 × GSTR3BCoverage
          + 0.20 × GSTR1Coverage
          + 0.10 × (100 - FilingGapPenalty)
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

    def _count_gstr1_filings(self, gstin: str) -> Tuple[int, int, List[str]]:
        """
        Count GSTR-1 filings for a GSTIN.
        Returns: (filed_count, late_count, filed_periods)
        """
        with self._driver.session() as neo_sess:
            result = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:FILED_GSTR1]->(g:GSTR1Filing)
                RETURN g.filing_period AS period,
                       g.filing_date   AS filing_date,
                       g.status        AS status
                ORDER BY g.filing_period
                """,
                gstin=gstin,
            )
            records = list(result)

        filed_count = 0
        late_count = 0
        filed_periods = []

        for rec in records:
            status = (rec["status"] or "").lower()
            if status == "filed":
                filed_count += 1
                filed_periods.append(rec["period"])

                # Check if late
                filing_date = rec["filing_date"]
                if filing_date and isinstance(filing_date, str):
                    try:
                        fd = datetime.strptime(filing_date, "%Y-%m-%d")
                        # Due by 11th of next month for GSTR-1
                        period = rec["period"]
                        if period and len(period) >= 6:
                            month = int(period[:2])
                            year = int(period[2:])
                            due_month = month + 1 if month < 12 else 1
                            due_year = year if month < 12 else year + 1
                            due_date = datetime(due_year, due_month, 11)
                            if fd > due_date:
                                late_count += 1
                    except (ValueError, TypeError):
                        pass
            elif status == "late":
                filed_count += 1
                late_count += 1
                filed_periods.append(rec["period"])

        return filed_count, late_count, filed_periods

    def _count_gstr3b_filings(self, gstin: str) -> Tuple[int, int, List[str]]:
        """
        Count GSTR-3B filings.
        Returns: (filed_count, late_count, filed_periods)
        """
        with self._driver.session() as neo_sess:
            result = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:FILED_GSTR3B]->(g:GSTR3BFiling)
                RETURN g.filing_period AS period,
                       g.status        AS status
                ORDER BY g.filing_period
                """,
                gstin=gstin,
            )
            records = list(result)

        filed_count = 0
        late_count = 0
        filed_periods = []

        for rec in records:
            status = (rec["status"] or "").lower()
            if status in ("filed", "late"):
                filed_count += 1
                filed_periods.append(rec["period"])
                if status == "late":
                    late_count += 1

        return filed_count, late_count, filed_periods

    def _compute_consecutive_misses(self, filed_periods: List[str], expected: int) -> int:
        """
        Calculate the longest streak of consecutive missing filing periods.
        """
        if not filed_periods:
            return expected

        # Generate all expected periods (assuming monthly, last 6 months)
        all_periods = set()
        now = datetime.now()
        for i in range(expected):
            dt = now - timedelta(days=30 * (i + 1))
            period = dt.strftime("%m%Y")
            all_periods.add(period)

        filed_set = set(filed_periods)
        missing = sorted(all_periods - filed_set)

        if not missing:
            return 0

        # Count longest consecutive run
        max_consec = 1
        current_consec = 1
        for i in range(1, len(missing)):
            # Simple approximation: consecutive if adjacent in sorted order
            current_consec += 1
            max_consec = max(max_consec, current_consec)

        return max_consec

    def score_vendor(self, gstin: str) -> FilingScoreBreakdown:
        """
        Compute complete filing frequency score for one vendor.

        Returns FilingScoreBreakdown with score 0–100.
        """
        breakdown = FilingScoreBreakdown(gstin=gstin)

        # ── GSTR-1 data ──
        g1_filed, g1_late, g1_periods = self._count_gstr1_filings(gstin)
        breakdown.gstr1_filed = g1_filed

        # ── GSTR-3B data ──
        g3b_filed, g3b_late, g3b_periods = self._count_gstr3b_filings(gstin)
        breakdown.gstr3b_filed = g3b_filed

        total_expected = EXPECTED_FILING_PERIODS * 2  # GSTR-1 + GSTR-3B
        total_filed = g1_filed + g3b_filed
        total_late = g1_late + g3b_late
        total_missing = (EXPECTED_FILING_PERIODS * 2) - total_filed
        breakdown.late_filings = total_late
        breakdown.missing_filings = max(total_missing, 0)

        # ── Sub-score A: On-time filing rate (40%) ──
        on_time = total_filed - total_late
        if total_filed > 0:
            ontime_rate = (on_time / total_filed) * 100
        else:
            ontime_rate = 0.0
        breakdown.ontime_rate_score = max(0.0, min(100.0, ontime_rate))

        # ── Sub-score B: GSTR-3B coverage (30%) ──
        g3b_coverage = (g3b_filed / EXPECTED_FILING_PERIODS) * 100
        breakdown.gstr3b_coverage_score = max(0.0, min(100.0, g3b_coverage))

        # ── Sub-score C: GSTR-1 coverage (20%) ──
        g1_coverage = (g1_filed / EXPECTED_FILING_PERIODS) * 100
        breakdown.gstr1_coverage_score = max(0.0, min(100.0, g1_coverage))

        # ── Sub-score D: Filing gap penalty (10%) ──
        all_periods = g1_periods + g3b_periods
        consec_misses = self._compute_consecutive_misses(
            all_periods, EXPECTED_FILING_PERIODS
        )
        breakdown.consecutive_misses = consec_misses

        # Each consecutive miss = 20 penalty points (max 100)
        gap_penalty = min(consec_misses * 20, 100)
        breakdown.filing_gap_penalty = gap_penalty
        gap_score = max(0.0, 100.0 - gap_penalty)

        # ── Final weighted score ──
        breakdown.final_score = (
            0.40 * breakdown.ontime_rate_score
            + 0.30 * breakdown.gstr3b_coverage_score
            + 0.20 * breakdown.gstr1_coverage_score
            + 0.10 * gap_score
        )

        # ── Generate detail messages ──
        if g3b_filed == 0:
            breakdown.details.append(
                f"CRITICAL: No GSTR-3B filed in {EXPECTED_FILING_PERIODS} months — "
                f"shell company indicator"
            )
        elif g3b_filed < EXPECTED_FILING_PERIODS:
            breakdown.details.append(
                f"GSTR-3B filed {g3b_filed}/{EXPECTED_FILING_PERIODS} months"
            )

        if g1_filed == 0:
            breakdown.details.append(
                f"No GSTR-1 filed in {EXPECTED_FILING_PERIODS} months"
            )
        elif g1_filed < EXPECTED_FILING_PERIODS:
            breakdown.details.append(
                f"GSTR-1 filed {g1_filed}/{EXPECTED_FILING_PERIODS} months"
            )

        if total_late > 0:
            breakdown.details.append(f"{total_late} returns filed late")

        if consec_misses >= 3:
            breakdown.details.append(
                f"{consec_misses} consecutive missed filings — high risk pattern"
            )

        if g1_filed > 0 and g3b_filed == 0:
            breakdown.details.append(
                "Filed GSTR-1 but NOT GSTR-3B — classic shell company pattern"
            )

        if not breakdown.details:
            breakdown.details.append("All returns filed on time — excellent compliance")

        return breakdown

    def score_all_vendors(self) -> List[FilingScoreBreakdown]:
        """Score all taxpayers in the database."""
        with self._driver.session() as neo_sess:
            result = neo_sess.run(
                "MATCH (t:Taxpayer) RETURN t.gstin AS gstin ORDER BY t.gstin"
            )
            gstins = [rec["gstin"] for rec in result]

        results = []
        for i, gstin in enumerate(gstins):
            score = self.score_vendor(gstin)
            results.append(score)
            if (i + 1) % 20 == 0:
                logger.info("Filing scores computed: %d/%d", i + 1, len(gstins))

        logger.info("Filing scores complete — %d vendors", len(results))
        return results