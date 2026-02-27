"""
Vendor Risk Engine — Master Orchestrator
==========================================
Combines all four scoring components and optional ML boosters
into a single 0–100 vendor compliance score.

Final Score =
    w1 × FilingScore      (25%)
  + w2 × DisputeScore     (25%)
  + w3 × NetworkScore     (25%)
  + w4 × PhysicalScore    (25%)
  ± ML adjustment         (optional boost/penalty)

Risk Bands:
    80–100 → SAFE       (green)
    50–79  → MODERATE   (yellow)
    30–49  → HIGH RISK  (orange)
    0–29   → FRAUD      (red)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .filing_scorer import FilingScorer, FilingScoreBreakdown
from .dispute_scorer import DisputeScorer, DisputeScoreBreakdown
from .network_scorer import NetworkScorer, NetworkScoreBreakdown
from .physical_scorer import PhysicalScorer, PhysicalScoreBreakdown

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"

# Default component weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "filing": 0.25,
    "dispute": 0.25,
    "network": 0.25,
    "physical": 0.25,
}

# Risk band thresholds
RISK_BANDS = [
    (80, 100, "SAFE",      "green"),
    (50,  79, "MODERATE",  "yellow"),
    (30,  49, "HIGH_RISK", "orange"),
    ( 0,  29, "FRAUD",     "red"),
]

# ML adjustment bounds
ML_MAX_BOOST = 10.0     # max points added for low ML risk
ML_MAX_PENALTY = 15.0   # max points deducted for high ML risk


@dataclass
class VendorRiskScore:
    """Complete vendor risk score with full breakdown."""
    gstin: str
    business_name: str = ""
    final_score: float = 0.0
    risk_category: str = "UNKNOWN"
    risk_color: str = "gray"

    # Component scores
    filing_score: float = 0.0
    dispute_score: float = 0.0
    network_score: float = 0.0
    physical_score: float = 0.0

    # ML adjustments
    gnn_fraud_probability: float = 0.0
    xgboost_risk_probability: float = 0.0
    ml_adjustment: float = 0.0

    # Pre-ML score (for transparency)
    base_score: float = 0.0

    # Component breakdowns
    filing_breakdown: dict = field(default_factory=dict)
    dispute_breakdown: dict = field(default_factory=dict)
    network_breakdown: dict = field(default_factory=dict)
    physical_breakdown: dict = field(default_factory=dict)

    # Aggregated explanation
    top_risk_factors: List[str] = field(default_factory=list)
    all_details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gstin": self.gstin,
            "business_name": self.business_name,
            "final_score": round(self.final_score, 2),
            "risk_category": self.risk_category,
            "risk_color": self.risk_color,
            "filing_score": round(self.filing_score, 2),
            "dispute_score": round(self.dispute_score, 2),
            "network_score": round(self.network_score, 2),
            "physical_score": round(self.physical_score, 2),
            "gnn_fraud_probability": round(self.gnn_fraud_probability, 4),
            "xgboost_risk_probability": round(self.xgboost_risk_probability, 4),
            "ml_adjustment": round(self.ml_adjustment, 2),
            "base_score": round(self.base_score, 2),
            "filing_breakdown": self.filing_breakdown,
            "dispute_breakdown": self.dispute_breakdown,
            "network_breakdown": self.network_breakdown,
            "physical_breakdown": self.physical_breakdown,
            "top_risk_factors": self.top_risk_factors,
            "all_details": self.all_details,
        }


class VendorRiskEngine:
    """
    Master risk-scoring engine that orchestrates all four component
    scorers and produces a final 0–100 vendor compliance score.
    """

    def __init__(self, driver=None, session=None, weights: dict = None):
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

        self._weights = weights or DEFAULT_WEIGHTS

        # Initialise sub-scorers
        self._filing_scorer = FilingScorer(
            driver=self._driver, session=self._session
        )
        self._dispute_scorer = DisputeScorer(
            driver=self._driver, session=self._session
        )
        self._network_scorer = NetworkScorer(driver=self._driver)
        self._physical_scorer = PhysicalScorer(
            driver=self._driver, session=self._session
        )

        # ML prediction caches
        self._gnn_predictions: Dict[str, float] = {}
        self._xgb_predictions: Dict[str, float] = {}

    def close(self):
        if self._owns_session and self._session:
            self._session.close()
        if self._owns_driver and self._driver:
            self._driver.close()

    def load_ml_predictions(self):
        """
        Attempt to load cached predictions from the GNN and XGBoost models.
        Silently skips if models haven't been trained yet.
        """
        # GNN predictions
        try:
            from backend.ml.predict import get_cached_predictions
            preds = get_cached_predictions()
            for p in preds:
                self._gnn_predictions[p["gstin"]] = p.get("fraud_probability", 0.0)
            logger.info("Loaded %d GNN predictions", len(self._gnn_predictions))
        except Exception as exc:
            logger.info("GNN predictions not available: %s", exc)

        # XGBoost predictions (invoice-level → aggregate to vendor)
        try:
            from backend.classifier.hybrid_classifier import get_cached_results
            results = get_cached_results()
            vendor_probs: Dict[str, List[float]] = {}
            for r in results:
                seller = r.get("seller_gstin", "")
                prob = r.get("ml_probability", 0.0)
                if seller:
                    vendor_probs.setdefault(seller, []).append(prob)

            for gstin, probs in vendor_probs.items():
                self._xgb_predictions[gstin] = sum(probs) / len(probs)
            logger.info("Loaded XGBoost agg predictions for %d vendors",
                        len(self._xgb_predictions))
        except Exception as exc:
            logger.info("XGBoost predictions not available: %s", exc)

    def _compute_ml_adjustment(self, gstin: str) -> float:
        """
        Compute ML-based score adjustment.

        If GNN says fraud_probability is HIGH → penalty
        If GNN says fraud_probability is LOW → small boost
        XGBoost mismatch probability adds additional signal.
        """
        gnn_prob = self._gnn_predictions.get(gstin, 0.5)
        xgb_prob = self._xgb_predictions.get(gstin, 0.5)

        # Combined ML risk (weighted)
        combined = 0.6 * gnn_prob + 0.4 * xgb_prob

        if combined < 0.2:
            # Low ML risk → small boost
            adjustment = ML_MAX_BOOST * (0.2 - combined) / 0.2
        elif combined > 0.5:
            # High ML risk → penalty
            adjustment = -ML_MAX_PENALTY * (combined - 0.5) / 0.5
        else:
            adjustment = 0.0

        return round(adjustment, 2)

    def _classify_risk(self, score: float) -> tuple:
        """Classify score into risk band. Returns (category, color)."""
        for low, high, category, color in RISK_BANDS:
            if low <= score <= high:
                return category, color
        return "FRAUD", "red"

    def _get_business_name(self, gstin: str) -> str:
        """Fetch business name from Neo4j."""
        with self._driver.session() as neo_sess:
            rec = neo_sess.run(
                "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t.name AS name",
                gstin=gstin,
            ).single()
        return rec["name"] if rec and rec["name"] else "Unknown"

    def _aggregate_risk_factors(
        self,
        filing: FilingScoreBreakdown,
        dispute: DisputeScoreBreakdown,
        network: NetworkScoreBreakdown,
        physical: PhysicalScoreBreakdown,
    ) -> List[str]:
        """
        Collect the top risk factors across all components,
        sorted by severity.
        """
        all_details = (
            filing.details + dispute.details
            + network.details + physical.details
        )

        # Prioritise items containing CRITICAL or high-risk keywords
        critical_keywords = [
            "CRITICAL", "shell company", "circular trading", "fraud",
            "missing IRN", "cancelled IRN", "no evidence", "0/",
        ]

        prioritised = []
        normal = []
        for detail in all_details:
            is_critical = any(kw.lower() in detail.lower() for kw in critical_keywords)
            if is_critical:
                prioritised.append(detail)
            else:
                normal.append(detail)

        return prioritised + normal

    # ────────────────────────────────────────
    # Score single vendor
    # ────────────────────────────────────────
    def score_vendor(self, gstin: str) -> VendorRiskScore:
        """
        Compute complete risk score for one vendor.
        """
        result = VendorRiskScore(gstin=gstin)
        result.business_name = self._get_business_name(gstin)

        # ── Component scores ──
        filing = self._filing_scorer.score_vendor(gstin)
        dispute = self._dispute_scorer.score_vendor(gstin)
        network = self._network_scorer.score_vendor(gstin)
        physical = self._physical_scorer.score_vendor(gstin)

        result.filing_score = filing.final_score
        result.dispute_score = dispute.final_score
        result.network_score = network.final_score
        result.physical_score = physical.final_score

        result.filing_breakdown = filing.to_dict()
        result.dispute_breakdown = dispute.to_dict()
        result.network_breakdown = network.to_dict()
        result.physical_breakdown = physical.to_dict()

        # ── Base score (weighted combination) ──
        base = (
            self._weights["filing"] * filing.final_score
            + self._weights["dispute"] * dispute.final_score
            + self._weights["network"] * network.final_score
            + self._weights["physical"] * physical.final_score
        )
        result.base_score = base

        # ── ML adjustment ──
        result.gnn_fraud_probability = self._gnn_predictions.get(gstin, 0.0)
        result.xgboost_risk_probability = self._xgb_predictions.get(gstin, 0.0)
        result.ml_adjustment = self._compute_ml_adjustment(gstin)

        # ── Final score ──
        final = base + result.ml_adjustment
        result.final_score = max(0.0, min(100.0, final))

        # ── Classification ──
        category, color = self._classify_risk(result.final_score)
        result.risk_category = category
        result.risk_color = color

        # ── Risk factors ──
        all_factors = self._aggregate_risk_factors(filing, dispute, network, physical)
        result.top_risk_factors = all_factors[:6]
        result.all_details = all_factors

        return result

    # ────────────────────────────────────────
    # Score all vendors
    # ────────────────────────────────────────
    def score_all_vendors(self) -> List[VendorRiskScore]:
        """
        Compute risk scores for every vendor in the database.
        """
        print("\n" + "╔" + "═" * 56 + "╗")
        print("║  Vendor Risk Scorer — Computing All Scores" + " " * 12 + "║")
        print("╚" + "═" * 56 + "╝\n")

        # Load ML predictions
        self.load_ml_predictions()

        # Get all GSTINs
        with self._driver.session() as neo_sess:
            result = neo_sess.run(
                "MATCH (t:Taxpayer) RETURN t.gstin AS gstin ORDER BY t.gstin"
            )
            gstins = [r["gstin"] for r in result]

        print(f"  Scoring {len(gstins)} vendors...\n")

        results = []
        for i, gstin in enumerate(gstins):
            score = self.score_vendor(gstin)
            results.append(score)

            if (i + 1) % 10 == 0 or (i + 1) == len(gstins):
                logger.info("Vendor scores computed: %d/%d", i + 1, len(gstins))

        # Sort by score ascending (riskiest first)
        results.sort(key=lambda r: r.final_score)

        # Print summary
        counts = {"SAFE": 0, "MODERATE": 0, "HIGH_RISK": 0, "FRAUD": 0}
        for r in results:
            cat = r.risk_category
            if cat in counts:
                counts[cat] += 1

        print("\n" + "=" * 56)
        print("  Vendor Risk Score — Summary")
        print("=" * 56)
        print(f"  Total vendors:  {len(results)}")
        print(f"  SAFE       (80-100) : {counts['SAFE']}")
        print(f"  MODERATE   (50-79)  : {counts['MODERATE']}")
        print(f"  HIGH RISK  (30-49)  : {counts['HIGH_RISK']}")
        print(f"  FRAUD      (0-29)   : {counts['FRAUD']}")
        print("-" * 56)

        # Print top 5 riskiest
        print("\n  Top 5 Riskiest Vendors:")
        for r in results[:5]:
            print(
                f"    {r.gstin} | {r.business_name:25s} | "
                f"Score: {r.final_score:5.1f} | {r.risk_category}"
            )

        # Print top 5 safest
        print("\n  Top 5 Safest Vendors:")
        for r in results[-5:]:
            print(
                f"    {r.gstin} | {r.business_name:25s} | "
                f"Score: {r.final_score:5.1f} | {r.risk_category}"
            )

        print("=" * 56 + "\n")

        return results


# ──────────────────────────────────────────────
# Module-level cache and convenience
# ──────────────────────────────────────────────
_cached_scores: List[dict] = []


def compute_all_vendor_scores(driver=None, session=None) -> List[dict]:
    """Run full vendor risk scoring and cache results."""
    global _cached_scores
    engine = VendorRiskEngine(driver=driver, session=session)
    try:
        results = engine.score_all_vendors()
        _cached_scores = [r.to_dict() for r in results]
        return _cached_scores
    finally:
        engine.close()


def get_cached_scores() -> List[dict]:
    return list(_cached_scores)