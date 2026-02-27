"""
Vendor Risk Scorer API Routes
================================
FastAPI router under ``/api/risk-scorer``.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"


def _get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _get_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(POSTGRES_URL)
    return sessionmaker(bind=engine)()


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class VendorScoreResponse(BaseModel):
    gstin: str
    business_name: str = ""
    final_score: float = 0.0
    risk_category: str = "UNKNOWN"
    risk_color: str = "gray"
    filing_score: float = 0.0
    dispute_score: float = 0.0
    network_score: float = 0.0
    physical_score: float = 0.0
    ml_adjustment: float = 0.0
    top_risk_factors: List[str] = []


class ComputeScoresResponse(BaseModel):
    status: str = "success"
    total_vendors: int = 0
    safe_count: int = 0
    moderate_count: int = 0
    high_risk_count: int = 0
    fraud_count: int = 0
    computation_time_seconds: float = 0.0
    scores: List[VendorScoreResponse] = []


class ITCDecisionResponse(BaseModel):
    buyer_gstin: str
    vendor_gstin: str
    vendor_name: str = ""
    vendor_score: float = 0.0
    itc_decision: str = ""
    itc_amount: float = 0.0
    eligible_amount: float = 0.0
    blocked_amount: float = 0.0
    decision_reason: str = ""


class ITCDecisionSummary(BaseModel):
    status: str = "success"
    total_pairs: int = 0
    auto_approve_count: int = 0
    manual_review_count: int = 0
    block_count: int = 0
    block_audit_count: int = 0
    total_itc: float = 0.0
    blocked_itc: float = 0.0
    decisions: List[ITCDecisionResponse] = []


class ExplanationResponse(BaseModel):
    gstin: str
    business_name: str = ""
    score: float = 0.0
    category: str = ""
    summary: str = ""
    score_breakdown: str = ""
    risk_factors: List[str] = []
    recommendation: str = ""
    itc_impact: str = ""
    full_report: str = ""


class RiskDistribution(BaseModel):
    safe: int = 0
    moderate: int = 0
    high_risk: int = 0
    fraud: int = 0
    total: int = 0


# ──────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────
_cached_scores: List[dict] = []
_cached_itc_decisions: List[dict] = []

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
router = APIRouter(prefix="/api/risk-scorer", tags=["Vendor Risk Scorer"])


# ━━━━━━━━━━ 1. COMPUTE ALL SCORES ━━━━━━━━━━
@router.post("/compute", response_model=ComputeScoresResponse)
def compute_vendor_scores():
    """Compute risk scores for all vendors."""
    global _cached_scores
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.risk_scorer.vendor_risk_engine import VendorRiskEngine

        t0 = time.time()
        engine = VendorRiskEngine(driver=driver, session=session)
        results = engine.score_all_vendors()
        elapsed = time.time() - t0

        _cached_scores = [r.to_dict() for r in results]

        counts = {"SAFE": 0, "MODERATE": 0, "HIGH_RISK": 0, "FRAUD": 0}
        for r in results:
            cat = r.risk_category
            if cat in counts:
                counts[cat] += 1

        return ComputeScoresResponse(
            status="success",
            total_vendors=len(results),
            safe_count=counts["SAFE"],
            moderate_count=counts["MODERATE"],
            high_risk_count=counts["HIGH_RISK"],
            fraud_count=counts["FRAUD"],
            computation_time_seconds=round(elapsed, 2),
            scores=[VendorScoreResponse(**s) for s in _cached_scores],
        )
    except Exception as exc:
        logger.exception("Vendor scoring failed")
        raise HTTPException(500, str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 2. GET ALL SCORES ━━━━━━━━━━
@router.get("/scores", response_model=List[VendorScoreResponse])
def get_scores(
    category: Optional[str] = Query(None, description="SAFE, MODERATE, HIGH_RISK, FRAUD"),
    sort_by: str = Query("score", description="score or name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return cached vendor scores with filters."""
    if not _cached_scores:
        raise HTTPException(400, "Run POST /api/risk-scorer/compute first")

    filtered = _cached_scores
    if category:
        filtered = [s for s in filtered if s["risk_category"] == category.upper()]

    if sort_by == "name":
        filtered = sorted(filtered, key=lambda s: s.get("business_name", ""))
    else:
        filtered = sorted(filtered, key=lambda s: s["final_score"])

    page = filtered[offset: offset + limit]
    return [VendorScoreResponse(**s) for s in page]


# ━━━━━━━━━━ 3. SINGLE VENDOR SCORE ━━━━━━━━━━
@router.get("/scores/{gstin}", response_model=VendorScoreResponse)
def get_vendor_score(gstin: str):
    """Get risk score for a specific vendor."""
    for s in _cached_scores:
        if s["gstin"] == gstin:
            return VendorScoreResponse(**s)

    # Compute on the fly
    driver = _get_driver()
    session = _get_session()
    try:
        from backend.risk_scorer.vendor_risk_engine import VendorRiskEngine
        engine = VendorRiskEngine(driver=driver, session=session)
        result = engine.score_vendor(gstin)
        return VendorScoreResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 4. ITC DECISIONS ━━━━━━━━━━
@router.post("/itc-decisions", response_model=ITCDecisionSummary)
def compute_itc_decisions():
    """Run auto ITC decisions for all buyer-vendor pairs."""
    global _cached_itc_decisions

    if not _cached_scores:
        raise HTTPException(400, "Compute vendor scores first (POST /api/risk-scorer/compute)")

    session = _get_session()
    try:
        from backend.risk_scorer.itc_decision import ITCDecisionEngine

        engine = ITCDecisionEngine(vendor_scores=_cached_scores, session=session)
        results = engine.decide_all()
        _cached_itc_decisions = [r.to_dict() for r in results]

        counts = {"AUTO_APPROVE": 0, "MANUAL_REVIEW": 0, "BLOCK": 0, "BLOCK_AND_AUDIT": 0}
        total_itc = 0.0
        blocked_itc = 0.0

        for r in results:
            counts[r.itc_decision] = counts.get(r.itc_decision, 0) + 1
            total_itc += r.itc_amount
            blocked_itc += r.blocked_amount

        return ITCDecisionSummary(
            status="success",
            total_pairs=len(results),
            auto_approve_count=counts.get("AUTO_APPROVE", 0),
            manual_review_count=counts.get("MANUAL_REVIEW", 0),
            block_count=counts.get("BLOCK", 0),
            block_audit_count=counts.get("BLOCK_AND_AUDIT", 0),
            total_itc=round(total_itc, 2),
            blocked_itc=round(blocked_itc, 2),
            decisions=[ITCDecisionResponse(**d) for d in _cached_itc_decisions],
        )
    except Exception as exc:
        logger.exception("ITC decision failed")
        raise HTTPException(500, str(exc))
    finally:
        session.close()


# ━━━━━━━━━━ 5. ITC DECISION FOR BUYER ━━━━━━━━━━
@router.get("/itc-decisions/{buyer_gstin}", response_model=List[ITCDecisionResponse])
def get_itc_decisions_for_buyer(buyer_gstin: str):
    """Get ITC decisions for a specific buyer."""
    if not _cached_itc_decisions:
        raise HTTPException(400, "Run POST /api/risk-scorer/itc-decisions first")

    filtered = [d for d in _cached_itc_decisions if d["buyer_gstin"] == buyer_gstin]
    if not filtered:
        raise HTTPException(404, f"No ITC decisions found for buyer {buyer_gstin}")
    return [ITCDecisionResponse(**d) for d in filtered]


# ━━━━━━━━━━ 6. VENDOR EXPLANATION ━━━━━━━━━━
@router.get("/explain/{gstin}", response_model=ExplanationResponse)
def explain_vendor_score(gstin: str):
    """Get full NL explanation for a vendor's risk score."""
    score_data = None
    for s in _cached_scores:
        if s["gstin"] == gstin:
            score_data = s
            break

    if score_data is None:
        # Compute on the fly
        driver = _get_driver()
        session = _get_session()
        try:
            from backend.risk_scorer.vendor_risk_engine import VendorRiskEngine
            engine = VendorRiskEngine(driver=driver, session=session)
            result = engine.score_vendor(gstin)
            score_data = result.to_dict()
        except Exception as exc:
            raise HTTPException(500, str(exc))
        finally:
            session.close()
            driver.close()

    from backend.risk_scorer.explainer import generate_vendor_explanation
    explanation = generate_vendor_explanation(score_data)
    return ExplanationResponse(**explanation)


# ━━━━━━━━━━ 7. RISK DISTRIBUTION ━━━━━━━━━━
@router.get("/distribution", response_model=RiskDistribution)
def get_risk_distribution():
    """Risk level distribution across all vendors."""
    if not _cached_scores:
        raise HTTPException(400, "Run POST /api/risk-scorer/compute first")

    counts = {"SAFE": 0, "MODERATE": 0, "HIGH_RISK": 0, "FRAUD": 0}
    for s in _cached_scores:
        cat = s.get("risk_category", "UNKNOWN")
        if cat in counts:
            counts[cat] += 1

    return RiskDistribution(
        safe=counts["SAFE"],
        moderate=counts["MODERATE"],
        high_risk=counts["HIGH_RISK"],
        fraud=counts["FRAUD"],
        total=len(_cached_scores),
    )


# ━━━━━━━━━━ 8. LEADERBOARD ━━━━━━━━━━
@router.get("/leaderboard")
def get_leaderboard(
    direction: str = Query("bottom", description="top (safest) or bottom (riskiest)"),
    n: int = Query(10, ge=1, le=50),
):
    """Top N safest or riskiest vendors."""
    if not _cached_scores:
        raise HTTPException(400, "Run POST /api/risk-scorer/compute first")

    sorted_scores = sorted(_cached_scores, key=lambda s: s["final_score"])

    if direction == "top":
        selected = sorted_scores[-n:]
        selected.reverse()
    else:
        selected = sorted_scores[:n]

    return {
        "direction": direction,
        "count": len(selected),
        "vendors": [
            {
                "rank": i + 1,
                "gstin": s["gstin"],
                "business_name": s.get("business_name", ""),
                "score": round(s["final_score"], 1),
                "category": s["risk_category"],
                "filing": round(s.get("filing_score", 0), 1),
                "dispute": round(s.get("dispute_score", 0), 1),
                "network": round(s.get("network_score", 0), 1),
                "physical": round(s.get("physical_score", 0), 1),
            }
            for i, s in enumerate(selected)
        ],
    }


# ━━━━━━━━━━ 9. DASHBOARD SUMMARY ━━━━━━━━━━
@router.get("/summary")
def get_dashboard_summary():
    """Comprehensive dashboard summary."""
    if not _cached_scores:
        raise HTTPException(400, "Run POST /api/risk-scorer/compute first")

    scores = [s["final_score"] for s in _cached_scores]
    avg_score = sum(scores) / max(len(scores), 1)

    counts = {"SAFE": 0, "MODERATE": 0, "HIGH_RISK": 0, "FRAUD": 0}
    for s in _cached_scores:
        cat = s.get("risk_category", "UNKNOWN")
        if cat in counts:
            counts[cat] += 1

    top_5_risky = sorted(_cached_scores, key=lambda s: s["final_score"])[:5]
    top_5_safe = sorted(_cached_scores, key=lambda s: s["final_score"], reverse=True)[:5]

    return {
        "total_vendors": len(_cached_scores),
        "average_score": round(avg_score, 1),
        "risk_distribution": counts,
        "compliance_rate": round(
            100 * counts["SAFE"] / max(len(_cached_scores), 1), 1
        ),
        "fraud_rate": round(
            100 * counts["FRAUD"] / max(len(_cached_scores), 1), 1
        ),
        "top_5_riskiest": [
            {"gstin": s["gstin"], "name": s.get("business_name", ""),
             "score": round(s["final_score"], 1), "category": s["risk_category"]}
            for s in top_5_risky
        ],
        "top_5_safest": [
            {"gstin": s["gstin"], "name": s.get("business_name", ""),
             "score": round(s["final_score"], 1), "category": s["risk_category"]}
            for s in top_5_safe
        ],
    }