"""
Classifier API Routes
=======================
FastAPI router for mismatch classification endpoints
under ``/api/classifier``.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _get_session():
    engine = create_engine(POSTGRES_URL)
    return sessionmaker(bind=engine)()


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class ClassificationResult(BaseModel):
    invoice_number: str
    seller_gstin: str = ""
    buyer_gstin: str = ""
    invoice_value: float = 0.0
    mismatch_type: str = "No Mismatch"
    all_mismatch_types: List[str] = []
    risk_level: str = "LOW"
    risk_probability: float = 0.0
    rule_score: float = 0.0
    ml_probability: float = 0.0
    root_cause: str = ""


class TrainClassifierResponse(BaseModel):
    status: str = "success"
    rule_engine_stats: dict = {}
    xgboost_metrics: dict = {}
    training_time_seconds: float = 0.0
    message: str = ""


class ClassifyResponse(BaseModel):
    status: str = "success"
    total_invoices: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    results: List[ClassificationResult] = []


class MismatchBreakdown(BaseModel):
    mismatch_type: str
    count: int
    avg_risk: float
    total_value_at_risk: float


class FinancialImpact(BaseModel):
    total_invoice_value: float = 0.0
    high_risk_value: float = 0.0
    medium_risk_value: float = 0.0
    estimated_itc_total: float = 0.0
    estimated_itc_at_risk: float = 0.0
    estimated_itc_needs_review: float = 0.0
    risk_percentage: float = 0.0


class EvaluationMetrics(BaseModel):
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix_details: dict = {}
    recall_analysis: dict = {}


# ──────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────
_cached_results: List[dict] = []
_cached_train_metrics: dict = {}

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
router = APIRouter(prefix="/api/classifier", tags=["Mismatch Classifier"])


# ━━━━━━━━━━ 1. TRAIN ━━━━━━━━━━
@router.post("/train", response_model=TrainClassifierResponse)
def train_classifier():
    """
    Train the hybrid classifier:
    1. Run rule engine → generate labels
    2. Extract features
    3. Train XGBoost
    """
    global _cached_train_metrics
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.classifier.hybrid_classifier import HybridClassifier

        t0 = time.time()
        hc = HybridClassifier(driver=driver, session=session)
        metrics = hc.train()
        elapsed = time.time() - t0

        _cached_train_metrics = metrics

        return TrainClassifierResponse(
            status="success",
            rule_engine_stats=metrics.get("rule_engine_stats", {}),
            xgboost_metrics=metrics.get("xgboost_metrics", {}),
            training_time_seconds=round(elapsed, 2),
            message="Hybrid classifier trained successfully",
        )
    except Exception as exc:
        logger.exception("Classifier training failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 2. CLASSIFY ALL ━━━━━━━━━━
@router.post("/classify", response_model=ClassifyResponse)
def classify_invoices():
    """Run hybrid classification on all invoices."""
    global _cached_results
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.classifier.hybrid_classifier import HybridClassifier

        hc = HybridClassifier(driver=driver, session=session)
        results = hc.classify()
        _cached_results = results

        # Count levels
        level_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in results:
            lv = r.get("risk_level", "LOW")
            if lv in level_counts:
                level_counts[lv] += 1

        return ClassifyResponse(
            status="success",
            total_invoices=len(results),
            critical_count=level_counts["CRITICAL"],
            high_count=level_counts["HIGH"],
            medium_count=level_counts["MEDIUM"],
            low_count=level_counts["LOW"],
            results=[ClassificationResult(**r) for r in results],
        )
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 3. GET RESULTS ━━━━━━━━━━
@router.get("/results", response_model=List[ClassificationResult])
def get_results(
    risk_level: Optional[str] = Query(None, description="Filter: CRITICAL, HIGH, MEDIUM, LOW"),
    mismatch_type: Optional[str] = Query(None, description="Filter by mismatch type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return cached classification results with filters."""
    if not _cached_results:
        raise HTTPException(400, "Run POST /api/classifier/classify first")

    filtered = _cached_results

    if risk_level:
        filtered = [r for r in filtered if r.get("risk_level") == risk_level.upper()]

    if mismatch_type:
        filtered = [r for r in filtered if r.get("mismatch_type") == mismatch_type]

    page = filtered[offset: offset + limit]
    return [ClassificationResult(**r) for r in page]


# ━━━━━━━━━━ 4. SINGLE INVOICE ━━━━━━━━━━
@router.get("/results/{invoice_number}", response_model=ClassificationResult)
def get_invoice_classification(invoice_number: str):
    """Get classification result for a specific invoice."""
    # Try cache first
    for r in _cached_results:
        if r.get("invoice_number") == invoice_number:
            return ClassificationResult(**r)

    # Compute on the fly
    driver = _get_driver()
    session = _get_session()
    try:
        from backend.classifier.hybrid_classifier import HybridClassifier

        hc = HybridClassifier(driver=driver, session=session)
        result = hc.classify_single(invoice_number)
        if result is None:
            raise HTTPException(404, f"Invoice {invoice_number} not found")
        return ClassificationResult(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 5. MISMATCH BREAKDOWN ━━━━━━━━━━
@router.get("/breakdown", response_model=List[MismatchBreakdown])
def get_mismatch_breakdown():
    """Breakdown of classification results by mismatch type."""
    if not _cached_results:
        raise HTTPException(400, "Run POST /api/classifier/classify first")

    from backend.classifier.evaluator import evaluate_by_mismatch_type

    stats = evaluate_by_mismatch_type(_cached_results)
    breakdown = []
    for mt, s in stats.items():
        breakdown.append(MismatchBreakdown(
            mismatch_type=mt,
            count=s["count"],
            avg_risk=s["avg_risk"],
            total_value_at_risk=s["total_value_at_risk"],
        ))
    return sorted(breakdown, key=lambda b: b.avg_risk, reverse=True)


# ━━━━━━━━━━ 6. FINANCIAL IMPACT ━━━━━━━━━━
@router.get("/financial-impact", response_model=FinancialImpact)
def get_financial_impact():
    """Estimate financial impact (ITC at risk) from mismatches."""
    if not _cached_results:
        raise HTTPException(400, "Run POST /api/classifier/classify first")

    from backend.classifier.evaluator import compute_financial_impact

    impact = compute_financial_impact(_cached_results)
    return FinancialImpact(**impact)


# ━━━━━━━━━━ 7. EVALUATION METRICS ━━━━━━━━━━
@router.get("/metrics", response_model=EvaluationMetrics)
def get_evaluation_metrics():
    """Return XGBoost evaluation metrics from last training."""
    if not _cached_train_metrics:
        raise HTTPException(400, "Run POST /api/classifier/train first")

    xgb_m = _cached_train_metrics.get("xgboost_metrics", {})
    return EvaluationMetrics(
        accuracy=xgb_m.get("accuracy", 0),
        precision=xgb_m.get("precision", 0),
        recall=xgb_m.get("recall", 0),
        f1_score=xgb_m.get("f1", 0),
        roc_auc=xgb_m.get("roc_auc", 0),
    )


# ━━━━━━━━━━ 8. RISK SUMMARY ━━━━━━━━━━
@router.get("/summary")
def get_classification_summary():
    """Dashboard summary of all classifications."""
    if not _cached_results:
        raise HTTPException(400, "Run POST /api/classifier/classify first")

    level_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    type_counts = {}
    total_risk_value = 0.0

    for r in _cached_results:
        lv = r.get("risk_level", "LOW")
        if lv in level_counts:
            level_counts[lv] += 1

        mt = r.get("mismatch_type", "Unknown")
        type_counts[mt] = type_counts.get(mt, 0) + 1

        if lv in ("HIGH", "CRITICAL"):
            total_risk_value += r.get("invoice_value", 0)

    return {
        "total_invoices": len(_cached_results),
        "risk_distribution": level_counts,
        "mismatch_distribution": type_counts,
        "total_value_at_risk": round(total_risk_value, 2),
        "estimated_itc_at_risk": round(total_risk_value * 0.18, 2),
        "critical_invoices": [
            r for r in _cached_results if r.get("risk_level") == "CRITICAL"
        ][:10],
    }