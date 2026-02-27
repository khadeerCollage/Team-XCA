"""
API routes for the Hybrid Mismatch Classifier.
==============================================
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classifier", tags=["Hybrid Mismatch Classifier"])

# Module-level cache for classifier state
_classifier = None
_latest_results = []
_training_metrics = {}

def get_classifier():
    """Lazy initialize the classifier."""
    global _classifier
    if _classifier is None:
        from .hybrid_classifier import HybridClassifier
        _classifier = HybridClassifier()
    return _classifier


class ClassifierConfig(BaseModel):
    pass


class TrainResponse(BaseModel):
    status: str
    rule_engine_stats: str
    xgboost_metrics: dict


class MismatchResult(BaseModel):
    invoice_number: str
    mismatch_type: str
    risk_level: str
    risk_probability: float
    rule_score: float
    ml_probability: float


class ClassifyResponse(BaseModel):
    status: str
    total_invoices: int
    results: List[MismatchResult]


@router.post("/train", response_model=TrainResponse)
def train_classifier():
    """Train the XGBoost component of the Hybrid Classifier."""
    try:
        classifier = get_classifier()
        metrics = classifier.train()
        global _training_metrics
        _training_metrics = metrics
        return TrainResponse(**metrics)
    except Exception as exc:
        logger.exception("Classifier training failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/classify", response_model=ClassifyResponse)
def classify_invoices():
    """Run the Hybrid Classifier to detect mismatches."""
    try:
        classifier = get_classifier()
        results = classifier.classify()
        
        global _latest_results
        _latest_results = results
        
        return ClassifyResponse(
            status="success",
            total_invoices=len(results),
            results=[MismatchResult(**r) for r in results]
        )
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/results", response_model=List[MismatchResult])
def get_classification_results(
    risk_level: Optional[str] = Query(None, description="Filter by HIGH, MEDIUM, LOW, CRITICAL")
):
    """Retrieve the latest classification results."""
    if not _latest_results:
        raise HTTPException(400, "No classification results available. Call POST /classify first.")
    
    filtered = _latest_results
    if risk_level:
        filtered = [r for r in filtered if r["risk_level"] == risk_level.upper()]
        
    return [MismatchResult(**r) for r in filtered]
