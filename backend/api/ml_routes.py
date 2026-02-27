"""
ML API Routes
==============
FastAPI router exposing GNN training, inference, and explainability
endpoints under ``/api/ml``.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config (mirrors backend/config.py)
# ──────────────────────────────────────────────
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="null1234"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _get_session():
    engine = create_engine(POSTGRES_URL)
    return sessionmaker(bind=engine)()


# ──────────────────────────────────────────────
# Pydantic response models
# ──────────────────────────────────────────────
class TrainResponse(BaseModel):
    status: str = "success"
    epochs_trained: int = 0
    optimal_threshold: float = 0.5
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    test_roc_auc: float = 0.0
    training_time_seconds: float = 0.0
    model_path: str = ""
    model_parameters: int = 0
    message: str = ""
    model_config = {"protected_namespaces": ()}


class FraudPrediction(BaseModel):
    gstin: str
    business_name: str
    fraud_probability: float
    risk_level: str
    confidence: str
    rank: int
    top_risk_factors: List[str] = []


class PredictResponse(BaseModel):
    status: str = "success"
    total_predictions: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    predictions: List[FraudPrediction] = []


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    direction: str
    raw_value: float
    interpretation: str


class NeighborInfluenceItem(BaseModel):
    gstin: str
    name: str
    influence_score: float
    risk_level: str
    compliance_rating: str = ""


class ITCImpact(BaseModel):
    total_invoices_issued: int = 0
    total_value: float = 0.0
    affected_buyers: int = 0
    estimated_itc_at_risk: float = 0.0


class ExplanationResponse(BaseModel):
    gstin: str
    business_name: str
    fraud_probability: float
    risk_level: str
    summary: str = ""
    top_risk_factors: List[str] = []
    feature_importance: List[dict] = []
    influential_neighbors: List[dict] = []
    subgraph: dict = {}
    recommendation: str = ""
    itc_impact: dict = {}


class ModelMetrics(BaseModel):
    is_trained: bool = False
    epochs_trained: int = 0
    optimal_threshold: float = 0.5
    test_metrics: dict = {}
    training_history: List[dict] = []
    model_parameters: int = 0
    training_time_seconds: float = 0.0
    last_trained: str = ""
    model_config = {"protected_namespaces": ()}


class RiskDistribution(BaseModel):
    total: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    high_percentage: float = 0.0
    medium_percentage: float = 0.0
    low_percentage: float = 0.0


# ──────────────────────────────────────────────
# Module-level caches
# ──────────────────────────────────────────────
_train_results: dict = {}
_predict_results: List[dict] = []

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])


# ━━━━━━━━━━ 1. TRAIN ━━━━━━━━━━
@router.post("/train", response_model=TrainResponse)
def train_gnn():
    """Train the GNN fraud-detection model from scratch."""
    global _train_results
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.ml.train import train_and_evaluate

        t0 = time.time()
        results = train_and_evaluate(driver, session)
        elapsed = time.time() - t0

        _train_results = results
        _train_results["last_trained"] = datetime.utcnow().isoformat()

        test = results.get("test_metrics", {})
        return TrainResponse(
            status="success",
            epochs_trained=results.get("best_epoch", 0),
            optimal_threshold=results.get("optimal_threshold", 0.5),
            test_accuracy=test.get("accuracy", 0),
            test_precision=test.get("precision", 0),
            test_recall=test.get("recall", 0),
            test_f1=test.get("f1", 0),
            test_roc_auc=test.get("roc_auc", 0),
            training_time_seconds=round(elapsed, 2),
            model_path=results.get("model_path", ""),
            model_parameters=results.get("model_parameters", 0),
            message="Model trained successfully",
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 2. PREDICT ━━━━━━━━━━
@router.post("/predict", response_model=PredictResponse)
def run_predictions():
    """Run GNN inference on all taxpayer nodes."""
    global _predict_results
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.ml.predict import predict_all, get_risk_distribution

        preds = predict_all(driver, session)
        _predict_results = preds
        dist = get_risk_distribution(preds)

        return PredictResponse(
            status="success",
            total_predictions=dist["total"],
            high_risk_count=dist["high"],
            medium_risk_count=dist["medium"],
            low_risk_count=dist["low"],
            predictions=[FraudPrediction(**p) for p in preds],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 3. GET PREDICTIONS ━━━━━━━━━━
@router.get("/predictions", response_model=List[FraudPrediction])
def get_predictions(
    risk_level: Optional[str] = Query(None, description="Filter: HIGH, MEDIUM, LOW"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return cached predictions (run POST /predict first)."""
    if not _predict_results:
        raise HTTPException(
            status_code=400,
            detail="No predictions available. Call POST /api/ml/predict first.",
        )
    filtered = _predict_results
    if risk_level:
        filtered = [p for p in filtered if p["risk_level"] == risk_level.upper()]
    page = filtered[offset : offset + limit]
    return [FraudPrediction(**p) for p in page]


# ━━━━━━━━━━ 4. GET SINGLE PREDICTION ━━━━━━━━━━
@router.get("/predictions/{gstin}", response_model=FraudPrediction)
def get_prediction_by_gstin(gstin: str):
    """Return prediction for a specific GSTIN."""
    if not _predict_results:
        raise HTTPException(400, "Run POST /api/ml/predict first.")
    for p in _predict_results:
        if p["gstin"] == gstin:
            return FraudPrediction(**p)
    raise HTTPException(404, f"GSTIN {gstin} not found in predictions.")


# ━━━━━━━━━━ 5. EXPLAIN ━━━━━━━━━━
@router.get("/explain/{gstin}", response_model=ExplanationResponse)
def explain_prediction(gstin: str):
    """Full explainability report for a GSTIN."""
    driver = _get_driver()
    session = _get_session()

    try:
        from backend.ml.explainer import generate_explanation

        result = generate_explanation(gstin, driver, session)
        if "error" in result:
            raise HTTPException(404, result["error"])
        return ExplanationResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Explanation failed for %s", gstin)
        raise HTTPException(500, str(exc))
    finally:
        session.close()
        driver.close()


# ━━━━━━━━━━ 6. MODEL METRICS ━━━━━━━━━━
@router.get("/metrics", response_model=ModelMetrics)
def get_model_metrics():
    """Return metrics from the last training run."""
    if not _train_results:
        return ModelMetrics(is_trained=False, message="Model not yet trained.")
    test = _train_results.get("test_metrics", {})
    return ModelMetrics(
        is_trained=True,
        epochs_trained=_train_results.get("best_epoch", 0),
        optimal_threshold=_train_results.get("optimal_threshold", 0.5),
        test_metrics=test,
        training_history=_train_results.get("training_history", []),
        model_parameters=_train_results.get("model_parameters", 0),
        training_time_seconds=_train_results.get("training_time_seconds", 0),
        last_trained=_train_results.get("last_trained", ""),
    )


# ━━━━━━━━━━ 7. RISK DISTRIBUTION ━━━━━━━━━━
@router.get("/risk-distribution", response_model=RiskDistribution)
def get_risk_dist():
    """Count of HIGH / MEDIUM / LOW from ML predictions."""
    if not _predict_results:
        raise HTTPException(400, "Run POST /api/ml/predict first.")
    from backend.ml.predict import get_risk_distribution

    dist = get_risk_distribution(_predict_results)
    return RiskDistribution(**dist)


# ━━━━━━━━━━ 8. EMBEDDINGS ━━━━━━━━━━
@router.get("/embeddings")
def get_embeddings():
    """Return 16-D node embeddings for every taxpayer (for t-SNE vis)."""
    driver = _get_driver()
    session = _get_session()
    try:
        from backend.ml.predict import get_node_embeddings

        emb = get_node_embeddings(driver, session)
        return {"status": "success", "count": len(emb), "embeddings": emb}
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Embedding extraction failed")
        raise HTTPException(500, str(exc))
    finally:
        session.close()
        driver.close()