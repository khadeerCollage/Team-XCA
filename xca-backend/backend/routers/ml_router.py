"""
ML Prediction Router
=====================
Extracts features from Neo4j knowledge graph, trains XGBoost model,
and predicts vendor risk scores using real ML inference.

Flow:
    Neo4j graph → Feature extraction (14 features per vendor)
    → XGBoost prediction → 4-component scoring
    → Final risk score per vendor

Endpoints:
    POST /ml/train    — Train XGBoost on graph-extracted features
    POST /ml/predict  — Run ML inference for all vendors
    GET  /ml/scores   — Return cached ML predictions (leaderboard)
    GET  /ml/scores/{gstin} — Single vendor ML prediction
"""

import logging
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.neo4j_connection import get_driver

logger = logging.getLogger(__name__)

# ── Router ────────────────────────────────────────────────────
ml_router = APIRouter(prefix="/ml", tags=["ML Predictions"])

# ── Model paths ───────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "..", "ml_model.pkl")
SCALER_PATH = os.path.join(_BASE_DIR, "..", "ml_scaler.pkl")
IMPORTANCES_PATH = os.path.join(_BASE_DIR, "..", "ml_importances.pkl")

# ── Module-level caches ──────────────────────────────────────
_cached_predictions: List[dict] = []
_cached_features: List[dict] = []
_feature_importances: Dict[str, float] = {}
_model_trained: bool = False

# ── Feature names (14 features extracted from Neo4j) ─────────
FEATURE_NAMES = [
    "gstr1_coverage",
    "gstr2b_coverage",
    "mismatch_rate",
    "itc_overclaim_ratio",
    "irn_coverage",
    "eway_coverage",
    "out_degree_norm",
    "in_degree_norm",
    "partner_count_norm",
    "circular_trade_flag",
    "risky_neighbor_ratio",
    "avg_invoice_value_norm",
    "turnover_norm",
    "initial_risk_score",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic response models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VendorMLPrediction(BaseModel):
    gstin: str
    label: str
    state: str = ""
    score: float
    risk_level: str
    # Component scores (0-100)
    filing_score: float = 0.0
    mismatch_score: float = 0.0
    network_score: float = 0.0
    physical_score: float = 0.0
    # Raw metrics for breakdown bars
    mismatchRate: float = 0.0
    delay: float = 0.0
    overclaim: float = 0.0
    circular: float = 0.0
    itc: float = 0.0
    tx: int = 0
    # ML details
    ml_probability: float = 0.0
    top_risk_factors: List[str] = []


class PredictResponse(BaseModel):
    status: str = "success"
    total_vendors: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    model_used: str = "xgboost"
    feature_importances: Dict[str, float] = {}
    predictions: List[VendorMLPrediction] = []


class TrainResponse(BaseModel):
    status: str = "success"
    message: str = ""
    model_type: str = "XGBClassifier"
    feature_count: int = 0
    vendor_count: int = 0
    training_time_seconds: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    feature_importances: Dict[str, float] = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature extraction from Neo4j
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_all_features(driver) -> List[dict]:
    """
    Extract 14 numerical features per vendor from the Neo4j graph.
    All queries match the actual schema from 03_load_to_neo4j.py.
    """
    with driver.session() as session:
        # Get all taxpayers
        taxpayers = session.run("""
            MATCH (t:Taxpayer)
            RETURN t.gstin AS gstin, t.name AS name, t.state AS state,
                   t.turnover AS turnover, t.risk_score AS risk_score,
                   t.business_type AS btype
            ORDER BY t.gstin
        """).data()

    features = []
    for tp in taxpayers:
        feat = _extract_vendor_features(driver, tp)
        features.append(feat)

    return features


def _extract_vendor_features(driver, tp_props: dict) -> dict:
    """Extract all 14 features + metadata for one vendor from Neo4j."""
    gstin = tp_props["gstin"]

    with driver.session() as session:
        # ── 1. Invoice stats ──
        inv_stats = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            RETURN count(inv) AS total_inv,
                   COALESCE(sum(inv.taxable_value), 0) AS total_value
        """, gstin=gstin).single()
        total_invoices = inv_stats["total_inv"] or 0
        total_value = float(inv_stats["total_value"] or 0)

        # ── 2. Received invoice stats (for ITC) ──
        recv_stats = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:RECEIVED]->(inv:Invoice)
            RETURN count(inv) AS recv_inv,
                   COALESCE(sum(inv.taxable_value), 0) AS recv_value
        """, gstin=gstin).single()
        recv_invoices = recv_stats["recv_inv"] or 0
        recv_value = float(recv_stats["recv_value"] or 0)

        # ── 3. GSTR-1 coverage (invoices reported in GSTR-1 returns) ──
        gstr1_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            OPTIONAL MATCH (inv)-[:REPORTED_IN]->(r:Return {return_type: "GSTR-1"})
            RETURN count(inv) AS total, count(r) AS reported
        """, gstin=gstin).single()
        gstr1_coverage = (gstr1_data["reported"] or 0) / max(gstr1_data["total"] or 1, 1)

        # ── 4. GSTR-2B coverage (invoices reflected in GSTR-2B) ──
        gstr2b_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            OPTIONAL MATCH (inv)-[:REFLECTED_IN]->(r:Return {return_type: "GSTR-2B"})
            RETURN count(inv) AS total, count(r) AS reflected
        """, gstin=gstin).single()
        gstr2b_coverage = (gstr2b_data["reflected"] or 0) / max(gstr2b_data["total"] or 1, 1)

        # ── 5. Mismatch rate ──
        mismatch_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            WITH count(inv) AS total,
                 sum(CASE WHEN inv.has_mismatch = true THEN 1 ELSE 0 END) AS mismatched,
                 sum(CASE WHEN inv.has_mismatch = true THEN inv.taxable_value ELSE 0 END) AS mismatch_value
            RETURN total, mismatched, mismatch_value
        """, gstin=gstin).single()
        mismatch_count = mismatch_data["mismatched"] or 0
        mismatch_rate = mismatch_count / max(mismatch_data["total"] or 1, 1)
        mismatch_value = float(mismatch_data["mismatch_value"] or 0)

        # ── 6. ITC overclaim ratio ──
        itc_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:RECEIVED]->(inv:Invoice)
            OPTIONAL MATCH (inv)-[:REFLECTED_IN]->(r2b:Return {return_type: "GSTR-2B"})
            RETURN COALESCE(sum(inv.taxable_value), 0) AS claimed_base,
                   COALESCE(sum(CASE WHEN r2b IS NOT NULL THEN inv.taxable_value ELSE 0 END), 0) AS eligible_base
        """, gstin=gstin).single()
        claimed = float(itc_data["claimed_base"] or 0)
        eligible = float(itc_data["eligible_base"] or 0)
        itc_overclaim = max(0, (claimed - eligible)) / max(claimed, 1)

        # ── 7. IRN coverage ──
        irn_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            OPTIONAL MATCH (inv)-[:HAS_IRN]->(irn:IRN)
            RETURN count(inv) AS total, count(irn) AS with_irn
        """, gstin=gstin).single()
        irn_coverage = (irn_data["with_irn"] or 0) / max(irn_data["total"] or 1, 1)

        # ── 8. e-Way Bill coverage ──
        eway_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            OPTIONAL MATCH (inv)-[:COVERED_BY]->(ewb:EWayBill)
            RETURN count(inv) AS total, count(ewb) AS with_eway
        """, gstin=gstin).single()
        eway_coverage = (eway_data["with_eway"] or 0) / max(eway_data["total"] or 1, 1)

        # ── 9. Degree centrality (out = issued, in = received) ──
        # out_degree already from total_invoices, in_degree from recv_invoices

        # ── 10. Unique partners (TRANSACTS_WITH) ──
        partner_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:TRANSACTS_WITH]-(n:Taxpayer)
            RETURN count(DISTINCT n) AS partners
        """, gstin=gstin).single()
        partners = partner_data["partners"] or 0

        # ── 11. Circular trade detection ──
        try:
            cycle_data = session.run("""
                MATCH path = (t:Taxpayer {gstin: $gstin})-[:TRANSACTS_WITH*2..4]->(t)
                RETURN count(path) AS cycles
            """, gstin=gstin).single()
            cycle_count = cycle_data["cycles"] or 0
        except Exception:
            cycle_count = 0
        circular_flag = min(cycle_count / 3.0, 1.0)

        # ── 12. Risky neighbor ratio ──
        risky_data = session.run("""
            MATCH (t:Taxpayer {gstin: $gstin})-[:TRANSACTS_WITH]-(n:Taxpayer)
            WITH count(n) AS total_n,
                 sum(CASE WHEN n.risk_level = 'HIGH' THEN 1 ELSE 0 END) AS risky
            RETURN total_n, risky,
                   CASE WHEN total_n > 0
                        THEN toFloat(risky) / total_n
                        ELSE 0.0 END AS risky_ratio
        """, gstin=gstin).single()
        risky_ratio = float(risky_data["risky_ratio"] or 0)

    # ── Derived metrics ──
    avg_value = total_value / max(total_invoices, 1)
    itc_at_risk = mismatch_value * 0.18  # GST component
    turnover = float(tp_props.get("turnover") or 0)
    initial_risk = float(tp_props.get("risk_score") or 0)
    filing_gap_days = max(0, (1.0 - gstr1_coverage) * 30)  # estimate

    return {
        "gstin": gstin,
        "name": tp_props.get("name") or "Unknown",
        "state": tp_props.get("state") or "",
        "btype": tp_props.get("btype") or "Unknown",
        # ── Raw metrics ──
        "total_invoices": total_invoices,
        "recv_invoices": recv_invoices,
        "total_value": round(total_value, 2),
        "recv_value": round(recv_value, 2),
        "mismatch_count": mismatch_count,
        "mismatch_value": round(mismatch_value, 2),
        "itc_at_risk": round(itc_at_risk, 2),
        "partners": partners,
        "cycle_count": cycle_count,
        "filing_gap_days": round(filing_gap_days, 1),
        "turnover": turnover,
        # ── 14 normalised features (for model input) ──
        "gstr1_coverage": round(gstr1_coverage, 4),
        "gstr2b_coverage": round(gstr2b_coverage, 4),
        "mismatch_rate": round(mismatch_rate, 4),
        "itc_overclaim_ratio": round(itc_overclaim, 4),
        "irn_coverage": round(irn_coverage, 4),
        "eway_coverage": round(eway_coverage, 4),
        "out_degree_norm": round(min(total_invoices / 20.0, 1.0), 4),
        "in_degree_norm": round(min(recv_invoices / 20.0, 1.0), 4),
        "partner_count_norm": round(min(partners / 10.0, 1.0), 4),
        "circular_trade_flag": round(circular_flag, 4),
        "risky_neighbor_ratio": round(risky_ratio, 4),
        "avg_invoice_value_norm": round(min(avg_value / 100000.0, 1.0), 4),
        "turnover_norm": round(min(turnover / 1e8, 1.0), 4),
        "initial_risk_score": round(initial_risk, 4),
    }


def _features_to_vector(feat: dict) -> np.ndarray:
    """Convert feature dict to numpy array (14 features)."""
    return np.array([feat[fn] for fn in FEATURE_NAMES], dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component scores (for explainability / breakdown)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_component_scores(feat: dict) -> dict:
    """
    Compute 4 component scores (0-100) from extracted features.
    Higher = safer / more compliant.
    """
    # Filing Score: GSTR-1 + GSTR-2B coverage
    filing = (
        0.50 * feat["gstr1_coverage"] * 100
        + 0.50 * feat["gstr2b_coverage"] * 100
    )

    # Mismatch / Dispute Score: lower mismatch_rate + lower overclaim = safer
    mismatch = (
        0.60 * (1.0 - feat["mismatch_rate"]) * 100
        + 0.40 * (1.0 - feat["itc_overclaim_ratio"]) * 100
    )

    # Network Score: no circular trades + low risky neighbors = safer
    network = (
        0.40 * (1.0 - feat["circular_trade_flag"]) * 100
        + 0.35 * (1.0 - feat["risky_neighbor_ratio"]) * 100
        + 0.25 * min(feat["partners"] / 5.0, 1.0) * 100
    )

    # Physical Score: IRN + eWay + payment verified = safer
    physical = (
        0.50 * feat["irn_coverage"] * 100
        + 0.50 * feat["eway_coverage"] * 100
    )

    return {
        "filing_score": round(max(0, min(100, filing)), 2),
        "mismatch_score": round(max(0, min(100, mismatch)), 2),
        "network_score": round(max(0, min(100, network)), 2),
        "physical_score": round(max(0, min(100, physical)), 2),
    }


def _generate_risk_factors(feat: dict, probability: float) -> List[str]:
    """Generate human-readable risk factors from features."""
    factors = []

    if feat["mismatch_rate"] > 0.3:
        factors.append(
            f"High mismatch rate: {feat['mismatch_rate']*100:.0f}% of invoices have discrepancies"
        )
    if feat["itc_overclaim_ratio"] > 0.1:
        factors.append(
            f"ITC overclaim detected: {feat['itc_overclaim_ratio']*100:.0f}% above eligible credit"
        )
    if feat["circular_trade_flag"] > 0.3:
        factors.append(
            f"Circular trading detected: {feat['cycle_count']} transaction cycles found"
        )
    if feat["risky_neighbor_ratio"] > 0.3:
        factors.append(
            f"Connected to high-risk entities: {feat['risky_neighbor_ratio']*100:.0f}% risky neighbors"
        )
    if feat["irn_coverage"] < 0.5:
        factors.append(
            f"Low IRN coverage: only {feat['irn_coverage']*100:.0f}% of invoices have valid IRN"
        )
    if feat["eway_coverage"] < 0.5:
        factors.append(
            f"Missing e-Way Bills: only {feat['eway_coverage']*100:.0f}% coverage"
        )
    if feat["gstr1_coverage"] < 0.8:
        factors.append(
            f"GSTR-1 filing gaps: {feat['gstr1_coverage']*100:.0f}% coverage"
        )
    if feat["gstr2b_coverage"] < 0.8:
        factors.append(
            f"GSTR-2B reflection gaps: {feat['gstr2b_coverage']*100:.0f}% coverage"
        )

    if not factors:
        if probability < 0.3:
            factors.append("No significant risk indicators — vendor appears compliant")
        else:
            factors.append("Risk primarily from network context and ML pattern detection")

    return factors[:6]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XGBoost training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _train_model(features: List[dict]) -> dict:
    """
    Train XGBoost on graph-extracted features.

    Labels are derived from component scores + initial graph risk:
    - Vendors with poor component scores → label 1 (risky)
    - Vendors with good component scores → label 0 (safe)

    The model learns NON-LINEAR feature interactions that a
    simple weighted sum cannot capture.
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    global _model_trained, _feature_importances

    # Build feature matrix
    X = np.array([_features_to_vector(f) for f in features], dtype=np.float32)

    # Generate labels from component scores + initial risk
    labels = []
    for feat in features:
        comp = _compute_component_scores(feat)
        avg_score = (
            comp["filing_score"] + comp["mismatch_score"]
            + comp["network_score"] + comp["physical_score"]
        ) / 4.0
        # Combine with initial graph risk (data-driven label)
        initial_risk = feat["initial_risk_score"]
        # Weighted combination: low component score + high initial risk = risky
        combined_signal = (1.0 - avg_score / 100.0) * 0.5 + initial_risk * 0.5
        label = 1 if combined_signal > 0.4 else 0
        labels.append(label)

    y = np.array(labels, dtype=np.int32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost
    n_samples = len(X)
    model = xgb.XGBClassifier(
        n_estimators=min(100, max(10, n_samples * 2)),
        max_depth=min(4, max(2, n_samples // 5)),
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        reg_alpha=0.5,
        reg_lambda=1.0,
    )
    model.fit(X_scaled, y)

    # Extract feature importances (LEARNED by the model)
    importances = model.feature_importances_
    _feature_importances = {
        FEATURE_NAMES[i]: round(float(importances[i]), 4)
        for i in range(len(FEATURE_NAMES))
    }

    # Compute training metrics
    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)

    # Save model artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(IMPORTANCES_PATH, "wb") as f:
        pickle.dump(_feature_importances, f)

    _model_trained = True

    logger.info(
        "XGBoost trained — %d vendors, accuracy=%.2f, f1=%.2f",
        n_samples, acc, f1,
    )

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "feature_importances": _feature_importances,
        "n_samples": n_samples,
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prediction pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _predict_all(features: List[dict]) -> List[dict]:
    """
    Run ML inference for all vendors.

    If trained model exists: XGBoost predicts fraud probability.
    If not: component scores are used as fallback (still data-driven).
    """
    global _cached_predictions, _cached_features
    _cached_features = features

    model = None
    scaler = None
    use_ml = False

    # Try to load trained model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            use_ml = True
            logger.info("Loaded trained XGBoost model for inference")
        except Exception as exc:
            logger.warning("Could not load model: %s", exc)

    # Load feature importances
    if os.path.exists(IMPORTANCES_PATH):
        try:
            with open(IMPORTANCES_PATH, "rb") as f:
                global _feature_importances
                _feature_importances = pickle.load(f)
        except Exception:
            pass

    predictions = []

    # Build feature matrix
    X = np.array([_features_to_vector(f) for f in features], dtype=np.float32)

    # ML prediction
    if use_ml and model is not None and scaler is not None:
        X_scaled = scaler.transform(X)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # P(fraud)
    else:
        probabilities = None

    for i, feat in enumerate(features):
        comp = _compute_component_scores(feat)

        if probabilities is not None:
            # ML-predicted fraud probability
            ml_prob = float(probabilities[i])
        else:
            # Fallback: derive from component scores
            avg_comp = (
                comp["filing_score"] + comp["mismatch_score"]
                + comp["network_score"] + comp["physical_score"]
            ) / 4.0
            ml_prob = 1.0 - (avg_comp / 100.0)

        # Final risk score (0-1, higher = riskier)
        # Blend ML probability with component-derived risk
        comp_risk = 1.0 - (
            (comp["filing_score"] + comp["mismatch_score"]
             + comp["network_score"] + comp["physical_score"]) / 400.0
        )
        if use_ml:
            # 70% ML, 30% component (ML has primary weight)
            final_score = 0.70 * ml_prob + 0.30 * comp_risk
        else:
            final_score = comp_risk

        final_score = round(max(0.0, min(1.0, final_score)), 4)

        # Risk level classification
        if final_score > 0.6:
            risk_level = "HIGH RISK"
        elif final_score > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW RISK"

        risk_factors = _generate_risk_factors(feat, final_score)

        predictions.append({
            "gstin": feat["gstin"],
            "label": feat["name"],
            "state": feat["state"],
            "score": final_score,
            "risk_level": risk_level,
            "filing_score": comp["filing_score"],
            "mismatch_score": comp["mismatch_score"],
            "network_score": comp["network_score"],
            "physical_score": comp["physical_score"],
            "mismatchRate": feat["mismatch_rate"],
            "delay": feat["filing_gap_days"],
            "overclaim": feat["itc_overclaim_ratio"],
            "circular": feat["circular_trade_flag"],
            "itc": feat["itc_at_risk"],
            "tx": feat["total_invoices"],
            "ml_probability": round(ml_prob, 4),
            "top_risk_factors": risk_factors,
        })

    # Sort by score descending (riskiest first)
    predictions.sort(key=lambda p: p["score"], reverse=True)

    # Assign rank
    for rank, p in enumerate(predictions, 1):
        p["id"] = rank

    _cached_predictions = predictions
    return predictions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@ml_router.post("/train", response_model=TrainResponse)
def train_model():
    """
    Train XGBoost model on features extracted from Neo4j graph.

    1. Queries Neo4j for 14 features per vendor
    2. Generates training labels from component scores + graph risk
    3. Trains XGBoost classifier
    4. Saves model checkpoint + feature importances
    """
    driver = get_driver()
    try:
        t0 = time.time()

        logger.info("Extracting features from Neo4j...")
        features = _extract_all_features(driver)
        if not features:
            raise HTTPException(400, "No taxpayer data in Neo4j. Run pipeline first.")

        logger.info("Training XGBoost on %d vendors...", len(features))
        metrics = _train_model(features)
        elapsed = time.time() - t0

        return TrainResponse(
            status="success",
            message=f"XGBoost trained on {len(features)} vendors — "
                    f"accuracy: {metrics['accuracy']:.2%}, "
                    f"F1: {metrics['f1_score']:.2%}",
            model_type="XGBClassifier",
            feature_count=len(FEATURE_NAMES),
            vendor_count=len(features),
            training_time_seconds=round(elapsed, 3),
            accuracy=metrics["accuracy"],
            f1_score=metrics["f1_score"],
            feature_importances=metrics["feature_importances"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ML training failed")
        raise HTTPException(500, detail=str(exc))


@ml_router.post("/predict", response_model=PredictResponse)
def run_predictions():
    """
    Run ML inference for all vendors.

    1. Extracts features from Neo4j
    2. If trained model exists → XGBoost predicts fraud probability
    3. Otherwise → component scores used as fallback
    4. Returns predictions sorted by risk (highest first)
    """
    driver = get_driver()
    try:
        t0 = time.time()

        features = _extract_all_features(driver)
        if not features:
            raise HTTPException(400, "No taxpayer data in Neo4j. Run pipeline first.")

        predictions = _predict_all(features)
        elapsed = time.time() - t0

        high = sum(1 for p in predictions if p["risk_level"] == "HIGH RISK")
        med = sum(1 for p in predictions if p["risk_level"] == "MEDIUM")
        low = sum(1 for p in predictions if p["risk_level"] == "LOW RISK")

        model_name = "xgboost" if os.path.exists(MODEL_PATH) else "component_scoring"

        logger.info(
            "ML predictions: %d vendors (H=%d, M=%d, L=%d) in %.2fs [%s]",
            len(predictions), high, med, low, elapsed, model_name,
        )

        return PredictResponse(
            status="success",
            total_vendors=len(predictions),
            high_risk_count=high,
            medium_risk_count=med,
            low_risk_count=low,
            model_used=model_name,
            feature_importances=_feature_importances,
            predictions=[VendorMLPrediction(**p) for p in predictions],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ML prediction failed")
        raise HTTPException(500, detail=str(exc))


@ml_router.get("/scores", response_model=List[VendorMLPrediction])
def get_scores(
    risk_level: Optional[str] = Query(None, description="Filter: HIGH RISK, MEDIUM, LOW RISK"),
    sort_by: str = Query("score", description="Sort field: score, mismatchRate, itc, delay"),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Return cached ML predictions (call POST /ml/predict first).
    """
    if not _cached_predictions:
        raise HTTPException(
            400, "No ML predictions available. Call POST /ml/predict first."
        )

    filtered = _cached_predictions
    if risk_level:
        filtered = [p for p in filtered if p["risk_level"] == risk_level]

    # Sort
    if sort_by == "mismatchRate":
        filtered = sorted(filtered, key=lambda p: p["mismatchRate"], reverse=True)
    elif sort_by == "itc":
        filtered = sorted(filtered, key=lambda p: p["itc"], reverse=True)
    elif sort_by == "delay":
        filtered = sorted(filtered, key=lambda p: p["delay"], reverse=True)
    else:
        filtered = sorted(filtered, key=lambda p: p["score"], reverse=True)

    return [VendorMLPrediction(**p) for p in filtered[:limit]]


@ml_router.get("/scores/{gstin}", response_model=VendorMLPrediction)
def get_single_score(gstin: str):
    """Return ML prediction for a specific vendor."""
    for p in _cached_predictions:
        if p["gstin"] == gstin:
            return VendorMLPrediction(**p)

    # Not in cache — compute on the fly
    driver = get_driver()
    try:
        with driver.session() as session:
            tp = session.run("""
                MATCH (t:Taxpayer {gstin: $gstin})
                RETURN t.gstin AS gstin, t.name AS name, t.state AS state,
                       t.turnover AS turnover, t.risk_score AS risk_score,
                       t.business_type AS btype
            """, gstin=gstin).single()

        if tp is None:
            raise HTTPException(404, f"Vendor {gstin} not found in Neo4j")

        feat = _extract_vendor_features(driver, dict(tp))
        preds = _predict_all([feat])
        return VendorMLPrediction(**preds[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@ml_router.get("/feature-importances")
def get_feature_importances():
    """Return learned feature importances from the trained model."""
    if not _feature_importances:
        if os.path.exists(IMPORTANCES_PATH):
            with open(IMPORTANCES_PATH, "rb") as f:
                return pickle.load(f)
        raise HTTPException(400, "No feature importances. Train the model first (POST /ml/train).")
    return _feature_importances
