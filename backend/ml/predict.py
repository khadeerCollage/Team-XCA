"""
GNN Inference Pipeline
=======================
Loads a trained GSTFraudGNN checkpoint and produces fraud
probabilities for every Taxpayer node in the graph.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .data_preparation import (
    build_edge_index,
    build_feature_matrix,
    build_node_mapping,
    extract_edges,
    extract_taxpayer_nodes,
    get_feature_names,
    get_gstin_to_idx,
    get_idx_to_gstin,
    get_node_properties,
)
from .graph_features import extract_all_features, get_neo4j_driver, get_sql_session
from .train import DEFAULT_MODEL_PATH, DEVICE, load_model

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("torch_geometric required")

logger = logging.getLogger(__name__)

# Module-level cache for latest predictions
_cached_predictions: List[dict] = []
_cached_features_df = None


def _rebuild_data(driver, session) -> Data:
    """Re-build the PyG Data object for inference (no labels needed)."""
    nodes = extract_taxpayer_nodes(driver)
    gstin_to_idx, idx_to_gstin = build_node_mapping(nodes)
    edges = extract_edges(driver)
    edge_index = build_edge_index(edges, gstin_to_idx)
    x = build_feature_matrix(driver, session, gstin_to_idx)
    data = Data(x=x, edge_index=edge_index)
    return data


# ──────────────────────────────────────────────
# Risk-factor generator
# ──────────────────────────────────────────────
def generate_risk_factors(
    gstin: str, features_df, prediction_prob: float, driver
) -> List[str]:
    """
    Build a human-readable list of the top risk factors for *gstin*
    by examining which features deviate most from safe values.
    """
    factors: List[str] = []

    if features_df is None or gstin not in features_df.index:
        return ["Insufficient data for risk-factor analysis."]

    feat = features_df.loc[gstin]

    # GSTR-3B filing rate
    rate = feat.get("gstr3b_filing_rate", 1.0)
    if rate < 0.5:
        months = int(round(rate * 6))
        factors.append(f"GSTR-3B filed only {months}/6 months (rate: {rate:.2f})")

    # Filing gap (shell company indicator)
    gap = feat.get("filing_gap", 0.0)
    if gap > 0.3:
        factors.append(
            f"Filing gap detected — GSTR-1 filed but GSTR-3B missing (gap: {gap:.2f})"
        )

    # ITC overclaim
    overclaim = feat.get("itc_overclaim_ratio", 0.0)
    if overclaim > 0.1:
        factors.append(f"ITC overclaim ratio: {overclaim*100:.0f}% above available credit")

    # IRN coverage
    irn = feat.get("irn_coverage", 1.0)
    if irn < 0.5:
        factors.append(f"IRN coverage critically low: {irn*100:.0f}%")

    # e-Way coverage
    eway = feat.get("eway_coverage", 1.0)
    if eway < 0.5:
        factors.append(f"e-Way bill coverage low: {eway*100:.0f}%")

    # Risky neighbours
    risky_ratio = feat.get("risky_neighbor_ratio", 0.0)
    if risky_ratio > 0.3:
        factors.append(
            f"Connected to high-risk entities (risky-neighbour ratio: {risky_ratio:.2f})"
        )

    # Cycle participation
    if feat.get("cycle_participation", 0.0) > 0.5:
        factors.append("Part of circular trading cycle")

    # Business type
    if feat.get("business_type_encoded", 0.0) > 0.6:
        factors.append("Business type 'Trader' — statistically higher risk category")

    # PageRank
    pr = feat.get("pagerank", 0.0)
    if pr > 0.05:
        factors.append(f"High centrality in supply network (PageRank: {pr:.4f})")

    if not factors:
        factors.append("No significant individual risk factor — risk primarily from network context")

    return factors


# ──────────────────────────────────────────────
# Predict all
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_all(
    driver,
    session,
    model_path: str = None,
) -> List[dict]:
    """
    Run GNN inference on every Taxpayer node.

    Returns list[dict] sorted by fraud_probability descending:
        gstin, business_name, fraud_probability, risk_level,
        confidence, rank, top_risk_factors
    """
    global _cached_predictions, _cached_features_df

    path = model_path or DEFAULT_MODEL_PATH
    model, threshold, feature_names = load_model(path)
    model.eval()

    data = _rebuild_data(driver, session).to(DEVICE)
    probs = model(data.x, data.edge_index).cpu().numpy()

    idx_to_gstin = get_idx_to_gstin()
    node_props = get_node_properties()

    # Raw features for explanation
    _cached_features_df = extract_all_features(driver, session).set_index("gstin")

    results: List[dict] = []
    for idx in range(len(probs)):
        gstin = idx_to_gstin.get(idx, "UNKNOWN")
        prob = float(probs[idx])
        props = node_props.get(gstin, {})

        if prob >= 0.70:
            risk_level = "HIGH"
            confidence = "high"
        elif prob >= 0.30:
            risk_level = "MEDIUM"
            confidence = "medium"
        else:
            risk_level = "LOW"
            confidence = "high"

        risk_factors = generate_risk_factors(gstin, _cached_features_df, prob, driver)

        results.append({
            "gstin": gstin,
            "business_name": props.get("name", "Unknown"),
            "fraud_probability": round(prob, 4),
            "risk_level": risk_level,
            "confidence": confidence,
            "rank": 0,  # filled below
            "top_risk_factors": risk_factors,
        })

    # Sort and assign rank
    results.sort(key=lambda r: r["fraud_probability"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    _cached_predictions = results
    logger.info("Predictions generated for %d taxpayers", len(results))
    return results


# ──────────────────────────────────────────────
# Predict single
# ──────────────────────────────────────────────
def predict_single(
    gstin: str,
    driver,
    session,
    model_path: str = None,
) -> Optional[dict]:
    """
    Return prediction for one GSTIN.
    Runs full-graph inference (GNN needs neighbourhood context).
    """
    all_preds = predict_all(driver, session, model_path)
    for pred in all_preds:
        if pred["gstin"] == gstin:
            return pred
    return None


# ──────────────────────────────────────────────
# Aggregations
# ──────────────────────────────────────────────
def get_risk_distribution(predictions: List[dict] = None) -> dict:
    """Return counts and percentages by risk level."""
    preds = predictions or _cached_predictions
    total = len(preds) or 1
    high = sum(1 for p in preds if p["risk_level"] == "HIGH")
    med = sum(1 for p in preds if p["risk_level"] == "MEDIUM")
    low = sum(1 for p in preds if p["risk_level"] == "LOW")
    return {
        "total": total,
        "high": high,
        "medium": med,
        "low": low,
        "high_percentage": round(100 * high / total, 1),
        "medium_percentage": round(100 * med / total, 1),
        "low_percentage": round(100 * low / total, 1),
    }


def get_top_risks(predictions: List[dict] = None, n: int = 10) -> List[dict]:
    """Return top-N highest-risk taxpayers."""
    preds = predictions or _cached_predictions
    return preds[:n]


# ──────────────────────────────────────────────
# Node embeddings (for frontend vis)
# ──────────────────────────────────────────────
@torch.no_grad()
def get_node_embeddings(
    driver,
    session,
    model_path: str = None,
) -> Dict[str, List[float]]:
    """
    Return 16-D embeddings for every node (for t-SNE / UMAP on dashboard).
    """
    path = model_path or DEFAULT_MODEL_PATH
    model, _, _ = load_model(path)
    model.eval()

    data = _rebuild_data(driver, session).to(DEVICE)
    emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    idx_to_gstin = get_idx_to_gstin()
    return {idx_to_gstin[i]: emb[i].tolist() for i in range(len(emb))}


def get_cached_predictions() -> List[dict]:
    """Return the latest cached prediction list (empty if predict_all hasn't run)."""
    return list(_cached_predictions)