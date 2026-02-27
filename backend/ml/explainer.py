"""
GNN Explainability Layer
==========================
Provides interpretable explanations for every fraud prediction:
    1. Gradient-based feature importance
    2. Neighbour influence analysis
    3. Subgraph extraction (for vis-network)
    4. Natural-language audit report generation
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sqlalchemy import text as sa_text

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
from .predict import predict_all
from .train import DEFAULT_MODEL_PATH, DEVICE, load_model

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("torch_geometric required")

logger = logging.getLogger(__name__)

RISK_COLOR = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#44bb44", "UNKNOWN": "#888888"}


# ──────────────────────────────────────────────
# 1. Gradient-based feature importance
# ──────────────────────────────────────────────
def compute_feature_importance(
    model, data: Data, node_idx: int, feature_names: List[str]
) -> List[dict]:
    """
    Compute the gradient of the output fraud probability w.r.t. the
    input features for node *node_idx*.

    Returns a list sorted by importance (descending), each element:
        { feature, importance, direction, value, interpretation }
    """
    model.eval()
    x = data.x.clone().requires_grad_(True)
    probs = model(x, data.edge_index)
    prob_value = probs[node_idx]
    prob_value.backward()

    grads = x.grad[node_idx].detach().cpu().numpy()
    raw_features = data.x[node_idx].detach().cpu().numpy()

    results = []
    for i, name in enumerate(feature_names):
        importance = float(abs(grads[i]))
        direction = "increases risk" if grads[i] > 0 else "decreases risk"
        val = float(raw_features[i])

        # Human-readable interpretation
        interp = _interpret_feature(name, val, grads[i])

        results.append({
            "feature": name,
            "importance": round(importance, 6),
            "direction": direction,
            "raw_value": round(val, 4),
            "interpretation": interp,
        })

    results.sort(key=lambda r: r["importance"], reverse=True)
    return results


def _interpret_feature(name: str, value: float, gradient: float) -> str:
    """Generate a one-line interpretation for a feature."""
    direction = "increases" if gradient > 0 else "decreases"
    templates = {
        "gstr3b_filing_rate": f"GSTR-3B filing rate (normalised: {value:.2f}) {direction} fraud probability",
        "gstr1_filing_rate": f"GSTR-1 filing rate (normalised: {value:.2f}) {direction} fraud probability",
        "filing_gap": f"Filing gap between GSTR-1 and GSTR-3B (normalised: {value:.2f}) {direction} fraud probability",
        "itc_overclaim_ratio": f"ITC overclaim ratio (normalised: {value:.2f}) {direction} fraud probability",
        "irn_coverage": f"IRN document coverage (normalised: {value:.2f}) {direction} fraud probability",
        "eway_coverage": f"e-Way bill coverage (normalised: {value:.2f}) {direction} fraud probability",
        "risky_neighbor_ratio": f"Proportion of high-risk neighbours (normalised: {value:.2f}) {direction} fraud probability",
        "cycle_participation": f"Circular-trading cycle membership (normalised: {value:.2f}) {direction} fraud probability",
        "pagerank": f"PageRank centrality (normalised: {value:.2f}) {direction} fraud probability",
        "betweenness_centrality": f"Betweenness centrality (normalised: {value:.2f}) {direction} fraud probability",
        "avg_neighbor_risk": f"Average neighbour risk (normalised: {value:.2f}) {direction} fraud probability",
        "business_type_encoded": f"Business-type risk factor (normalised: {value:.2f}) {direction} fraud probability",
    }
    return templates.get(name, f"{name} (normalised: {value:.2f}) {direction} fraud probability")


# ──────────────────────────────────────────────
# 2. Neighbour influence
# ──────────────────────────────────────────────
def compute_neighbor_influence(
    model, data: Data, node_idx: int, idx_to_gstin: dict, driver
) -> List[dict]:
    """
    For each 1-hop neighbour of *node_idx*, estimate how much removing
    its edges changes the prediction (leave-one-out approximation).
    """
    model.eval()

    # Baseline probability
    with torch.no_grad():
        baseline_prob = model(data.x, data.edge_index)[node_idx].item()

    # Find 1-hop neighbours
    edge_src = data.edge_index[0].cpu().numpy()
    edge_tgt = data.edge_index[1].cpu().numpy()

    neighbours = set()
    for i in range(len(edge_src)):
        if edge_src[i] == node_idx:
            neighbours.add(int(edge_tgt[i]))
        elif edge_tgt[i] == node_idx:
            neighbours.add(int(edge_src[i]))

    # Node properties for names / ratings
    with driver.session() as neo_sess:
        result = neo_sess.run(
            """
            MATCH (t:Taxpayer)
            RETURN t.gstin AS gstin, t.name AS name, t.compliance_rating AS rating
            """
        )
        node_info = {r["gstin"]: {"name": r["name"], "rating": r["rating"]} for r in result}

    influences = []
    for nb_idx in neighbours:
        # Build edge_index without edges to/from this neighbour
        mask = ~((edge_src == nb_idx) | (edge_tgt == nb_idx))
        ei_reduced = data.edge_index[:, torch.tensor(mask, dtype=torch.bool)]

        with torch.no_grad():
            new_prob = model(data.x, ei_reduced)[node_idx].item()

        delta = baseline_prob - new_prob  # positive ⇒ neighbour increases fraud score

        nb_gstin = idx_to_gstin.get(nb_idx, "UNKNOWN")
        info = node_info.get(nb_gstin, {})

        # Determine risk level of neighbour
        nb_rating = info.get("rating", "C")
        if nb_rating == "D":
            nb_risk = "HIGH"
        elif nb_rating == "C":
            nb_risk = "MEDIUM"
        else:
            nb_risk = "LOW"

        influences.append({
            "gstin": nb_gstin,
            "name": info.get("name", "Unknown"),
            "influence_score": round(delta, 4),
            "risk_level": nb_risk,
            "compliance_rating": nb_rating,
        })

    influences.sort(key=lambda r: abs(r["influence_score"]), reverse=True)
    return influences


# ──────────────────────────────────────────────
# 3. Subgraph extraction
# ──────────────────────────────────────────────
def extract_explanation_subgraph(driver, gstin: str, hops: int = 2) -> dict:
    """
    Pull the *hops*-hop neighbourhood from Neo4j and return in
    vis-network JSON format.
    """
    query = """
    MATCH path = (t:Taxpayer {gstin: $gstin})-[:SUPPLIES_TO|CLAIMED_ITC_FROM*1..%d]-(conn:Taxpayer)
    WITH t, conn, relationships(path) AS rels
    UNWIND rels AS r
    WITH t, conn,
         startNode(r).gstin AS src,
         endNode(r).gstin   AS tgt,
         type(r)             AS rtype
    RETURN COLLECT(DISTINCT {gstin: conn.gstin, name: conn.name,
                              rating: conn.compliance_rating, btype: conn.business_type})  AS neighbours,
           COLLECT(DISTINCT {source: src, target: tgt, rel: rtype}) AS edges
    """ % hops

    nodes_list = []
    edges_list = []
    seen_nodes = set()

    with driver.session() as neo_sess:
        record = neo_sess.run(query, gstin=gstin).single()

        if record is None:
            # Fallback: just the target node
            res = neo_sess.run(
                "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t.name AS name, t.compliance_rating AS rating",
                gstin=gstin,
            ).single()
            name = res["name"] if res else "Unknown"
            rating = res["rating"] if res else "C"
            return {
                "nodes": [{"id": gstin, "label": name, "group": "HIGH",
                           "color": RISK_COLOR["HIGH"], "size": 30,
                           "borderWidth": 4, "font": {"size": 14}}],
                "edges": [],
                "target_node": gstin,
            }

        # Target node
        res = neo_sess.run(
            "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t.name AS name, t.compliance_rating AS rating",
            gstin=gstin,
        ).single()
        target_name = res["name"] if res else "Unknown"
        target_rating = res["rating"] if res else "C"

        nodes_list.append({
            "id": gstin,
            "label": f"⚠ {target_name}",
            "group": "TARGET",
            "color": "#ff0000",
            "size": 40,
            "borderWidth": 5,
            "font": {"size": 16, "bold": True},
        })
        seen_nodes.add(gstin)

        # Neighbour nodes
        for nb in record["neighbours"]:
            g = nb["gstin"]
            if g in seen_nodes:
                continue
            seen_nodes.add(g)
            rating = nb.get("rating", "C")
            if rating == "D":
                group = "HIGH"
            elif rating == "C":
                group = "MEDIUM"
            else:
                group = "LOW"
            nodes_list.append({
                "id": g,
                "label": nb.get("name", g),
                "group": group,
                "color": RISK_COLOR.get(group, "#888888"),
                "size": 25,
                "borderWidth": 2,
                "font": {"size": 12},
            })

        # Edges
        seen_edges = set()
        for e in record["edges"]:
            key = (e["source"], e["target"], e["rel"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges_list.append({
                "from": e["source"],
                "to": e["target"],
                "label": e["rel"],
                "arrows": "to",
                "color": {"color": "#666666"},
                "width": 2,
            })

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "target_node": gstin,
    }


# ──────────────────────────────────────────────
# 4. Natural-language summary
# ──────────────────────────────────────────────
def generate_nl_summary(
    gstin: str,
    prediction: dict,
    risk_factors: List[str],
    feature_importance: List[dict],
    neighbors: List[dict],
) -> str:
    """
    Produce a multi-paragraph natural-language audit summary.
    """
    name = prediction.get("business_name", gstin)
    prob = prediction.get("fraud_probability", 0)
    level = prediction.get("risk_level", "UNKNOWN")

    lines = []
    lines.append(
        f"{name} (GSTIN: {gstin}) has been flagged as {level} RISK "
        f"with {prob*100:.0f}% fraud probability.\n"
    )

    if risk_factors:
        lines.append("Key Risk Factors:")
        for i, rf in enumerate(risk_factors[:6], 1):
            lines.append(f"  {i}. {rf}")
        lines.append("")

    if neighbors:
        risky_nb = [n for n in neighbors if n["risk_level"] == "HIGH"]
        if risky_nb:
            names = ", ".join(n["name"] for n in risky_nb[:3])
            lines.append(
                f"Network Concern: Connected to {len(risky_nb)} high-risk "
                f"entities including {names}."
            )

    if feature_importance:
        top = feature_importance[0]
        lines.append(
            f"\nMost influential feature: '{top['feature']}' — {top['interpretation']}"
        )

    if level == "HIGH":
        lines.append(
            "\nRecommendation: IMMEDIATE AUDIT RECOMMENDED. "
            "Consider blocking ITC claims associated with this entity."
        )
    elif level == "MEDIUM":
        lines.append(
            "\nRecommendation: Enhanced monitoring recommended. "
            "Schedule audit within next quarter."
        )
    else:
        lines.append("\nRecommendation: Routine monitoring. No immediate action required.")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 5. ITC impact estimation
# ──────────────────────────────────────────────
def _compute_itc_impact(gstin: str, driver, session) -> dict:
    """Estimate the monetary ITC impact of this entity."""
    with driver.session() as neo_sess:
        rec = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
            RETURN count(inv) AS inv_count,
                   COALESCE(SUM(inv.total_value), 0) AS total_value
            """,
            gstin=gstin,
        ).single()
        inv_count = rec["inv_count"] if rec else 0
        total_value = float(rec["total_value"]) if rec else 0.0

        rec2 = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)-[:RECEIVED_BY]->(buyer:Taxpayer)
            RETURN count(DISTINCT buyer.gstin) AS buyer_count
            """,
            gstin=gstin,
        ).single()
        buyer_count = rec2["buyer_count"] if rec2 else 0

    # Approximate ITC = 18% of total_value
    est_itc = round(total_value * 0.18, 2)

    return {
        "total_invoices_issued": inv_count,
        "total_value": total_value,
        "affected_buyers": buyer_count,
        "estimated_itc_at_risk": est_itc,
    }


# ──────────────────────────────────────────────
# Master explanation generator
# ──────────────────────────────────────────────
def generate_explanation(
    gstin: str,
    driver,
    session,
    model=None,
    data=None,
    model_path: str = None,
) -> dict:
    """
    Comprehensive explanation for one GSTIN.

    If *model* / *data* are not provided they are loaded / rebuilt automatically.
    """
    # Load model if needed
    if model is None:
        path = model_path or DEFAULT_MODEL_PATH
        model, threshold, feature_names = load_model(path)
    else:
        feature_names = get_feature_names()

    # Rebuild data if needed
    if data is None:
        from .data_preparation import (
            build_edge_index as _bei,
            build_feature_matrix as _bfm,
            build_node_mapping as _bnm,
            extract_edges as _ee,
            extract_taxpayer_nodes as _etn,
        )

        nodes = _etn(driver)
        g2i, i2g = _bnm(nodes)
        edges = _ee(driver)
        ei = _bei(edges, g2i)
        x = _bfm(driver, session, g2i)
        data = Data(x=x, edge_index=ei).to(DEVICE)
    else:
        data = data.to(DEVICE)

    gstin_to_idx = get_gstin_to_idx()
    idx_to_gstin = get_idx_to_gstin()
    node_props = get_node_properties()

    if gstin not in gstin_to_idx:
        return {"error": f"GSTIN {gstin} not found in graph"}

    node_idx = gstin_to_idx[gstin]
    props = node_props.get(gstin, {})

    # Prediction
    model.eval()
    with torch.no_grad():
        prob = model(data.x, data.edge_index)[node_idx].item()

    risk_level = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.30 else "LOW")

    prediction = {
        "gstin": gstin,
        "business_name": props.get("name", "Unknown"),
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
    }

    # Feature importance
    feat_imp = compute_feature_importance(model, data, node_idx, feature_names)

    # Neighbour influence
    nb_influence = compute_neighbor_influence(model, data, node_idx, idx_to_gstin, driver)

    # Subgraph
    subgraph = extract_explanation_subgraph(driver, gstin, hops=2)

    # Risk factors (from predict module)
    features_df = extract_all_features(driver, session).set_index("gstin")
    from .predict import generate_risk_factors

    risk_factors = generate_risk_factors(gstin, features_df, prob, driver)

    # NL summary
    summary = generate_nl_summary(gstin, prediction, risk_factors, feat_imp, nb_influence)

    # ITC impact
    itc_impact = _compute_itc_impact(gstin, driver, session)

    # Recommendation
    if risk_level == "HIGH":
        recommendation = (
            f"IMMEDIATE AUDIT RECOMMENDED. Block all ITC claims originating from "
            f"this entity. Estimated revenue at risk: ₹{itc_impact['estimated_itc_at_risk']:,.0f}."
        )
    elif risk_level == "MEDIUM":
        recommendation = (
            "Enhanced monitoring recommended. Schedule compliance review within 30 days."
        )
    else:
        recommendation = "No immediate action required. Continue routine monitoring."

    return {
        "gstin": gstin,
        "business_name": props.get("name", "Unknown"),
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
        "summary": summary,
        "top_risk_factors": risk_factors,
        "feature_importance": feat_imp[:10],
        "influential_neighbors": nb_influence[:10],
        "subgraph": subgraph,
        "recommendation": recommendation,
        "itc_impact": itc_impact,
    }