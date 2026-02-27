"""
Vendor Score Explainer
========================
Generates human-readable natural-language explanations for
each vendor's risk score. Powers the audit trail UI.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_vendor_explanation(vendor_score: dict) -> dict:
    """
    Generate a comprehensive NL explanation for a vendor's risk score.

    Parameters
    ----------
    vendor_score : dict
        Output from VendorRiskEngine.score_vendor().to_dict()

    Returns
    -------
    dict
        {
            "gstin": str,
            "summary": str,
            "score_breakdown": str,
            "risk_factors": list[str],
            "recommendation": str,
            "itc_impact": str,
            "full_report": str,
        }
    """
    gstin = vendor_score.get("gstin", "Unknown")
    name = vendor_score.get("business_name", "Unknown")
    score = vendor_score.get("final_score", 0)
    category = vendor_score.get("risk_category", "UNKNOWN")

    filing = vendor_score.get("filing_score", 0)
    dispute = vendor_score.get("dispute_score", 0)
    network = vendor_score.get("network_score", 0)
    physical = vendor_score.get("physical_score", 0)
    ml_adj = vendor_score.get("ml_adjustment", 0)
    base = vendor_score.get("base_score", score)

    gnn_prob = vendor_score.get("gnn_fraud_probability", 0)

    risk_factors = vendor_score.get("top_risk_factors", [])

    # ── Summary ──
    if category == "SAFE":
        summary = (
            f"{name} (GSTIN: {gstin}) has been assessed as SAFE with a "
            f"compliance score of {score:.0f}/100. This vendor demonstrates "
            f"strong filing discipline, clean dispute history, and healthy "
            f"supply network connections."
        )
    elif category == "MODERATE":
        summary = (
            f"{name} (GSTIN: {gstin}) has a MODERATE risk score of "
            f"{score:.0f}/100. Some compliance concerns have been identified "
            f"that warrant monitoring but do not indicate fraud."
        )
    elif category == "HIGH_RISK":
        summary = (
            f"{name} (GSTIN: {gstin}) has been flagged as HIGH RISK with a "
            f"score of {score:.0f}/100. Multiple compliance failures detected. "
            f"ITC claims from this vendor should be reviewed carefully."
        )
    else:  # FRAUD
        summary = (
            f"⚠ ALERT: {name} (GSTIN: {gstin}) scores only {score:.0f}/100 "
            f"— classified as SUSPECTED FRAUD. This entity shows patterns "
            f"consistent with shell company / circular trading behaviour. "
            f"IMMEDIATE AUDIT RECOMMENDED."
        )

    # ── Score breakdown ──
    breakdown = (
        f"Score Breakdown (0–100 per component):\n"
        f"  Filing Frequency    : {filing:5.1f}/100  (weight 25%)\n"
        f"  Dispute/Mismatch    : {dispute:5.1f}/100  (weight 25%)\n"
        f"  Network Risk        : {network:5.1f}/100  (weight 25%)\n"
        f"  Physical Existence  : {physical:5.1f}/100  (weight 25%)\n"
        f"  ─────────────────────────────────────────\n"
        f"  Base Score          : {base:5.1f}/100\n"
        f"  ML Adjustment       : {ml_adj:+5.1f}\n"
        f"  ─────────────────────────────────────────\n"
        f"  FINAL SCORE         : {score:5.1f}/100 → {category}"
    )

    # ── Recommendation ──
    if category == "SAFE":
        recommendation = (
            "No action required. Continue routine monitoring. "
            "ITC claims from this vendor can be auto-approved."
        )
    elif category == "MODERATE":
        recommendation = (
            "Enhanced monitoring recommended. Review ITC claims manually "
            "before approval. Schedule periodic compliance review."
        )
    elif category == "HIGH_RISK":
        recommendation = (
            "BLOCK ITC claims from this vendor until compliance review is complete. "
            "Schedule audit within 30 days. Notify downstream buyers."
        )
    else:
        recommendation = (
            "IMMEDIATELY BLOCK all ITC claims. Flag for audit by GST Intelligence. "
            "Investigate connected entities for fraud network. Consider GSTIN "
            "suspension proceedings."
        )

    # ── ITC impact estimate ──
    filing_bd = vendor_score.get("filing_breakdown", {})
    dispute_bd = vendor_score.get("dispute_breakdown", {})

    total_inv = dispute_bd.get("total_invoices", 0)

    # Rough ITC estimate
    if category in ("HIGH_RISK", "FRAUD"):
        itc_impact = (
            f"Estimated ITC exposure: This vendor has {total_inv} invoices in the system. "
            f"All associated ITC claims should be reviewed. "
            f"Blocking ITC from this vendor protects government revenue."
        )
    else:
        itc_impact = (
            f"This vendor has {total_inv} invoices. "
            f"ITC claims are within acceptable risk parameters."
        )

    # ── Full report ──
    risk_section = ""
    if risk_factors:
        risk_lines = "\n".join(f"  {i+1}. {rf}" for i, rf in enumerate(risk_factors[:8]))
        risk_section = f"\nKey Risk Factors:\n{risk_lines}"

    if gnn_prob > 0.5:
        ml_section = (
            f"\nAI Analysis: GNN fraud detection model assigns "
            f"{gnn_prob*100:.0f}% fraud probability to this entity."
        )
    elif gnn_prob > 0:
        ml_section = (
            f"\nAI Analysis: GNN model assigns {gnn_prob*100:.0f}% "
            f"fraud probability (within normal range)."
        )
    else:
        ml_section = ""

    full_report = (
        f"{'='*60}\n"
        f"  VENDOR COMPLIANCE REPORT\n"
        f"{'='*60}\n\n"
        f"{summary}\n\n"
        f"{breakdown}\n"
        f"{risk_section}\n"
        f"{ml_section}\n\n"
        f"Recommendation:\n  {recommendation}\n\n"
        f"ITC Impact:\n  {itc_impact}\n"
        f"\n{'='*60}"
    )

    return {
        "gstin": gstin,
        "business_name": name,
        "score": round(score, 2),
        "category": category,
        "summary": summary,
        "score_breakdown": breakdown,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "itc_impact": itc_impact,
        "full_report": full_report,
    }


def generate_batch_report(vendor_scores: List[dict]) -> dict:
    """
    Generate summary report for all vendors.
    """
    explanations = []
    for vs in vendor_scores:
        explanations.append(generate_vendor_explanation(vs))

    # Stats
    categories = {}
    for e in explanations:
        cat = e["category"]
        categories[cat] = categories.get(cat, 0) + 1

    fraud_list = [e for e in explanations if e["category"] == "FRAUD"]
    high_risk_list = [e for e in explanations if e["category"] == "HIGH_RISK"]

    return {
        "total_vendors": len(explanations),
        "category_distribution": categories,
        "fraud_vendors": [
            {"gstin": e["gstin"], "name": e["business_name"], "score": e["score"]}
            for e in fraud_list
        ],
        "high_risk_vendors": [
            {"gstin": e["gstin"], "name": e["business_name"], "score": e["score"]}
            for e in high_risk_list
        ],
        "explanations": explanations,
    }