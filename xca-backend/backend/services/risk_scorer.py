"""
services/risk_scorer.py
"""

from database.neo4j_connection import run_query
from schemas.vendor_risk import VendorRiskScore


# ════════════════════════════════════════════════════════════════
#  WEIGHTS — Tune these to change scoring behavior
# ════════════════════════════════════════════════════════════════
WEIGHTS = {
    "mismatch_rate":       0.35,  # Most important signal
    "avg_filing_delay":    0.25,  # Late filing = risky
    "itc_overclaim_ratio": 0.20,  # Claiming more than eligible
    "circular_txn_flag":   0.15,  # In a fraud ring
    "new_vendor_flag":     0.05,  # New vendor = slightly risky
}


def score_vendor(gstin: str) -> VendorRiskScore | None:
    """
    Compute the risk score for a single vendor (by GSTIN).
    Returns VendorRiskScore or None if vendor not found.
    """
    # ── Fetch vendor info ──────────────────────────────────────
    vendor_rows = run_query(
        "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t",
        {"gstin": gstin}
    )
    if not vendor_rows:
        return None
    vendor = vendor_rows[0]['t']

    # ── Mismatch rate ──────────────────────────────────────────
    total_q = run_query(
        "MATCH (:Taxpayer {gstin: $g})-[:ISSUED]->(inv:Invoice) RETURN count(inv) AS c",
        {"g": gstin}
    )
    total_txns = total_q[0]['c'] if total_q else 0

    mismatch_q = run_query(
        """
        MATCH (:Taxpayer {gstin: $g})-[:ISSUED]->(inv:Invoice)-[:HAS_MISMATCH]->()
        RETURN count(inv) AS c
        """,
        {"g": gstin}
    )
    mismatch_count = mismatch_q[0]['c'] if mismatch_q else 0
    mismatch_rate = (mismatch_count / total_txns) if total_txns > 0 else 0.0

    # ── ITC overclaim ratio ────────────────────────────────────
    itc_q = run_query(
        """
        MATCH (:Taxpayer {gstin: $g})-[:RECEIVED]->(inv:Invoice)
              -[:REPORTED_IN]->(r:Return {return_type: 'GSTR-2B'})
        RETURN
            sum(r.itc_eligible) AS eligible,
            sum(r.taxable_value * coalesce(inv.gst_rate,18)/100) AS claimed
        """,
        {"g": gstin}
    )
    if itc_q and itc_q[0]['eligible'] and itc_q[0]['eligible'] > 0:
        claimed  = itc_q[0]['claimed']  or 0
        eligible = itc_q[0]['eligible'] or 0
        itc_overclaim = min(max((claimed - eligible) / eligible, 0), 1)
        total_itc_claimed  = claimed
        total_itc_eligible = eligible
        total_itc_at_risk  = max(claimed - eligible, 0)
    else:
        itc_overclaim = 0.0
        total_itc_claimed = total_itc_eligible = total_itc_at_risk = 0.0

    # ── Circular transaction flag ──────────────────────────────
    cycle_q = run_query(
        """
        MATCH path = (a:Taxpayer {gstin: $g})-[:TRANSACTS_WITH*3..5]->(a)
        RETURN count(path) AS c LIMIT 1
        """,
        {"g": gstin}
    )
    circular_flag = 1.0 if (cycle_q and cycle_q[0]['c'] > 0) else 0.0

    # ── Filing delay (normalized 0–1, max 90 days late = 1.0) ─
    delay_q = run_query(
        """
        MATCH (:Taxpayer {gstin: $g})-[:FILED]->(r:Return)
        WHERE r.filing_date IS NOT NULL AND r.due_date IS NOT NULL
        RETURN avg(
            duration.between(date(r.due_date), date(r.filing_date)).days
        ) AS avg_delay
        """,
        {"g": gstin}
    )
    avg_delay_days = delay_q[0]['avg_delay'] if delay_q and delay_q[0]['avg_delay'] else 0
    avg_filing_delay = min(max(avg_delay_days / 90, 0), 1)

    # ── New vendor flag (<6 months old = 1.0) ─────────────────
    reg_date = vendor.get('registration_date')
    new_vendor_flag = 0.0
    if reg_date:
        from datetime import datetime
        try:
            reg = datetime.strptime(str(reg_date), "%Y-%m-%d")
            months_old = (datetime.now() - reg).days / 30
            new_vendor_flag = 1.0 if months_old < 6 else 0.0
        except Exception:
            new_vendor_flag = 0.0

    # ── Weighted final score ───────────────────────────────────
    risk_score = round(
        WEIGHTS["mismatch_rate"]       * mismatch_rate     +
        WEIGHTS["avg_filing_delay"]    * avg_filing_delay  +
        WEIGHTS["itc_overclaim_ratio"] * itc_overclaim     +
        WEIGHTS["circular_txn_flag"]   * circular_flag     +
        WEIGHTS["new_vendor_flag"]     * new_vendor_flag,
        4
    )

    risk_level = (
        "HIGH"   if risk_score > 0.6 else
        "MEDIUM" if risk_score > 0.3 else
        "LOW"
    )

    recommendation = (
        "BLOCK + AUDIT"  if risk_score > 0.6 else
        "REVIEW"         if risk_score > 0.3 else
        "AUTO-APPROVE"
    )

    return VendorRiskScore(
        gstin                = gstin,
        name                 = vendor.get('name', ''),
        state                = vendor.get('state'),
        mismatch_rate        = round(mismatch_rate, 4),
        avg_filing_delay     = round(avg_filing_delay, 4),
        itc_overclaim_ratio  = round(itc_overclaim, 4),
        circular_txn_flag    = circular_flag,
        new_vendor_flag      = new_vendor_flag,
        risk_score           = risk_score,
        risk_level           = risk_level,
        total_transactions   = total_txns,
        total_itc_claimed    = round(total_itc_claimed, 2),
        total_itc_eligible   = round(total_itc_eligible, 2),
        total_itc_at_risk    = round(total_itc_at_risk, 2),
        recommendation       = recommendation,
    )


def score_all_vendors() -> list[VendorRiskScore]:
    """Score every vendor in the DB. Used for the leaderboard."""
    gstins_q = run_query("MATCH (t:Taxpayer) RETURN t.gstin AS gstin")
    scores = []
    for row in gstins_q:
        result = score_vendor(row['gstin'])
        if result:
            scores.append(result)
    # Sort: highest risk first
    return sorted(scores, key=lambda x: x.risk_score, reverse=True)
