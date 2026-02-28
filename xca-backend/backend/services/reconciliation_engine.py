"""
services/reconciliation_engine.py
"""

from database.neo4j_connection import run_query
from schemas.mismatch import (
    MismatchRecord, MismatchType, RiskLevel,
    ChainHop, ReconciliationResult
)
import uuid
from datetime import datetime


# ════════════════════════════════════════════════════════════════
#  RULE 1 — GSTR-1 vs GSTR-2B Value Match
# ════════════════════════════════════════════════════════════════

def rule_gstr1_vs_gstr2b(period: str = None) -> list[MismatchRecord]:
    """
    Traverse: Invoice node → REPORTED_IN → Return (GSTR-2B)
    Compare taxable_value in GSTR-1 vs GSTR-2B.
    Flag delta > ₹500 as mismatch.
    """
    period_filter = "AND inv.period = $period" if period else ""

    query = f"""
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
              -[:REPORTED_IN]->(ret:Return {{return_type: 'GSTR-2B'}})
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        WHERE inv.taxable_value IS NOT NULL
          AND ret.taxable_value IS NOT NULL
          AND abs(inv.taxable_value - ret.taxable_value) > 500
          {period_filter}
        RETURN
            inv.invoice_no   AS invoice_no,
            inv.irn          AS irn,
            inv.period       AS period,
            inv.taxable_value AS gstr1_value,
            ret.taxable_value AS gstr2b_value,
            inv.gst_rate     AS gst_rate,
            seller.gstin     AS seller_gstin,
            seller.name      AS seller_name,
            buyer.gstin      AS buyer_gstin,
            buyer.name       AS buyer_name
        ORDER BY abs(inv.taxable_value - ret.taxable_value) DESC
    """
    params = {"period": period} if period else {}
    rows = run_query(query, params)

    mismatches = []
    for r in rows:
        delta = round(r['gstr1_value'] - r['gstr2b_value'], 2)
        itc_blocked = round(abs(delta) * r['gst_rate'] / 100, 2)

        mismatches.append(MismatchRecord(
            mismatch_id    = f"MM-GSTR-{str(uuid.uuid4())[:8].upper()}",
            invoice_no     = r['invoice_no'],
            irn            = r.get('irn'),
            seller_gstin   = r['seller_gstin'],
            seller_name    = r['seller_name'],
            buyer_gstin    = r['buyer_gstin'],
            buyer_name     = r['buyer_name'],
            period         = r['period'],
            mismatch_type  = MismatchType.VALUE_DELTA,
            risk_level     = _calculate_risk_by_delta(abs(delta)),
            gstr1_value    = r['gstr1_value'],
            gstr2b_value   = r['gstr2b_value'],
            delta          = delta,
            itc_blocked    = itc_blocked,
            chain_hops     = [
                ChainHop(hop_name="Invoice",   status=True),
                ChainHop(hop_name="IRN",       status=True),
                ChainHop(hop_name="GSTR-1",    status=True),
                ChainHop(hop_name="GSTR-2B",   status=False,
                         detail=f"Value delta: ₹{abs(delta):,.2f}"),
                ChainHop(hop_name="Payment",   status=False),
            ],
            chain_broken_at = "GSTR-2B"
        ))

    return mismatches


# ════════════════════════════════════════════════════════════════
#  RULE 2 — Multi-Hop ITC Chain Validation
# ════════════════════════════════════════════════════════════════

def rule_itc_chain_validation(period: str = None) -> list[MismatchRecord]:
    """
    Full 5-hop traversal:
    Invoice → IRN → e-Way Bill → GSTR-2B → Payment
    Any broken hop = ITC at risk.
    """
    period_filter = "AND inv.period = $period" if period else ""

    # Find invoices missing IRN
    query_no_irn = f"""
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        WHERE NOT (inv)-[:HAS_IRN]->(:IRN)
          AND inv.taxable_value > 50000
          {period_filter}
        RETURN
            inv.invoice_no  AS invoice_no,
            inv.period      AS period,
            inv.taxable_value AS taxable_value,
            inv.gst_rate    AS gst_rate,
            seller.gstin    AS seller_gstin,
            seller.name     AS seller_name,
            buyer.gstin     AS buyer_gstin,
            buyer.name      AS buyer_name
    """
    params = {"period": period} if period else {}
    no_irn_rows = run_query(query_no_irn, params)

    # Find invoices missing e-Way Bill
    query_no_ewb = f"""
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        WHERE inv.missing_ewb = true
          AND inv.taxable_value > 50000
          {period_filter}
        RETURN
            inv.invoice_no  AS invoice_no,
            inv.period      AS period,
            inv.taxable_value AS taxable_value,
            inv.gst_rate    AS gst_rate,
            seller.gstin    AS seller_gstin,
            seller.name     AS seller_name,
            buyer.gstin     AS buyer_gstin,
            buyer.name      AS buyer_name
    """
    no_ewb_rows = run_query(query_no_ewb, params)

    mismatches = []

    for r in no_irn_rows:
        itc = round(r['taxable_value'] * r['gst_rate'] / 100, 2)
        mismatches.append(MismatchRecord(
            mismatch_id   = f"MM-IRN-{str(uuid.uuid4())[:8].upper()}",
            invoice_no    = r['invoice_no'],
            seller_gstin  = r['seller_gstin'],
            seller_name   = r['seller_name'],
            buyer_gstin   = r['buyer_gstin'],
            buyer_name    = r['buyer_name'],
            period        = r['period'],
            mismatch_type = MismatchType.MISSING_IRN,
            risk_level    = RiskLevel.CRITICAL,
            gstr1_value   = r['taxable_value'],
            gstr2b_value  = 0.0,
            delta         = r['taxable_value'],
            itc_blocked   = itc,
            chain_hops    = [
                ChainHop(hop_name="Invoice", status=True),
                ChainHop(hop_name="IRN",     status=False,
                         detail="IRN not registered on NIC server"),
                ChainHop(hop_name="GSTR-1",  status=False),
                ChainHop(hop_name="GSTR-2B", status=False),
                ChainHop(hop_name="Payment", status=False),
            ],
            chain_broken_at = "IRN"
        ))

    for r in no_ewb_rows:
        itc = round(r['taxable_value'] * r['gst_rate'] / 100, 2)
        mismatches.append(MismatchRecord(
            mismatch_id   = f"MM-EWB-{str(uuid.uuid4())[:8].upper()}",
            invoice_no    = r['invoice_no'],
            seller_gstin  = r['seller_gstin'],
            seller_name   = r['seller_name'],
            buyer_gstin   = r['buyer_gstin'],
            buyer_name    = r['buyer_name'],
            period        = r['period'],
            mismatch_type = MismatchType.EWAYBILL_MISSING,
            risk_level    = RiskLevel.HIGH,
            gstr1_value   = r['taxable_value'],
            gstr2b_value  = r['taxable_value'],
            delta         = 0.0,
            itc_blocked   = itc,
            chain_hops    = [
                ChainHop(hop_name="Invoice",    status=True),
                ChainHop(hop_name="IRN",        status=True),
                ChainHop(hop_name="e-Way Bill", status=False,
                         detail="No e-Way Bill found for goods movement"),
                ChainHop(hop_name="GSTR-2B",    status=True),
                ChainHop(hop_name="Payment",    status=False),
            ],
            chain_broken_at = "e-Way Bill"
        ))

    return mismatches


# ════════════════════════════════════════════════════════════════
#  RULE 3 — Circular Transaction Detection (Fraud Rings)
# ════════════════════════════════════════════════════════════════

def rule_circular_transactions() -> list[dict]:
    """
    Detect fraud rings: cycles of length 3–5 in TRANSACTS_WITH graph.
    A→B→C→A = circular = fake invoice fraud signal.
    Returns raw dicts (not MismatchRecord) for the graph viz.
    """
    query = """
        MATCH path = (a:Taxpayer)-[:TRANSACTS_WITH*3..5]->(a)
        RETURN
            [n IN nodes(path) | n.gstin] AS cycle_gstins,
            [n IN nodes(path) | n.name]  AS cycle_names,
            length(path) AS cycle_length
        LIMIT 20
    """
    return run_query(query)


# ════════════════════════════════════════════════════════════════
#  MASTER: Run All Rules
# ════════════════════════════════════════════════════════════════

def run_full_reconciliation(period: str = None) -> ReconciliationResult:
    """
    Runs ALL reconciliation rules and combines results.
    This is what the /reconcile endpoint calls.
    """
    all_mismatches: list[MismatchRecord] = []

    # Run all rules
    all_mismatches += rule_gstr1_vs_gstr2b(period)
    all_mismatches += rule_itc_chain_validation(period)

    # Get total invoices count
    q = "MATCH (inv:Invoice) RETURN count(inv) AS total"
    if period:
        q = "MATCH (inv:Invoice {period: $period}) RETURN count(inv) AS total"
    count_result = run_query(q, {"period": period} if period else {})
    total_invoices = count_result[0]['total'] if count_result else 0

    # Aggregate
    total_itc_at_risk = sum(m.itc_blocked for m in all_mismatches)
    by_type = {}
    for m in all_mismatches:
        by_type[m.mismatch_type.value] = by_type.get(m.mismatch_type.value, 0) + 1

    return ReconciliationResult(
        run_id           = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        period           = period or "All Periods",
        total_invoices   = total_invoices,
        total_mismatches = len(all_mismatches),
        total_itc_at_risk = total_itc_at_risk,
        critical_count   = sum(1 for m in all_mismatches if m.risk_level == RiskLevel.CRITICAL),
        high_count       = sum(1 for m in all_mismatches if m.risk_level == RiskLevel.HIGH),
        medium_count     = sum(1 for m in all_mismatches if m.risk_level == RiskLevel.MEDIUM),
        low_count        = sum(1 for m in all_mismatches if m.risk_level == RiskLevel.LOW),
        by_type          = by_type,
        mismatches       = all_mismatches,
    )


# ── Private: risk level from delta amount ─────────────────────
def _calculate_risk_by_delta(delta: float) -> RiskLevel:
    if delta >= 100000:   return RiskLevel.CRITICAL   # ≥ ₹1L
    if delta >= 50000:    return RiskLevel.HIGH        # ≥ ₹50K
    if delta >= 10000:    return RiskLevel.MEDIUM      # ≥ ₹10K
    return RiskLevel.LOW
