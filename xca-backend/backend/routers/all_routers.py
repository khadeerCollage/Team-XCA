"""
routers/all_routers.py
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from database.neo4j_connection import run_query, run_write_query
from schemas.taxpayer import TaxpayerCreate, TaxpayerResponse, TaxpayerSummary
from schemas.invoice import InvoiceCreate, InvoiceResponse
from schemas.mismatch import (
    ReconcileRequest, ReconciliationResult,
    MismatchRecord, RiskLevel
)
from services.reconciliation_engine import run_full_reconciliation
from services.risk_scorer import score_vendor, score_all_vendors
from services.audit_generator import generate_audit_note, _fallback_audit_note


# ══════════════════════════════════════════════════════════════
#  TAXPAYERS ROUTER  — /taxpayers
# ══════════════════════════════════════════════════════════════
taxpayers_router = APIRouter(prefix="/taxpayers", tags=["Taxpayers"])


@taxpayers_router.get("/", response_model=list[TaxpayerSummary])
def list_taxpayers(
    state:      Optional[str] = Query(None, description="Filter by state"),
    risk_level: Optional[str] = Query(None, description="LOW / MEDIUM / HIGH"),
    limit:      int           = Query(50, le=200),
    offset:     int           = Query(0)
):
    """
    List all taxpayers with their risk summary.
    Used for the Vendor Leaderboard in the dashboard.
    """
    where_clauses = []
    params = {"limit": limit, "offset": offset}

    if state:
        where_clauses.append("t.state = $state")
        params["state"] = state

    where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        MATCH (t:Taxpayer)
        {where_str}
        OPTIONAL MATCH (t)-[:ISSUED]->(inv:Invoice)-[:HAS_MISMATCH]->()
        RETURN
            t.gstin          AS gstin,
            t.name           AS name,
            t.state          AS state,
            t.risk_score     AS risk_score,
            count(inv)       AS mismatch_count
        ORDER BY t.risk_score DESC
        SKIP $offset LIMIT $limit
    """
    rows = run_query(query, params)

    result = []
    for r in rows:
        score = r.get('risk_score') or 0.0
        result.append(TaxpayerSummary(
            gstin          = r['gstin'],
            name           = r['name'],
            state          = r.get('state', ''),
            risk_score     = score,
            risk_level     = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW",
            mismatch_count = r.get('mismatch_count', 0),
        ))

    # Filter by risk level after scoring
    if risk_level:
        result = [t for t in result if t.risk_level == risk_level.upper()]

    return result


@taxpayers_router.get("/{gstin}", response_model=TaxpayerResponse)
def get_taxpayer(gstin: str):
    """Get a single taxpayer by GSTIN."""
    rows = run_query(
        "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t",
        {"gstin": gstin.upper()}
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Taxpayer {gstin} not found")
    t = rows[0]['t']
    return TaxpayerResponse(**t)


@taxpayers_router.post("/", response_model=TaxpayerResponse, status_code=201)
def create_taxpayer(body: TaxpayerCreate):
    """Create a new Taxpayer node in Neo4j."""
    run_write_query("""
        MERGE (t:Taxpayer {gstin: $gstin})
        SET t.name              = $name,
            t.state             = $state,
            t.turnover          = $turnover,
            t.business_type     = $business_type,
            t.registration_date = $registration_date,
            t.risk_score        = $risk_score
    """, body.model_dump())

    return TaxpayerResponse(**body.model_dump())


# ══════════════════════════════════════════════════════════════
#  INVOICES ROUTER  — /invoices
# ══════════════════════════════════════════════════════════════
invoices_router = APIRouter(prefix="/invoices", tags=["Invoices"])


@invoices_router.get("/", response_model=list[InvoiceResponse])
def list_invoices(
    period:       Optional[str] = Query(None),
    seller_gstin: Optional[str] = Query(None),
    buyer_gstin:  Optional[str] = Query(None),
    has_mismatch: Optional[bool] = Query(None),
    limit:        int = Query(50, le=200),
    offset:       int = Query(0)
):
    """List invoices with optional filters."""
    conditions = []
    params = {"limit": limit, "offset": offset}

    if period:
        conditions.append("inv.period = $period")
        params["period"] = period
    if seller_gstin:
        conditions.append("seller.gstin = $seller_gstin")
        params["seller_gstin"] = seller_gstin.upper()
    if buyer_gstin:
        conditions.append("buyer.gstin = $buyer_gstin")
        params["buyer_gstin"] = buyer_gstin.upper()

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        {where}
        RETURN
            inv.invoice_no    AS invoice_no,
            inv.irn           AS irn,
            seller.gstin      AS seller_gstin,
            buyer.gstin       AS buyer_gstin,
            inv.invoice_date  AS invoice_date,
            inv.taxable_value AS taxable_value,
            inv.gst_rate      AS gst_rate,
            inv.igst          AS igst,
            inv.cgst          AS cgst,
            inv.sgst          AS sgst,
            inv.period        AS period,
            inv.status        AS status,
            inv.missing_ewb   AS missing_ewb
        SKIP $offset LIMIT $limit
    """
    rows = run_query(query, params)
    return [InvoiceResponse(
        **{k: v for k, v in r.items() if k in InvoiceResponse.model_fields},
        total_gst=((r.get('igst') or 0) + (r.get('cgst') or 0) + (r.get('sgst') or 0))
    ) for r in rows]


@invoices_router.get("/{invoice_no}", response_model=InvoiceResponse)
def get_invoice(invoice_no: str):
    """Get a single invoice by invoice number."""
    rows = run_query(
        """
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice {invoice_no: $inv})
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        RETURN inv, seller.gstin AS seller_gstin, buyer.gstin AS buyer_gstin
        """,
        {"inv": invoice_no}
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_no} not found")
    r = rows[0]
    inv = r['inv']
    return InvoiceResponse(
        invoice_no    = inv['invoice_no'],
        irn           = inv.get('irn'),
        seller_gstin  = r['seller_gstin'],
        buyer_gstin   = r['buyer_gstin'],
        invoice_date  = inv.get('invoice_date', ''),
        taxable_value = inv.get('taxable_value', 0),
        gst_rate      = inv.get('gst_rate', 18),
        igst          = inv.get('igst', 0),
        cgst          = inv.get('cgst', 0),
        sgst          = inv.get('sgst', 0),
        total_gst     = (inv.get('igst',0)+inv.get('cgst',0)+inv.get('sgst',0)),
        period        = inv.get('period', ''),
        status        = inv.get('status', 'Filed'),
        missing_ewb   = inv.get('missing_ewb', False),
    )


# ══════════════════════════════════════════════════════════════
#  RECONCILIATION ROUTER — /reconcile  ← THE CORE ENDPOINT
# ══════════════════════════════════════════════════════════════
reconcile_router = APIRouter(prefix="/reconcile", tags=["Reconciliation"])


@reconcile_router.post("/run", response_model=ReconciliationResult)
def run_reconciliation(body: ReconcileRequest):
    """
    🔑 CORE ENDPOINT — Triggers the full reconciliation engine.
    Runs all graph traversal rules and returns all mismatches.
    Pass run_audit_ai=false for faster response (no LLM calls).
    """
    result = run_full_reconciliation(period=body.period)

    if body.run_audit_ai:
        from services.audit_generator import batch_generate_audit_notes
        result.mismatches = batch_generate_audit_notes(
            result.mismatches,
            use_ai=True
        )
    else:
        from services.audit_generator import batch_generate_audit_notes
        result.mismatches = batch_generate_audit_notes(
            result.mismatches,
            use_ai=False  # fallback rule-based notes, instant
        )

    return result


@reconcile_router.get("/mismatches", response_model=list[MismatchRecord])
def get_mismatches(
    period:        Optional[str]      = Query(None),
    risk_level:    Optional[str]      = Query(None, description="LOW/MEDIUM/HIGH/CRITICAL"),
    mismatch_type: Optional[str]      = Query(None),
    gstin:         Optional[str]      = Query(None, description="Seller or buyer GSTIN"),
    limit:         int                = Query(50, le=200),
    offset:        int                = Query(0),
):
    """
    Get list of mismatches with filters.
    Used by the Mismatch Table in the dashboard.
    """
    result = run_full_reconciliation(period=period)
    mismatches = result.mismatches

    # Apply filters
    if risk_level:
        mismatches = [m for m in mismatches
                      if m.risk_level.value == risk_level.upper()]
    if mismatch_type:
        mismatches = [m for m in mismatches
                      if mismatch_type.lower() in m.mismatch_type.value.lower()]
    if gstin:
        g = gstin.upper()
        mismatches = [m for m in mismatches
                      if m.seller_gstin == g or m.buyer_gstin == g]

    # Add fallback audit notes (no AI, fast)
    from services.audit_generator import batch_generate_audit_notes
    mismatches = batch_generate_audit_notes(mismatches, use_ai=False)

    return mismatches[offset: offset + limit]


@reconcile_router.get("/summary")
def get_reconciliation_summary(period: Optional[str] = Query(None)):
    """
    Quick summary stats for the dashboard header.
    Returns counts and total ITC at risk — no mismatch details.
    """
    result = run_full_reconciliation(period=period)
    return {
        "run_id":             result.run_id,
        "period":             result.period,
        "total_invoices":     result.total_invoices,
        "total_mismatches":   result.total_mismatches,
        "total_itc_at_risk":  result.total_itc_at_risk,
        "critical_count":     result.critical_count,
        "high_count":         result.high_count,
        "medium_count":       result.medium_count,
        "low_count":          result.low_count,
        "by_type":            result.by_type,
    }


# ══════════════════════════════════════════════════════════════
#  RISK ROUTER — /risk
# ══════════════════════════════════════════════════════════════
risk_router = APIRouter(prefix="/risk", tags=["Vendor Risk"])


@risk_router.get("/vendor/{gstin}")
def get_vendor_risk(gstin: str):
    """Get risk score for a single vendor."""
    score = score_vendor(gstin.upper())
    if not score:
        raise HTTPException(status_code=404, detail=f"Vendor {gstin} not found")
    return score


@risk_router.get("/leaderboard")
def get_risk_leaderboard(
    limit: int = Query(20, le=100)
):
    """
    Get all vendors sorted by risk score (highest risk first).
    This powers the Vendor Compliance Leaderboard.
    """
    all_scores = score_all_vendors()
    return all_scores[:limit]


# ══════════════════════════════════════════════════════════════
#  AUDIT ROUTER — /audit
# ══════════════════════════════════════════════════════════════
audit_router = APIRouter(prefix="/audit", tags=["Audit Trail"])


@audit_router.get("/invoice/{invoice_no}")
def get_audit_trail(invoice_no: str, use_ai: bool = Query(False)):
    """
    Get the full audit trail for a specific invoice.
    Returns hop chain + AI-generated audit note.
    This powers the Audit Trail Panel.
    """
    # Run reconciliation and find this invoice's mismatch
    result = run_full_reconciliation()
    mismatch = next(
        (m for m in result.mismatches if m.invoice_no == invoice_no),
        None
    )

    if not mismatch:
        # No mismatch — invoice is clean
        return {
            "invoice_no": invoice_no,
            "status": "CLEAN",
            "message": "No mismatches found for this invoice. ITC is eligible.",
            "chain_hops": [
                {"hop_name": "Invoice",    "status": True},
                {"hop_name": "IRN",        "status": True},
                {"hop_name": "e-Way Bill", "status": True},
                {"hop_name": "GSTR-2B",    "status": True},
                {"hop_name": "Payment",    "status": True},
            ],
            "audit_note": "This invoice has passed all verification checks. ITC claim is valid."
        }

    # Add audit note
    mismatch.audit_note = (
        generate_audit_note(mismatch) if use_ai
        else _fallback_audit_note(mismatch)
    )

    return mismatch


# ══════════════════════════════════════════════════════════════
#  GRAPH ROUTER — /graph  (feeds the React force-graph)
# ══════════════════════════════════════════════════════════════
graph_router = APIRouter(prefix="/graph", tags=["Graph Visualization"])


@graph_router.get("/nodes-edges")
def get_graph_data(
    limit_nodes: int = Query(50, le=200, description="Max taxpayer nodes to return"),
    period:      Optional[str] = Query(None)
):
    """
    Returns nodes + edges in a format ready for react-force-graph.
    Nodes = Taxpayers, Edges = Transactions.
    Color-coded by risk level.
    """
    # Get taxpayer nodes
    nodes_q = run_query("""
        MATCH (t:Taxpayer)
        RETURN
            t.gstin      AS id,
            t.name       AS label,
            t.state      AS state,
            t.risk_score AS risk_score
        LIMIT $limit
    """, {"limit": limit_nodes})

    # Get edges
    period_filter = "AND inv.period = $period" if period else ""
    params = {"period": period} if period else {}
    edges_q = run_query(f"""
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        {f'WHERE inv.period = $period' if period else ''}
        RETURN
            seller.gstin AS source,
            buyer.gstin  AS target,
            inv.invoice_no AS invoice_no,
            inv.taxable_value AS value,
            inv.missing_ewb AS missing_ewb,
            EXISTS((inv)-[:HAS_MISMATCH]->()) AS has_mismatch
        LIMIT 300
    """, params)

    # Format nodes
    nodes = []
    for n in nodes_q:
        score = n.get('risk_score') or 0.0
        nodes.append({
            "id":         n['id'],
            "label":      n['label'],
            "state":      n.get('state', ''),
            "risk_score": score,
            "risk_level": "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW",
            "color": "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.3 else "#22c55e",
        })

    # Format edges
    edges = []
    for e in edges_q:
        edges.append({
            "source":     e['source'],
            "target":     e['target'],
            "invoice_no": e['invoice_no'],
            "value":      e.get('value', 0),
            "has_mismatch": e.get('has_mismatch', False),
            "color": "#ef4444" if e.get('has_mismatch') else "#475569",
        })

    # Get circular transaction clusters
    circles = run_query("""
        MATCH path = (a:Taxpayer)-[:TRANSACTS_WITH*3..5]->(a)
        RETURN [n IN nodes(path) | n.gstin] AS cycle
        LIMIT 5
    """)

    return {
        "nodes": nodes,
        "edges": edges,
        "circular_clusters": [c['cycle'] for c in circles],
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "fraud_rings": len(circles),
        }
    }
