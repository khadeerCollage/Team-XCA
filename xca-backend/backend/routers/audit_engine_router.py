"""
routers/audit_engine_router.py
────────────────────────────────
Integrates the gst-audit-engine into xca-backend.
Extracts ValidationInput fields from Neo4j per invoice,
runs the 5-step compliance chain, and returns structured AuditResult.

Endpoints:
    POST /audit-engine/generate/{invoice_no}  — Full audit for one invoice
    POST /audit-engine/batch                  — Audit all invoices with mismatches
    GET  /audit-engine/results                — Cached batch results
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from database.neo4j_connection import run_query
from services.audit_engine import generate_audit_report, ValidationInput, AuditResult

audit_engine_router = APIRouter(
    prefix="/audit-engine",
    tags=["Audit Engine"],
)

# ── In-memory cache for batch results ─────────────────────────
_cached_audit_results: list[dict] = []


# ══════════════════════════════════════════════════════════════
#  HELPER: Extract ValidationInput from Neo4j for an invoice
# ══════════════════════════════════════════════════════════════

def _extract_invoice_data(invoice_no: str) -> dict | None:
    """
    Runs a single comprehensive Cypher query to extract all fields
    needed by the audit engine's ValidationInput model.

    Returns a dict ready to be passed to ValidationInput(...) or None if not found.
    """
    query = """
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice {invoice_no: $invoice_no})
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        OPTIONAL MATCH (inv)-[:HAS_IRN]->(irn:IRN)
        OPTIONAL MATCH (inv)-[:COVERED_BY]->(ewb:EWayBill)
        OPTIONAL MATCH (inv)-[:REPORTED_IN]->(r1:Return {return_type: 'GSTR-1'})
        OPTIONAL MATCH (inv)-[:REFLECTED_IN]->(r2:Return {return_type: 'GSTR-2B'})
        RETURN
            inv.invoice_no       AS invoice_no,
            inv.taxable_value    AS taxable_value,
            inv.gst_rate         AS gst_rate,
            inv.period           AS period,
            inv.missing_ewb      AS missing_ewb,
            inv.has_mismatch     AS has_mismatch,
            seller.gstin         AS seller_gstin,
            seller.name          AS seller_name,
            buyer.gstin          AS buyer_gstin,
            buyer.name           AS buyer_name,
            irn IS NOT NULL      AS has_irn,
            ewb IS NOT NULL      AS has_ewb,
            ewb.from_state       AS from_state,
            ewb.to_state         AS to_state,
            r1 IS NOT NULL       AS has_gstr1,
            r2 IS NOT NULL       AS has_gstr2b
        LIMIT 1
    """
    rows = run_query(query, {"invoice_no": invoice_no})
    if not rows:
        return None

    r = rows[0]

    # Calculate ITC amount
    taxable_value = r.get("taxable_value") or 0.0
    gst_rate = r.get("gst_rate") or 18.0
    itc_amount = round(taxable_value * gst_rate / 100, 2)
    if itc_amount <= 0:
        itc_amount = 1.0  # Minimum to satisfy ValidationInput constraint

    # Direct checks from Neo4j
    has_gstr1 = bool(r.get("has_gstr1"))
    has_gstr2b = bool(r.get("has_gstr2b"))
    has_irn = bool(r.get("has_irn"))
    has_ewb = bool(r.get("has_ewb"))
    missing_ewb = bool(r.get("missing_ewb"))
    has_mismatch = bool(r.get("has_mismatch"))

    # ── Derive boolean fields ──────────────────────────────────
    # gstr1_filed: GSTR-1 Return node exists for this invoice
    gstr1_filed = has_gstr1

    # gstr3b_filed: Derived — if seller filed GSTR-1 AND GSTR-2B reflected,
    # GSTR-3B is likely filed (GSTR-3B is a summary that includes GSTR-1 data)
    gstr3b_filed = has_gstr1 and has_gstr2b and (not has_mismatch)

    # tax_paid: Derived — if GSTR-3B filed AND no value mismatch,
    # tax is considered paid (no Payment nodes in Neo4j currently)
    tax_paid = gstr3b_filed and (not has_mismatch)

    # irn_valid: Direct check — HAS_IRN relationship exists
    irn_valid = has_irn

    # eway_bill_verified: Direct check — COVERED_BY relationship exists
    eway_bill_verified = has_ewb and (not missing_ewb)

    # route: Extract from EWayBill node if available
    from_state = r.get("from_state")
    to_state = r.get("to_state")
    route = f"{from_state}-{to_state}" if (from_state and to_state) else None

    return {
        "invoice_number": r["invoice_no"],
        "buyer_name": r["buyer_name"],
        "buyer_gstin": r["buyer_gstin"],
        "seller_name": r["seller_name"],
        "seller_gstin": r["seller_gstin"],
        "itc_amount": itc_amount,
        "gstr1_filed": gstr1_filed,
        "gstr3b_filed": gstr3b_filed,
        "tax_paid": tax_paid,
        "irn_valid": irn_valid,
        "eway_bill_verified": eway_bill_verified,
        "route": route,
        # Extra fields for frontend (not part of ValidationInput)
        "_taxable_value": taxable_value,
        "_gst_rate": gst_rate,
        "_period": period if (period := r.get("period")) else "Unknown",
        "_has_gstr2b": has_gstr2b,
        "_has_mismatch": has_mismatch,
    }


def _run_audit_for_invoice(invoice_no: str) -> dict:
    """
    Extract data from Neo4j, run the audit engine, return a rich result dict.
    """
    raw = _extract_invoice_data(invoice_no)
    if not raw:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_no} not found in Neo4j")

    # Separate extra fields from ValidationInput fields
    extra = {k: raw.pop(k) for k in list(raw.keys()) if k.startswith("_")}

    # Run the audit engine
    vi = ValidationInput(**raw)
    result: AuditResult = generate_audit_report(vi)

    # Build response with audit result + extra context
    resp = result.model_dump()
    resp["period"] = extra.get("_period", "Unknown")
    resp["taxable_value"] = extra.get("_taxable_value", 0.0)
    resp["gst_rate"] = extra.get("_gst_rate", 18.0)
    resp["has_gstr2b"] = extra.get("_has_gstr2b", False)
    resp["has_mismatch"] = extra.get("_has_mismatch", False)

    # Map engine checkpoints to frontend 6-hop format
    cp_map = {cp["name"]: cp["status"] for cp in resp["checkpoints"]}
    resp["hops"] = {
        "invoice": True,  # Invoice always exists if we found it
        "irn": cp_map.get("IRN") == "PASS",
        "ewayBill": cp_map.get("EWB") in ("PASS", "SKIPPED"),
        "gstr2b": extra.get("_has_gstr2b", False),
        "gstr3b": cp_map.get("GSTR3B") == "PASS",
        "payment": cp_map.get("TAX") == "PASS",
    }

    return resp


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@audit_engine_router.post("/generate/{invoice_no}")
def generate_audit(invoice_no: str):
    """
    Generate a full structured audit report for a single invoice.
    Extracts all data from Neo4j and runs the 5-step compliance chain.

    Returns: AuditResult + hops mapping + extra context fields.
    """
    return _run_audit_for_invoice(invoice_no)


@audit_engine_router.post("/batch")
def generate_batch_audit(limit: int = Query(50, le=200)):
    """
    Run audit engine on ALL invoices (or those with mismatches).
    Caches results for GET /results.

    Returns: list of AuditResults with summary stats.
    """
    global _cached_audit_results

    # Get all invoices (prioritize those with mismatches)
    query = """
        MATCH (seller:Taxpayer)-[:ISSUED]->(inv:Invoice)
        MATCH (buyer:Taxpayer)-[:RECEIVED]->(inv)
        RETURN inv.invoice_no AS invoice_no
        ORDER BY inv.has_mismatch DESC, inv.taxable_value DESC
        LIMIT $limit
    """
    rows = run_query(query, {"limit": limit})

    results = []
    errors = []
    for r in rows:
        inv_no = r["invoice_no"]
        try:
            audit = _run_audit_for_invoice(inv_no)
            results.append(audit)
        except Exception as e:
            errors.append({"invoice_no": inv_no, "error": str(e)})

    # Cache results
    _cached_audit_results = results

    # Summary stats
    total = len(results)
    approved = sum(1 for r in results if r["decision"] == "ITC APPROVED")
    rejected = total - approved
    by_risk = {}
    for r in results:
        lvl = r["risk_level"]
        by_risk[lvl] = by_risk.get(lvl, 0) + 1

    avg_score = round(sum(r["risk_score"] for r in results) / total, 1) if total else 0

    return {
        "total_audited": total,
        "approved": approved,
        "rejected": rejected,
        "by_risk_level": by_risk,
        "avg_risk_score": avg_score,
        "errors": errors,
        "results": results,
    }


@audit_engine_router.get("/results")
def get_cached_results(
    risk_level: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """
    Return cached batch audit results with optional filtering.
    Call POST /batch first to populate results.
    """
    results = _cached_audit_results

    if risk_level:
        results = [r for r in results if r["risk_level"] == risk_level.upper()]
    if decision:
        decision_upper = decision.upper()
        results = [r for r in results if decision_upper in r["decision"].upper()]

    return {
        "total": len(results),
        "results": results[:limit],
    }


@audit_engine_router.get("/summary")
def get_audit_summary():
    """
    Return high-level summary stats from the latest batch run.
    """
    results = _cached_audit_results
    if not results:
        return {
            "status": "no_data",
            "message": "No audit results. Run POST /audit-engine/batch first.",
        }

    total = len(results)
    approved = sum(1 for r in results if r["decision"] == "ITC APPROVED")
    rejected = total - approved

    by_risk = {}
    for r in results:
        lvl = r["risk_level"]
        by_risk[lvl] = by_risk.get(lvl, 0) + 1

    # Top failures
    checkpoint_failures = {}
    for r in results:
        for cp in r["checkpoints"]:
            if cp["status"] == "FAIL":
                name = cp["name"]
                checkpoint_failures[name] = checkpoint_failures.get(name, 0) + 1

    avg_score = round(sum(r["risk_score"] for r in results) / total, 1) if total else 0

    return {
        "total_audited": total,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approved / total * 100, 1) if total else 0,
        "by_risk_level": by_risk,
        "avg_risk_score": avg_score,
        "checkpoint_failures": checkpoint_failures,
    }
