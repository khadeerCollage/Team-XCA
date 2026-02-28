/**
 * api.js — Backend API Service
 * ═══════════════════════════════════════════════════════════
 * Connects ALL FastAPI endpoints to the React frontend.
 * Transforms backend response shapes → frontend component shapes.
 *
 * All calls go through Vite proxy: /api/* → http://localhost:8000/*
 *
 * Endpoints covered:
 *   GET  /health
 *   GET  /taxpayers/           GET /taxpayers/{gstin}
 *   GET  /invoices/            GET /invoices/{invoice_no}
 *   POST /reconcile/run        GET /reconcile/mismatches      GET /reconcile/summary
 *   GET  /risk/vendor/{gstin}  GET /risk/leaderboard
 *   GET  /audit/invoice/{inv}
 *   GET  /graph/nodes-edges
 *   POST /pipeline/generate    POST /pipeline/run    GET /pipeline/mismatches
 *   POST /ml/train  POST /ml/predict  GET /ml/scores  GET /ml/feature-importances
 *   POST /audit-engine/generate/{inv}  POST /audit-engine/batch  GET /audit-engine/results  GET /audit-engine/summary
 */

const API = '/api';

// ═══════════════════════════════════════════════════════════
//  HEALTH
// ═══════════════════════════════════════════════════════════

export async function fetchHealth() {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error('Backend unreachable');
  return res.json();
}

// ═══════════════════════════════════════════════════════════
//  TAXPAYERS — /taxpayers
// ═══════════════════════════════════════════════════════════

export async function fetchTaxpayers({ state, risk_level, limit = 50 } = {}) {
  const params = new URLSearchParams();
  if (state) params.set('state', state);
  if (risk_level) params.set('risk_level', risk_level);
  params.set('limit', String(limit));
  const res = await fetch(`${API}/taxpayers/?${params}`);
  if (!res.ok) throw new Error('Failed to fetch taxpayers');
  return res.json();
}

export async function fetchTaxpayer(gstin) {
  const res = await fetch(`${API}/taxpayers/${encodeURIComponent(gstin)}`);
  if (!res.ok) throw new Error(`Taxpayer ${gstin} not found`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════
//  INVOICES — /invoices
// ═══════════════════════════════════════════════════════════

export async function fetchInvoices({ period, seller_gstin, buyer_gstin, limit = 50 } = {}) {
  const params = new URLSearchParams();
  if (period) params.set('period', period);
  if (seller_gstin) params.set('seller_gstin', seller_gstin);
  if (buyer_gstin) params.set('buyer_gstin', buyer_gstin);
  params.set('limit', String(limit));
  const res = await fetch(`${API}/invoices/?${params}`);
  if (!res.ok) throw new Error('Failed to fetch invoices');
  return res.json();
}

export async function fetchInvoice(invoiceNo) {
  const res = await fetch(`${API}/invoices/${encodeURIComponent(invoiceNo)}`);
  if (!res.ok) throw new Error(`Invoice ${invoiceNo} not found`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════
//  RECONCILIATION — /reconcile  (the core endpoint)
// ═══════════════════════════════════════════════════════════

/**
 * Trigger full reconciliation on the backend.
 * This runs the 4-hop graph traversal and detects all mismatches.
 */
export async function runReconciliation(period = null, useAi = false) {
  const res = await fetch(`${API}/reconcile/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ period, run_audit_ai: useAi }),
  });
  if (!res.ok) throw new Error('Reconciliation failed');
  return res.json();
}

/**
 * Fetch mismatches from the reconciliation engine.
 * Returns data transformed to the shape MismatchTable expects.
 */
export async function fetchMismatches({ risk_level, mismatch_type, gstin, limit = 200 } = {}) {
  const params = new URLSearchParams();
  if (risk_level && risk_level !== 'ALL') params.set('risk_level', risk_level);
  if (mismatch_type) params.set('mismatch_type', mismatch_type);
  if (gstin) params.set('gstin', gstin);
  params.set('limit', String(limit));
  const res = await fetch(`${API}/reconcile/mismatches?${params}`);
  if (!res.ok) throw new Error('Failed to fetch mismatches');
  const data = await res.json();
  return data.map(transformMismatch);
}

/**
 * Fetch reconciliation summary stats for the dashboard header.
 */
export async function fetchReconcileSummary(period = null) {
  const params = new URLSearchParams();
  if (period) params.set('period', period);
  const res = await fetch(`${API}/reconcile/summary?${params}`);
  if (!res.ok) throw new Error('Failed to fetch summary');
  return res.json();
}

// ═══════════════════════════════════════════════════════════
//  VENDOR RISK — /risk
// ═══════════════════════════════════════════════════════════

/**
 * Fetch risk leaderboard — all vendors sorted by risk score.
 * Returns data transformed to the shape VendorScorecard expects.
 */
export async function fetchRiskLeaderboard(limit = 50) {
  const res = await fetch(`${API}/risk/leaderboard?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch risk leaderboard');
  const data = await res.json();
  return data.map(transformVendor);
}

/**
 * Fetch risk score for a single vendor.
 */
export async function fetchVendorRisk(gstin) {
  const res = await fetch(`${API}/risk/vendor/${encodeURIComponent(gstin)}`);
  if (!res.ok) throw new Error(`Vendor ${gstin} not found`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════
//  AUDIT TRAIL — /audit
// ═══════════════════════════════════════════════════════════

/**
 * Fetch audit trail for a specific invoice.
 * Returns a mismatch object in frontend shape, or a "clean" status.
 */
export async function fetchAuditTrail(invoiceNo, useAi = false) {
  const res = await fetch(
    `${API}/audit/invoice/${encodeURIComponent(invoiceNo)}?use_ai=${useAi}`
  );
  if (!res.ok) throw new Error('Failed to fetch audit trail');
  const data = await res.json();

  // Backend returns { status: "CLEAN", ... } for clean invoices
  if (data.status === 'CLEAN') {
    return {
      clean: true,
      invoice_no: data.invoice_no,
      audit: data.audit_note || data.message,
      hops: transformChainHops(data.chain_hops || []),
    };
  }

  // Otherwise it's a MismatchRecord — transform to frontend shape
  return transformMismatch(data, 0);
}

// ═══════════════════════════════════════════════════════════
//  GRAPH VISUALIZATION — /graph
// ═══════════════════════════════════════════════════════════

/**
 * Fetch live graph nodes + edges from Neo4j.
 * Note: This returns a simpler node shape than graph_output.json.
 * Use for real-time graph refresh.
 */
export async function fetchGraphNodesEdges({ period, limit_nodes = 100 } = {}) {
  const params = new URLSearchParams();
  if (period) params.set('period', period);
  params.set('limit_nodes', String(limit_nodes));
  const res = await fetch(`${API}/graph/nodes-edges?${params}`);
  if (!res.ok) throw new Error('Failed to fetch graph data');
  return res.json();
}


// ═══════════════════════════════════════════════════════════
//  DATA TRANSFORMERS
//  Backend response shapes → Frontend component shapes
// ═══════════════════════════════════════════════════════════

/**
 * Transform backend MismatchRecord → frontend mismatch shape.
 *
 * Backend: { mismatch_id, seller_name, seller_gstin, invoice_no,
 *            gstr1_value, gstr2b_value, delta, risk_level,
 *            mismatch_type, chain_hops[], audit_note }
 *
 * Frontend: { id, vendor, gstin, inv, gstr1, gstr2b, gstr3b,
 *             delta, risk, type, period, hops:{}, audit }
 */
function transformMismatch(m, idx) {
  // Build the 6-hop chain map from backend chain_hops array
  const hops = {
    invoice: true, irn: true, ewayBill: true,
    gstr2b: true, gstr3b: true, payment: true,
  };

  if (m.chain_hops && m.chain_hops.length > 0) {
    for (const hop of m.chain_hops) {
      const key = hopNameToKey(hop.hop_name);
      if (key) hops[key] = hop.status;
    }
  }

  // If chain is broken, mark everything after the break as false
  if (m.chain_broken_at) {
    const breakKey = hopNameToKey(m.chain_broken_at);
    if (breakKey) {
      const hopOrder = ['invoice', 'irn', 'ewayBill', 'gstr2b', 'gstr3b', 'payment'];
      const breakIdx = hopOrder.indexOf(breakKey);
      if (breakIdx >= 0) {
        for (let i = breakIdx; i < hopOrder.length; i++) {
          hops[hopOrder[i]] = false;
        }
      }
    }
  }

  return {
    id: idx + 1,
    vendor: m.seller_name || 'Unknown Vendor',
    gstin: m.seller_gstin || '',
    inv: m.invoice_no || '',
    gstr1: m.gstr1_value || 0,
    gstr2b: m.gstr2b_value || 0,
    gstr3b: hops.gstr3b ? (m.gstr2b_value || 0) : 0,
    delta: Math.abs(m.delta || 0),
    risk: m.risk_level || 'MEDIUM',
    type: m.mismatch_type || 'Value Delta',
    period: m.period || '',
    hops,
    audit: m.audit_note ||
      `Mismatch detected on invoice ${m.invoice_no}. ` +
      `The verification chain broke at ${m.chain_broken_at || 'unknown stage'}. ` +
      `Review recommended for the ${m.period} period.`,
  };
}

/**
 * Transform backend VendorRiskScore → frontend node shape.
 *
 * Backend: { gstin, name, state, mismatch_rate, avg_filing_delay,
 *            itc_overclaim_ratio, circular_txn_flag, new_vendor_flag,
 *            risk_score, risk_level, total_transactions, total_itc_at_risk }
 *
 * Frontend: { id, label, gstin, state, risk, score, itc, tx,
 *             mismatchRate, delay, overclaim, circular }
 */
function transformVendor(v, idx) {
  const risk = (v.risk_level || 'low').toLowerCase();
  return {
    id: `V${String(idx + 1).padStart(2, '0')}`,
    label: v.name || 'Unknown',
    gstin: v.gstin || '',
    state: v.state || 'N/A',
    risk,
    score: v.risk_score || 0,
    itc: v.total_itc_at_risk || 0,
    tx: v.total_transactions || 0,
    mismatchRate: v.mismatch_rate || 0,
    delay: Math.round((v.avg_filing_delay || 0) * 90), // normalized → days
    overclaim: v.itc_overclaim_ratio || 0,
    circular: v.circular_txn_flag || 0,
  };
}

/**
 * Transform backend chain_hops array → frontend hops map.
 * Backend: [{ hop_name: "Invoice", status: true }, ...]
 * Frontend: { invoice: true, irn: true, ewayBill: false, ... }
 */
function transformChainHops(chainHops) {
  const hops = {
    invoice: true, irn: true, ewayBill: true,
    gstr2b: true, gstr3b: true, payment: true,
  };
  for (const hop of chainHops) {
    const key = hopNameToKey(hop.hop_name);
    if (key) hops[key] = hop.status;
  }
  return hops;
}

/**
 * Map backend hop names → frontend hop keys.
 */
function hopNameToKey(name) {
  const map = {
    'Invoice': 'invoice',
    'IRN': 'irn',
    'e-Way Bill': 'ewayBill',
    'GSTR-1': null,       // No direct frontend hop for GSTR-1
    'GSTR-2B': 'gstr2b',
    'GSTR-3B': 'gstr3b',
    'Payment': 'payment',
  };
  return map[name] !== undefined ? map[name] : null;
}


// ═══════════════════════════════════════════════════════════
//  ML PREDICTIONS
// ═══════════════════════════════════════════════════════════

/**
 * POST /ml/train — Train XGBoost model on Neo4j graph features.
 * Returns: { status, message, accuracy, f1_score, feature_importances }
 */
export async function trainMLModel() {
  const res = await fetch(`${API}/ml/train`, { method: 'POST' });
  if (!res.ok) throw new Error(`ML train failed: ${res.status}`);
  return res.json();
}

/**
 * POST /ml/predict — Run ML inference for all vendors.
 * Returns: { predictions[], feature_importances, model_used }
 */
export async function runMLPredictions() {
  const res = await fetch(`${API}/ml/predict`, { method: 'POST' });
  if (!res.ok) throw new Error(`ML predict failed: ${res.status}`);
  return res.json();
}

/**
 * GET /ml/scores — Get cached ML predictions (leaderboard).
 * @param {number} limit
 * @returns {Promise<Array>} vendor predictions
 */
export async function fetchMLScores(limit = 100) {
  const res = await fetch(`${API}/ml/scores?limit=${limit}`);
  if (!res.ok) throw new Error(`ML scores fetch failed: ${res.status}`);
  return res.json();
}

/**
 * GET /ml/scores/{gstin} — Single vendor ML prediction.
 */
export async function fetchMLVendorScore(gstin) {
  const res = await fetch(`${API}/ml/scores/${encodeURIComponent(gstin)}`);
  if (!res.ok) throw new Error(`ML vendor score failed: ${res.status}`);
  return res.json();
}

/**
 * GET /ml/feature-importances — Learned feature weights from model.
 */
export async function fetchFeatureImportances() {
  const res = await fetch(`${API}/ml/feature-importances`);
  if (!res.ok) return {};
  return res.json();
}


// ═════════════════════════════════════════════════════════
//  PIPELINE DATA  — /pipeline/mismatches
// ═════════════════════════════════════════════════════════

/**
 * GET /pipeline/mismatches — Returns ALL mismatches from graph_output.json.
 * This is the most reliable data source (reads what the pipeline built).
 * Returns: { mismatches[], summary{} }
 */
export async function fetchPipelineMismatches() {
  const res = await fetch(`${API}/pipeline/mismatches`);
  if (!res.ok) throw new Error('Failed to fetch pipeline mismatches');
  return res.json();
}

/**
 * Fetch ALL graph data from the static graph_output.json.
 * Used as fallback when backend is not running.
 */
export async function fetchGraphOutputStatic() {
  const res = await fetch('/graph_output.json');
  if (!res.ok) throw new Error('graph_output.json not found');
  return res.json();
}


// ══════════════════════════════════════════════════════════════
//  AUDIT ENGINE — 5-step GST compliance chain validator
// ══════════════════════════════════════════════════════════════

/**
 * POST /audit-engine/generate/{invoice_no} — Full structured audit for one invoice.
 * Returns: AuditResult with checkpoints[], risk_score, decision, report, hops mapping.
 */
export async function generateInvoiceAudit(invoiceNo) {
  const res = await fetch(`${API}/audit-engine/generate/${encodeURIComponent(invoiceNo)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Audit generation failed: ${res.status}`);
  return res.json();
}

/**
 * POST /audit-engine/batch — Run audit engine on all invoices.
 * Returns: { total_audited, approved, rejected, by_risk_level, results[] }
 */
export async function runBatchAudit(limit = 50) {
  const res = await fetch(`${API}/audit-engine/batch?limit=${limit}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Batch audit failed: ${res.status}`);
  return res.json();
}

/**
 * GET /audit-engine/results — Cached batch results with optional filters.
 */
export async function fetchAuditResults(filters = {}) {
  const params = new URLSearchParams();
  if (filters.risk_level) params.set('risk_level', filters.risk_level);
  if (filters.decision) params.set('decision', filters.decision);
  if (filters.limit) params.set('limit', filters.limit);
  const qs = params.toString();
  const res = await fetch(`${API}/audit-engine/results${qs ? '?' + qs : ''}`);
  if (!res.ok) throw new Error(`Audit results fetch failed: ${res.status}`);
  return res.json();
}

/**
 * GET /audit-engine/summary — High-level audit stats.
 */
export async function fetchAuditSummary() {
  const res = await fetch(`${API}/audit-engine/summary`);
  if (!res.ok) throw new Error(`Audit summary failed: ${res.status}`);
  return res.json();
}
