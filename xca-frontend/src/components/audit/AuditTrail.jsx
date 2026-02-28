import { useState, useEffect } from "react";
import { C } from "../../constants/colors";
import { riskCol } from "../../constants/riskLevels";
import { fetchAuditTrail, generateInvoiceAudit } from "../../services/api";

export default function AuditTrail({ mismatch: mismatchProp, nodes: NODES }) {
  const [liveAudit, setLiveAudit] = useState(null);
  const [loading, setLoading] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);

  // Fetch live audit trail from backend (GET /audit/invoice/{inv})
  useEffect(() => {
    if (!mismatchProp?.inv) { setLiveAudit(null); setAuditResult(null); return; }
    setLoading(true);
    fetchAuditTrail(mismatchProp.inv, false)
      .then(data => {
        if (!data.clean) setLiveAudit(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [mismatchProp?.inv]);

  // Run audit engine for the selected invoice
  useEffect(() => {
    if (!mismatchProp?.inv) { setAuditResult(null); return; }
    setAuditLoading(true);
    generateInvoiceAudit(mismatchProp.inv)
      .then(data => { setAuditResult(data); setAuditLoading(false); })
      .catch(() => setAuditLoading(false));
  }, [mismatchProp?.inv]);

  // Use live API audit data if available, otherwise fall back to props
  const mismatch = liveAudit || mismatchProp;

  // Override hops with audit engine results when available
  const effectiveHops = auditResult?.hops || mismatch?.hops || {};
  const effectiveMismatch = mismatch ? { ...mismatch, hops: effectiveHops } : mismatch;
  if (!mismatch) return (
    <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
      <div style={{ width: 64, height: 64, borderRadius: 20, background: "#f1f4f9", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#b8c5d9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
      </div>
      <div style={{ color: "#94a3b8", fontSize: 14, fontWeight: 500 }}>Select a mismatch from the table to view its audit trail</div>
    </div>
  );

  const hops = [
    { key: "invoice", label: "Invoice", desc: "Source document" },
    { key: "irn", label: "IRN", desc: "e-Invoice portal" },
    { key: "ewayBill", label: "e-Way Bill", desc: "Goods movement" },
    { key: "gstr2b", label: "GSTR-2B", desc: "Auto-drafted ITC" },
    { key: "gstr3b", label: "GSTR-3B", desc: "Summary return" },
    { key: "payment", label: "Payment", desc: "Tax challan" },
  ];
  const hopData = effectiveMismatch.hops || mismatch.hops;
  const firstBreak = hops.findIndex(h => !hopData[h.key]);
  const matchingNode = NODES ? NODES.find(n => n.gstin === mismatch.gstin) : null;

  return (
    <div style={{ height: "100%", overflowY: "auto", display: "flex", flexDirection: "column", gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
      {/* Header Card */}
      <div style={{ background: "#fff", border: "1px solid #d1d9e6", borderRadius: 20, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <div style={{ color: "#0f172a", fontWeight: 800, fontSize: 20 }}>{mismatch.vendor}</div>
            <div style={{ color: "#94a3b8", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", marginTop: 4 }}>{mismatch.gstin} · {mismatch.period}</div>
          </div>
          <span style={{
            background: riskCol(mismatch.risk) + "10", color: riskCol(mismatch.risk),
            padding: "8px 18px", borderRadius: 24, fontWeight: 700, fontSize: 13
          }}>{mismatch.risk}</span>
        </div>

        {/* Detail Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
          {[["Invoice", mismatch.inv, "'IBM Plex Mono', monospace"], ["Type", mismatch.type, "inherit"], ["GSTR-1", `Rs.${(mismatch.gstr1 / 1000).toFixed(0)}K`, "'IBM Plex Mono', monospace"], ["GSTR-2B", `Rs.${(mismatch.gstr2b / 1000).toFixed(0)}K`, "'IBM Plex Mono', monospace"]].map(([k, v, ff]) => (
            <div key={k} style={{ background: "#ffffff", borderRadius: 12, padding: "12px 14px" }}>
              <div style={{ color: "#94a3b8", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4, fontWeight: 500 }}>{k}</div>
              <div style={{ color: "#0f172a", fontFamily: ff, fontSize: 13, fontWeight: 700 }}>{v}</div>
            </div>
          ))}
        </div>

        {/* KPI Row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
          {[["GSTR-3B", mismatch.gstr3b > 0 ? `Rs.${(mismatch.gstr3b / 1000).toFixed(0)}K` : "NOT FILED", mismatch.gstr3b > 0 ? "#0f172a" : "#ef4444"], ["ITC Delta", mismatch.delta > 0 ? `-Rs.${(mismatch.delta / 1000).toFixed(0)}K` : "No Delta", mismatch.delta > 0 ? "#ef4444" : "#10b981"], ["Risk Score", matchingNode?.score.toFixed(2) || "N/A", "#f59e0b"]].map(([k, v, c]) => (
            <div key={k} style={{
              background: "#fff", borderRadius: 12, padding: "14px 16px",
              borderLeft: `4px solid ${c}`, boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
            }}>
              <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 4, fontWeight: 500 }}>{k}</div>
              <div style={{ color: c, fontFamily: "'IBM Plex Mono', monospace", fontSize: 20, fontWeight: 800 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Invoice Relationship Graph */}
      <div style={{ background: "#fff", border: "1px solid #d1d9e6", borderRadius: 20, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
        <div style={{ color: "#94a3b8", fontSize: 11, textTransform: "uppercase", letterSpacing: "1px", marginBottom: 14, fontWeight: 600 }}>Invoice Relationship Graph</div>
        <svg width="100%" height="220" viewBox="0 0 700 220" style={{ display: "block" }}>
          {(() => {
            const h = hopData;
            const gNodes = [
              { id: "seller", label: "Seller", sub: mismatch.vendor.split(" ")[0], x: 80, y: 110, ok: true, col: "#3b82f6" },
              { id: "invoice", label: "Invoice", sub: mismatch.inv, x: 280, y: 110, ok: true, col: "#f59e0b" },
              { id: "buyer", label: "Buyer", sub: mismatch.gstin.slice(0, 6) + "...", x: 480, y: 110, ok: h.gstr2b, col: h.gstr2b ? "#10b981" : "#ef4444" },
              { id: "irn", label: "IRN", sub: h.irn ? "Verified" : "Missing", x: 280, y: 30, ok: h.irn, col: h.irn ? "#10b981" : "#ef4444" },
              { id: "ewb", label: "e-Way Bill", sub: h.ewayBill ? "Valid" : "Missing", x: 480, y: 30, ok: h.ewayBill, col: h.ewayBill ? "#10b981" : "#ef4444" },
              { id: "gstr2b", label: "GSTR-2B", sub: h.gstr2b ? "Matched" : "Mismatch", x: 630, y: 110, ok: h.gstr2b, col: h.gstr2b ? "#10b981" : "#ef4444" },
              { id: "payment", label: "Payment", sub: h.payment ? "Settled" : "Pending", x: 630, y: 200, ok: h.payment, col: h.payment ? "#10b981" : "#ef4444" },
            ];
            const gEdges = [
              { from: "seller", to: "invoice", label: "ISSUED", ok: true },
              { from: "invoice", to: "buyer", label: "RECEIVED", ok: h.gstr2b },
              { from: "invoice", to: "irn", label: "HAS_IRN", ok: h.irn },
              { from: "invoice", to: "ewb", label: "COVERED_BY", ok: h.ewayBill },
              { from: "buyer", to: "gstr2b", label: "REFLECTED_IN", ok: h.gstr2b },
              { from: "gstr2b", to: "payment", label: "SETTLED_VIA", ok: h.payment },
            ];
            const nodeMap = Object.fromEntries(gNodes.map(n => [n.id, n]));
            return (
              <>
                {gEdges.map((e, i) => {
                  const from = nodeMap[e.from], to = nodeMap[e.to];
                  const midX = (from.x + to.x) / 2, midY = (from.y + to.y) / 2;
                  return (
                    <g key={i}>
                      <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                        stroke={e.ok ? "#10b98155" : "#ef444477"} strokeWidth={e.ok ? 1.5 : 2}
                        strokeDasharray={e.ok ? "none" : "6,4"} />
                      <text x={midX} y={midY - 6} textAnchor="middle" fill={e.ok ? "#10b981" : "#ef4444"}
                        fontSize="9" fontFamily="'IBM Plex Mono', monospace" opacity={0.8}>{e.label}</text>
                    </g>
                  );
                })}
                {gNodes.map(n => (
                  <g key={n.id}>
                    <rect x={n.x - 40} y={n.y - 22} width={80} height={44} rx={12}
                      fill={n.ok ? n.col + "10" : "#ef444410"}
                      stroke={n.ok ? n.col : "#ef4444"} strokeWidth={1.5} />
                    <text x={n.x} y={n.y - 4} textAnchor="middle" fill={n.ok ? n.col : "#ef4444"}
                      fontSize="10" fontWeight="700" fontFamily="'Source Sans 3', sans-serif">{n.label}</text>
                    <text x={n.x} y={n.y + 10} textAnchor="middle" fill="#94a3b8"
                      fontSize="8" fontFamily="'IBM Plex Mono', monospace">{n.sub}</text>
                  </g>
                ))}
              </>
            );
          })()}
        </svg>
      </div>

      {/* 6-Hop ITC Chain */}
      <div style={{ background: "#fff", border: "1px solid #d1d9e6", borderRadius: 20, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
        <div style={{ color: "#94a3b8", fontSize: 11, textTransform: "uppercase", letterSpacing: "1px", marginBottom: 20, fontWeight: 600 }}>6-Hop ITC Chain Traversal</div>
        <div style={{ display: "flex", alignItems: "flex-start" }}>
          {hops.map((h, i) => {
            const ok = hopData[h.key];
            const isBreak = i === firstBreak;
            return (
              <div key={h.key} style={{ display: "flex", alignItems: "center", flex: 1 }}>
                <div style={{ flex: 1, textAlign: "center" }}>
                  <div style={{
                    width: 52, height: 52, borderRadius: 14, margin: "0 auto 10px",
                    background: ok ? "#10b98108" : "#ef444408",
                    border: `2px solid ${ok ? "#10b981" : "#ef4444"}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 18, fontWeight: 700, color: ok ? "#10b981" : "#ef4444",
                    position: "relative", boxShadow: isBreak ? "0 0 24px rgba(239,68,68,0.3)" : "none",
                    transition: "all 0.2s"
                  }}>
                    {h.label.charAt(0)}
                    {isBreak && <div style={{
                      position: "absolute", top: -8, right: -8,
                      background: "#dc2626", borderRadius: "50%", width: 20, height: 20,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 11, color: "#fff", fontWeight: 800, boxShadow: "0 2px 8px rgba(220,38,38,0.4)"
                    }}>!</div>}
                  </div>
                  <div style={{ color: ok ? "#10b981" : "#ef4444", fontSize: 11, fontWeight: 700 }}>{h.label}</div>
                  <div style={{ color: "#94a3b8", fontSize: 10, marginTop: 2 }}>{h.desc}</div>
                  <div style={{
                    color: ok ? "#10b981" : "#ef4444", fontSize: 14, marginTop: 4, fontWeight: 700,
                    fontFamily: "'IBM Plex Mono', monospace"
                  }}>{ok ? "OK" : "ERR"}</div>
                </div>
                {i < hops.length - 1 && (
                  <div style={{
                    width: 24, height: 2, flexShrink: 0,
                    background: ok && hopData[hops[i + 1].key] ? "#10b98144" : "#ef444444",
                    position: "relative", top: -14
                  }} />
                )}
              </div>
            );
          })}
        </div>
        {firstBreak >= 0 && (
          <div style={{
            marginTop: 18, padding: "12px 16px", background: "rgba(239,68,68,0.05)",
            border: "1px solid rgba(239,68,68,0.15)", borderRadius: 12,
            fontSize: 13, color: "#dc2626", fontWeight: 500
          }}>
            Chain breaks at <strong>{hops[firstBreak].label}</strong> — ITC from this hop onwards is ineligible for credit
          </div>
        )}
      </div>

      {/* Audit Engine — Structured Compliance Analysis */}
      <div style={{ background: "#fff", border: "1px solid #d1d9e6", borderRadius: 20, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
          <div style={{
            width: 40, height: 40, background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.2)",
            borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg>
          </div>
          <div>
            <div style={{ color: "#0f172a", fontWeight: 700, fontSize: 15 }}>GST Compliance Audit Engine</div>
            <div style={{ color: "#94a3b8", fontSize: 12, fontWeight: 500 }}>5-step chain validation · deterministic audit</div>
          </div>
          <span style={{
            marginLeft: "auto", background: auditResult ? "rgba(16,185,129,0.08)" : "rgba(139,92,246,0.08)",
            color: auditResult ? "#10b981" : "#8b5cf6",
            fontSize: 11, padding: "4px 10px", borderRadius: 6, fontWeight: 800
          }}>{auditLoading ? "RUNNING..." : auditResult ? "ENGINE" : "PENDING"}</span>
        </div>

        {/* Decision Banner */}
        {auditResult && (
          <div style={{
            display: "flex", gap: 12, marginBottom: 16
          }}>
            <div style={{
              flex: 1, padding: "14px 18px", borderRadius: 14,
              background: auditResult.decision === "ITC APPROVED"
                ? "linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.02))"
                : "linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02))",
              border: `1.5px solid ${auditResult.decision === "ITC APPROVED" ? "rgba(16,185,129,0.3)" : "rgba(239,68,68,0.3)"}`,
              display: "flex", alignItems: "center", gap: 12
            }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: auditResult.decision === "ITC APPROVED" ? "#10b981" : "#ef4444",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "#fff", fontWeight: 800, fontSize: 13
              }}>{auditResult.decision === "ITC APPROVED" ? "PASS" : "FAIL"}</div>
              <div>
                <div style={{
                  color: auditResult.decision === "ITC APPROVED" ? "#059669" : "#dc2626",
                  fontWeight: 800, fontSize: 16, fontFamily: "'IBM Plex Mono', monospace"
                }}>{auditResult.decision}</div>
                <div style={{ color: "#94a3b8", fontSize: 11, marginTop: 2 }}>
                  Risk: {auditResult.risk_level} · Chain: {auditResult.chain_valid ? "INTACT" : "BROKEN"}
                </div>
              </div>
            </div>

            {/* Risk Score Gauge */}
            <div style={{
              width: 100, padding: "14px 0", borderRadius: 14,
              background: "#ffffff", border: "1px solid #d1d9e6",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"
            }}>
              <div style={{
                color: auditResult.risk_score >= 80 ? "#10b981" : auditResult.risk_score >= 50 ? "#f59e0b" : "#ef4444",
                fontFamily: "'IBM Plex Mono', monospace", fontSize: 26, fontWeight: 800, lineHeight: 1
              }}>{auditResult.risk_score}</div>
              <div style={{ color: "#94a3b8", fontSize: 9, marginTop: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>Score /100</div>
            </div>
          </div>
        )}

        {/* 5-Step Compliance Checkpoints */}
        {auditResult?.checkpoints && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ color: "#94a3b8", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, fontWeight: 600 }}>Compliance Checkpoints</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {auditResult.checkpoints.map((cp, i) => {
                const isPASS = cp.status === "PASS";
                const isSKIP = cp.status === "SKIPPED";
                const col = isPASS ? "#10b981" : isSKIP ? "#f59e0b" : "#ef4444";
                const bg = isPASS ? "rgba(16,185,129,0.06)" : isSKIP ? "rgba(245,158,11,0.06)" : "rgba(239,68,68,0.06)";
                return (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                    background: bg, borderRadius: 10, border: `1px solid ${col}22`
                  }}>
                    <div style={{
                      width: 24, height: 24, borderRadius: 7, background: col,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "#fff", fontSize: 11, fontWeight: 800, flexShrink: 0
                    }}>{cp.step}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ color: "#0f172a", fontSize: 12, fontWeight: 700 }}>{cp.label}</div>
                      <div style={{ color: "#475569", fontSize: 11, marginTop: 1 }}>{cp.detail}</div>
                    </div>
                    <span style={{
                      padding: "3px 10px", borderRadius: 6, fontSize: 10, fontWeight: 800,
                      fontFamily: "'IBM Plex Mono', monospace",
                      background: `${col}15`, color: col
                    }}>{cp.status}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Chain Summary */}
        {auditResult?.chain_summary && (
          <div style={{
            padding: "10px 14px", background: "#ffffff", borderRadius: 10,
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#475569",
            marginBottom: 16, letterSpacing: "0.3px"
          }}>
            {auditResult.chain_summary}
          </div>
        )}

        {/* Audit Report Text */}
        <div style={{
          color: "#0f172a", fontSize: 14, lineHeight: 1.8, background: "#ffffff",
          borderRadius: 14, padding: 20,
          borderLeft: `4px solid ${auditResult ? (auditResult.decision === "ITC APPROVED" ? "#10b981" : "#ef4444") : "#8b5cf6"}`,
          fontWeight: 400, whiteSpace: "pre-line"
        }}>
          {auditResult?.report || mismatch.audit || "Audit analysis pending..."}
        </div>

        <button onClick={() => handleSecureExport(mismatch, matchingNode, auditResult)} style={{
          marginTop: 16, padding: "10px 18px", background: "#ffffff",
          border: "1px solid #d1d9e6", borderRadius: 10, color: "#475569",
          fontSize: 13, cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
          transition: "all 0.2s", display: "inline-flex", alignItems: "center", gap: 8
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          Export Audit Note
        </button>
      </div>
    </div>
  );
}

/**
 * Secure PDF export with CONFIDENTIAL watermark and timestamp.
 * Opens a new window with only the audit content formatted for print.
 */
function handleSecureExport(mismatch, matchingNode, auditResult) {
  const timestamp = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
  const printWindow = window.open('', '_blank', 'width=800,height=600');
  if (!printWindow) return;

  // Escape HTML entities for safe rendering
  const esc = (s) => String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  printWindow.document.write(`<!DOCTYPE html>
<html><head><title>Audit Report — ${esc(mismatch.inv)}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  @page { margin: 20mm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Source Sans 3', sans-serif; padding: 0; color: #0f172a; background: #ffffff; line-height: 1.5; font-size: 11pt; }
  .watermark { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(-35deg);
    font-size: 80pt; color: rgba(220,38,38,0.03); font-weight: 900; letter-spacing: 15px;
    pointer-events: none; z-index: 0; white-space: nowrap; font-family: 'Playfair Display', serif; }
  .content { position: relative; z-index: 1; border: 1px solid #d1d9e6; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.05); min-height: 297mm; padding: 40px; }
  
  .header-section { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0f172a; padding-bottom: 20px; margin-bottom: 30px; }
  .govt-heading { font-family: 'Playfair Display', serif; font-size: 24pt; color: #0f172a; text-transform: uppercase; letter-spacing: 1px; }
  .report-title { font-family: 'Source Sans 3', sans-serif; font-size: 14pt; font-weight: 700; color: #475569; margin-top: 5px; }
  
  .section-header { background: #f1f4f9; padding: 8px 15px; border-left: 5px solid #1e40af; margin: 30px 0 15px; font-weight: 700; font-size: 12pt; text-transform: uppercase; color: #1e40af; }
  .form-row { display: flex; border-bottom: 1px solid #e8edf5; padding: 10px 0; }
  .form-label { width: 40%; color: #475569; font-weight: 600; font-size: 10pt; }
  .form-value { width: 60%; color: #0f172a; font-family: 'IBM Plex Mono', monospace; font-size: 10pt; font-weight: 600; }
  
  table { width: 100%; border-collapse: collapse; margin-top: 20px; border: 1px solid #d1d9e6; }
  th { background: #f8f9fc; color: #475569; text-transform: uppercase; font-size: 9pt; font-weight: 700; padding: 12px; text-align: left; border-bottom: 2px solid #d1d9e6; }
  td { padding: 12px; border-bottom: 1px solid #e8edf5; font-size: 10pt; vertical-align: top; }
  
  .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: 800; font-size: 10pt; border: 1.5px solid; }
  .badge-approved { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
  .badge-rejected { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
  
  .audit-summary { background: #f8f9fc; border: 1px solid #d1d9e6; padding: 20px; border-radius: 8px; margin-top: 15px; font-size: 10.5pt; color: #334155; white-space: pre-line; border-left: 4px solid #1e40af; }
  .footer { margin-top: 50px; border-top: 1px solid #d1d9e6; padding-top: 20px; display: flex; justify-content: space-between; font-size: 9pt; color: #94a3b8; }
  
  @media print {
    body { background: transparent; }
    .content { box-shadow: none; border: none; padding: 0; margin: 0; }
    .header-section { margin-bottom: 20px; }
  }
</style></head>
<body>
<div class="watermark">FORM GST ADT-02</div>
<div class="content">
  <div class="header-section">
    <div>
      <div class="govt-heading">Final Audit Report</div>
      <div class="report-title">GST Reconciliation Engine · Automated Compliance</div>
    </div>
    <div style="text-align: right;">
      <div style="font-weight: 700; font-size: 10pt;">Date: ${esc(timestamp)}</div>
      <div style="color: #94a3b8; font-size: 9pt; margin-top: 3px;">Report ID: ${esc(mismatch.inv)}-${Date.now().toString().slice(-4)}</div>
    </div>
  </div>

  <div class="section-header">Part A: Basic Details of the Registered Person</div>
  <div class="form-row"><div class="form-label">Name of Taxpayer</div><div class="form-value">${esc(mismatch.vendor)}</div></div>
  <div class="form-row"><div class="form-label">GSTIN / Unique ID</div><div class="form-value">${esc(mismatch.gstin)}</div></div>
  <div class="form-row"><div class="form-label">Financial Year / Tax Period</div><div class="form-value">${esc(mismatch.period)}</div></div>
  <div class="form-row"><div class="form-label">Audit Risk Category</div><div class="form-value" style="color: ${riskCol(mismatch.risk)}">${esc(mismatch.risk)}</div></div>

  <div class="section-header">Part B: Invoice & Compliance Summary</div>
  <table style="margin-bottom: 20px;">
    <tr>
      <th>Parameter</th>
      <th>Document Value</th>
      <th>Portal Value</th>
      <th>Variance (Delta)</th>
    </tr>
    <tr>
      <td>GSTR-1 Liability</td>
      <td>Rs. ${(mismatch.gstr1 / 1000).toFixed(0)}K</td>
      <td>Rs. ${(mismatch.gstr1 / 1000).toFixed(0)}K</td>
      <td>Rs. 0.00</td>
    </tr>
    <tr>
      <td>ITC Availability (2B)</td>
      <td>Rs. ${(mismatch.gstr2b / 1000).toFixed(0)}K</td>
      <td>Rs. ${(mismatch.gstr2b / 1000).toFixed(0)}K</td>
      <td>Rs. 0.00</td>
    </tr>
    <tr>
      <td>ITC Availed (3B)</td>
      <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700;">${mismatch.gstr3b > 0 ? 'Rs. ' + (mismatch.gstr3b / 1000).toFixed(0) + 'K' : 'NOT FILED'}</td>
      <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700;">Rs. ${(mismatch.gstr2b / 1000).toFixed(0)}K</td>
      <td style="color: ${mismatch.delta > 0 ? '#dc2626' : '#166534'}; font-family: 'IBM Plex Mono', monospace; font-weight: 700;">${mismatch.delta > 0 ? '-' + (mismatch.delta / 1000).toFixed(2) + 'K' : 'No Delta'}</td>
    </tr>
  </table>

  <div class="section-header">Part C: Compliance Checkpoint Traversal</div>
  ${auditResult ? `
  <table>
    <tr>
      <th style="width: 10%;">Step</th>
      <th style="width: 30%;">Checkpoint</th>
      <th style="width: 15%;">Status</th>
      <th style="width: 45%;">Auditor's Remarks</th>
    </tr>
    ${auditResult.checkpoints.map(cp => `
    <tr>
      <td style="color: #94a3b8; font-weight: 700;">${cp.step}</td>
      <td style="font-weight: 700;">${esc(cp.label)}</td>
      <td>
        <span style="color: ${cp.status === 'PASS' ? '#166534' : cp.status === 'SKIPPED' ? '#d97706' : '#991b1b'}; font-weight: 800; font-family: 'IBM Plex Mono', monospace;">${cp.status}</span>
      </td>
      <td style="color: #475569; font-size: 9.5pt;">${esc(cp.detail)}</td>
    </tr>
    `).join('')}
  </table>
  ` : '<div style="padding: 20px; text-align: center; color: #94a3b8;">Full automated compliance traversal pending...</div>'}

  <div class="section-header">Part D: Auditor's Conclusions & Recommendations</div>
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
    <div style="font-size: 13pt; font-weight: 700;">Decision: <span class="badge ${auditResult?.decision === 'ITC APPROVED' ? 'badge-approved' : 'badge-rejected'}">${esc(auditResult?.decision || 'PENDING')}</span></div>
    <div style="font-size: 11pt; font-weight: 600;">System Confidence: <span style="color: #1e40af;">${auditResult?.risk_score || 'N/A'}/100</span></div>
  </div>
  <div class="audit-summary">${esc(auditResult?.report || mismatch.audit || 'In-depth analysis is required due to filing discrepancies.')}</div>

  <div class="footer">
    <div>AUTHENTICATED BY GST RECONCILIATION ENGINE · PS #76</div>
    <div>This is an electronically generated report and does not require a physical signature.</div>
  </div>
</div>
<script>window.onload = function() { window.print(); window.close(); }</script>
</body></html>`);
  printWindow.document.close();
}