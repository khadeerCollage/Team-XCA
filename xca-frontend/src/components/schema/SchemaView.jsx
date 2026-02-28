import { useState } from "react";

export default function SchemaView() {
  const nodes = [
    { id: "taxpayer", label: "Taxpayer", color: "#1e40af", x: 50, y: 50, props: ["gstin", "pan", "name", "state", "turnover", "risk_score"] },
    { id: "invoice", label: "Invoice", color: "#16a34a", x: 20, y: 35, props: ["irn", "invoice_no", "date", "taxable_value", "igst", "cgst", "sgst", "status"] },
    { id: "irn", label: "IRN", color: "#d97706", x: 20, y: 65, props: ["irn_no", "ack_no", "ack_date", "signed_invoice", "portal_status"] },
    { id: "return", label: "GST Return", color: "#4f46e5", x: 80, y: 35, props: ["return_type", "period", "filing_date", "total_itc", "tax_paid", "status"] },
    { id: "ewaybill", label: "e-Way Bill", color: "#dc2626", x: 50, y: 15, props: ["ewb_no", "validity_date", "distance_km", "vehicle_no", "transporter_id"] },
    { id: "payment", label: "Payment", color: "#16a34a", x: 80, y: 65, props: ["challan_no", "amount", "date", "tax_head", "cpin", "bank_ref"] },
  ];

  const rels = [
    { from: "taxpayer", to: "invoice", label: "ISSUED", color: "#16a34a" },
    { from: "taxpayer", to: "invoice", label: "RECEIVED", color: "#1e40af" },
    { from: "taxpayer", to: "return", label: "FILED", color: "#4f46e5" },
    { from: "invoice", to: "irn", label: "HAS_IRN", color: "#d97706" },
    { from: "invoice", to: "ewaybill", label: "COVERED_BY", color: "#dc2626" },
    { from: "invoice", to: "return", label: "REPORTED_IN", color: "#4f46e5" },
    { from: "return", to: "payment", label: "SETTLED_VIA", color: "#16a34a" },
    { from: "taxpayer", to: "taxpayer", label: "TRANSACTS_WITH", color: "#475569" },
  ];

  const [hovNode, setHovNode] = useState(null);

  return (
    <div style={{ height: "100%", display: "flex", gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
      {/* Schema Canvas */}
      <div style={{
        flex: 1.4, background: "#f8f9fc", border: "1px solid #d1d9e6", borderRadius: 8,
        padding: 24, position: "relative", overflow: "hidden"
      }}>
        <div className="dot-grid" style={{ position: "absolute", inset: 0, opacity: 0.4, pointerEvents: "none" }} />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ color: "#475569", fontSize: 14, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2, fontWeight: 600 }}>Knowledge Graph Schema</div>
          <div style={{ color: "#0f172a", fontSize: 16, fontWeight: 700, marginBottom: 20, letterSpacing: "-0.3px" }}>GST Ecosystem Data Model</div>
        </div>

        {/* Edges */}
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: 0 }}>
          {rels.filter(r => r.from !== r.to).map((r, i) => {
            const fn = nodes.find(n => n.id === r.from), tn = nodes.find(n => n.id === r.to);
            return (
              <g key={i}>
                <line x1={fn.x + "%"} y1={fn.y + "%"} x2={tn.x + "%"} y2={tn.y + "%"}
                  stroke={r.color} strokeWidth="1" strokeOpacity="0.2" strokeDasharray="6,4" />
                <text x={`${(fn.x + tn.x) / 2}%`} y={`${(fn.y + tn.y) / 2}%`}
                  textAnchor="middle" fill={r.color} fontSize="8" opacity="0.4"
                  fontFamily="'IBM Plex Mono', monospace">{r.label}</text>
              </g>
            );
          })}
        </svg>

        {/* Nodes */}
        {nodes.map(n => (
          <div key={n.id} onMouseEnter={() => setHovNode(n)} onMouseLeave={() => setHovNode(null)} style={{
            position: "absolute", left: `${n.x - 8}%`, top: `${n.y - 6}%`, transform: "translate(-50%, -50%)",
            background: "#ffffff", border: `1px solid ${hovNode?.id === n.id ? n.color : "#d1d9e6"}`,
            borderRadius: 8, padding: "10px 14px", minWidth: 110, cursor: "pointer", zIndex: 2,
            boxShadow: hovNode?.id === n.id ? `0 0 20px ${n.color}22` : "none",
            transition: "border-color 0.2s, box-shadow 0.2s"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: n.color, boxShadow: `0 0 6px ${n.color}66`, display: "inline-block" }} />
              <span style={{ color: "#0f172a", fontWeight: 700, fontSize: 14 }}>{n.label}</span>
            </div>
            <div style={{ color: "#94a3b8", fontSize: 13, fontFamily: "'IBM Plex Mono', monospace" }}>{n.props.length} properties</div>
          </div>
        ))}

        <div style={{ position: "absolute", bottom: 16, right: 20, background: "#ffffff", border: "1px solid #d1d9e6", borderRadius: 6, padding: "6px 10px", zIndex: 2 }}>
          <div style={{ color: "#475569", fontSize: 13, fontFamily: "'IBM Plex Mono', monospace" }}>
            (Taxpayer)─[:TRANSACTS_WITH]→(Taxpayer)<br />
            <span style={{ color: "#be123c", fontWeight: 600 }}>Cycles = Fraud Ring Detection</span>
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
        <div style={{ border: "1px solid #d1d9e6", borderRadius: 8, padding: 16, background: "#ffffff" }}>
          <div style={{ color: "#475569", fontSize: 14, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, fontWeight: 600 }}>
            {hovNode ? `${hovNode.label} Properties` : "Hover a node to inspect"}
          </div>
          {hovNode ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {hovNode.props.map(p => (
                <span key={p} style={{
                  background: `${hovNode.color}10`, color: hovNode.color,
                  padding: "3px 8px", borderRadius: 4, fontSize: 14,
                  fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600,
                  border: `1px solid ${hovNode.color}22`
                }}>{p}</span>
              ))}
            </div>
          ) : (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {nodes.map(n => (
                <span key={n.id} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 13, color: "#475569" }}>
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: n.color, boxShadow: `0 0 4px ${n.color}44`, display: "inline-block" }} />
                  {n.label}
                </span>
              ))}
            </div>
          )}
        </div>

        <div style={{ border: "1px solid #d1d9e6", borderRadius: 8, padding: 16, flex: 1, background: "#ffffff" }}>
          <div style={{ color: "#475569", fontSize: 14, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, fontWeight: 600 }}>8 Relationship Types</div>
          {rels.map((r, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 0", borderBottom: "1px solid #f8f9fc", fontSize: 14 }}>
              <span style={{ color: "#1e40af", fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, fontWeight: 600 }}>{r.from}</span>
              <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ flex: 1, height: 1, background: `${r.color}22` }} />
                <span style={{ color: r.color, fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", whiteSpace: "nowrap", fontWeight: 600 }}>:{r.label}</span>
                <div style={{ flex: 1, height: 1, background: `${r.color}22` }} />
                <span style={{ color: "#94a3b8" }}>→</span>
              </div>
              <span style={{ color: "#1e40af", fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, fontWeight: 600 }}>{r.to}</span>
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {[
            ["6 Node Types", "Taxpayer, Invoice, IRN, Return, e-Way Bill, Payment", "#1e40af"],
            ["8 Relationships", "FILED, ISSUED, RECEIVED, HAS_IRN, COVERED_BY, REPORTED_IN, SETTLED_VIA, TRANSACTS_WITH", "#4f46e5"],
            ["Multi-hop Traversal", "Up to 6 hops for ITC validation", "#16a34a"],
            ["Graph DB", "Neo4j / NetworkX / ArangoDB", "#d97706"]
          ].map(([t, d, c]) => (
            <div key={t} style={{
              border: "1px solid #d1d9e6", borderRadius: 8, padding: "12px 14px", background: "#ffffff",
              borderLeft: `3px solid ${c}`
            }}>
              <div style={{ color: c, fontWeight: 700, fontSize: 14, marginBottom: 3, textShadow: 'none' }}>{t}</div>
              <div style={{ color: "#475569", fontSize: 14, lineHeight: 1.4 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
