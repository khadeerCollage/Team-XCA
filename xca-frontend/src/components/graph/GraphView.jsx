import { useState, useRef } from "react";
import { fmtINR } from "../../utils/formatters";
import { useGraphSimulation } from "../../hooks/useGraphSimulation";
import { nodeCol } from "../../constants/riskLevels";

export default function GraphView({ nodes: NODES, edges: EDGES, mismatches: MISMATCHES, rings, onSelectMismatch, setTab }) {
  const cvRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [selected, setSelected] = useState(null);

  const { onDown, onUp, onMoveDrag, nsRef, screenToGraph } = useGraphSimulation(NODES, EDGES, cvRef, setSelected);

  const getNode = (sx, sy) => {
    const { x, y } = screenToGraph(sx, sy);
    return nsRef.current.find(n => {
      const r = 13 + Math.sqrt(n.tx) * 0.9;
      return Math.hypot(n.x - x, n.y - y) < r + 6;
    });
  };

  const onMove = e => {
    const rect = cvRef.current.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (onMoveDrag(sx, sy)) return;
    const n = getNode(sx, sy);
    setTooltip(n ? { x: e.clientX, y: e.clientY, n } : null);
    cvRef.current.style.cursor = n ? "pointer" : "default";
  };
  const handleDown = e => { const rect = cvRef.current.getBoundingClientRect(); onDown(getNode(e.clientX - rect.left, e.clientY - rect.top)); };
  const handleUp = e => { const rect = cvRef.current.getBoundingClientRect(); onUp(getNode(e.clientX - rect.left, e.clientY - rect.top)); };

  const totalAtRisk = NODES.reduce((s, n) => s + n.itc, 0);
  const ringNames = (rings && rings.length > 0) ? rings[0].map(nid => NODES.find(n => n.id === nid)).filter(Boolean) : [];

  return (
    <div style={{ display: "flex", height: "100%", gap: 0, fontFamily: "'Source Sans 3', sans-serif" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {/* Stats bar */}
        <div style={{ display: "flex", gap: 6, padding: "6px 0 10px", alignItems: "center", flexWrap: "wrap" }}>
          <GlowStat label="ITC at Risk" value={fmtINR(totalAtRisk)} color="#dc2626" />
          <GlowStat label="High Risk" value={NODES.filter(n => n.risk === "high").length} color="#dc2626" />
          <GlowStat label="Fraud Rings" value={rings?.length || 0} color="#be123c" />
          <GlowStat label="Clean" value={EDGES.filter(e => e.ok).length} color="#16a34a" />
          <GlowStat label="Mismatches" value={MISMATCHES.length} color="#d97706" />

          <div style={{ marginLeft: "auto", display: "flex", gap: 12, fontSize: 14, color: "#475569", alignItems: "center" }}>
            {[["#dc2626", "High"], ["#d97706", "Medium"], ["#16a34a", "Low"]].map(([c, l]) => (
              <span key={l} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}66`, display: "inline-block" }} />{l}
              </span>
            ))}
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 14, borderTop: "2px dashed #dc2626", display: "inline-block" }} />Mismatch
            </span>
          </div>
        </div>

        {/* Graph canvas */}
        <div style={{ flex: 1, position: "relative", borderRadius: 8, overflow: "hidden", border: "1px solid #d1d9e6" }}>
          <div className="dot-grid" style={{ position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.4 }} />
          <canvas ref={cvRef} style={{ width: "100%", height: "100%", display: "block", background: "#f8f9fc" }}
            onMouseMove={onMove} onMouseDown={handleDown} onMouseUp={handleUp} onMouseLeave={() => setTooltip(null)} />
        </div>
        <div style={{ fontSize: 14, color: "#94a3b8", padding: "4px 0", textAlign: "right" }}>
          {NODES.length} nodes · {EDGES.length} edges · drag to rearrange · click to inspect
        </div>
      </div>

      {/* Sidebar */}
      <div style={{ width: 260, borderLeft: "1px solid #d1d9e6", paddingLeft: 16, marginLeft: 16, overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {selected ? (
          <div style={{ animation: "fadeIn 0.2s" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0 12px", borderBottom: "1px solid #d1d9e6" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: nodeCol(selected.risk), boxShadow: `0 0 8px ${nodeCol(selected.risk)}66` }} />
              <span style={{ color: "#0f172a", fontWeight: 700, fontSize: 14 }}>{selected.label}</span>
            </div>
            <div style={{ padding: "6px 0" }}>
              {[
                ["GSTIN", selected.gstin, true],
                ["State", selected.state],
                ["Transactions", selected.tx],
                ["ITC at Risk", fmtINR(selected.itc)],
                ["Risk Score", selected.score.toFixed(2), true],
                ["Risk Level", selected.risk.toUpperCase()]
              ].map(([k, v, mono]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: "1px solid #f1f4f9", fontSize: 14 }}>
                  <span style={{ color: "#475569" }}>{k}</span>
                  <span style={{
                    color: k === "Risk Level" ? nodeCol(selected.risk) : k === "Risk Score" ? (selected.score > 0.6 ? "#dc2626" : selected.score > 0.3 ? "#d97706" : "#16a34a") : "#0f172a",
                    fontFamily: mono ? "'IBM Plex Mono', monospace" : "inherit",
                    fontSize: k === "GSTIN" ? 9 : 12, fontWeight: 600,
                    textShadow: k === "Risk Level" ? `0 0 10px ${nodeCol(selected.risk)}44` : "none"
                  }}>{v}</span>
                </div>
              ))}
            </div>
            {MISMATCHES.filter(m => m.gstin === selected.gstin).length > 0 && (
              <button onClick={() => setTab("mismatches")} style={{
                width: "100%", padding: "7px", background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.2)",
                borderRadius: 6, color: "#dc2626", fontSize: 13, cursor: "pointer", fontWeight: 600,
                fontFamily: "inherit", marginTop: 8, transition: "all 0.15s"
              }}>
                View {MISMATCHES.filter(m => m.gstin === selected.gstin).length} Mismatches →
              </button>
            )}
          </div>
        ) : (
          <div style={{ color: "#94a3b8", fontSize: 14, textAlign: "center", marginTop: 60, lineHeight: 1.8 }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#d1d9e6" strokeWidth="1.5" style={{ display: "block", margin: "0 auto 12px" }}>
              <circle cx="12" cy="12" r="10" /><path d="M8 12h8M12 8v8" />
            </svg>
            Select a node<br />to view details
          </div>
        )}

        {/* Fraud Ring */}
        <div style={{ marginTop: "auto", borderTop: "1px solid #d1d9e6", paddingTop: 12 }}>
          <div style={{ fontSize: 14, color: "#475569", fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase", marginBottom: 8 }}>Fraud Ring</div>
          {ringNames.length > 0 ? (
            <>
              {ringNames.map((n, i, arr) => (
                <div key={n.id} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, fontSize: 13 }}>
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#be123c", boxShadow: "0 0 6px rgba(190,18,60,0.5)", display: "inline-block" }} />
                  <span style={{ color: "#be123c", fontWeight: 600 }}>{n.label}</span>
                  {i < arr.length - 1 && <span style={{ color: "#d1d9e6", marginLeft: "auto" }}>→</span>}
                </div>
              ))}
              <div style={{ fontSize: 14, color: "#be123c", background: "rgba(190,18,60,0.08)", padding: "5px 8px", borderRadius: 4, marginTop: 4, fontWeight: 500, border: "1px solid rgba(190,18,60,0.15)" }}>
                ⚠ Fake ITC chain detected
              </div>
            </>
          ) : <div style={{ color: "#d1d9e6", fontSize: 13 }}>No rings detected</div>}
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: "fixed", left: tooltip.x + 14, top: tooltip.y - 10,
          background: "#ffffff", border: "1px solid #d1d9e6", borderRadius: 8,
          padding: "10px 14px", fontSize: 13, color: "#0f172a", pointerEvents: "none", zIndex: 999,
          boxShadow: "0 8px 24px rgba(0,0,0,0.4)"
        }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>{tooltip.n.label}</div>
          <div style={{ color: "#475569", fontSize: 13, fontFamily: "'IBM Plex Mono', monospace" }}>{tooltip.n.gstin}</div>
          <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
            <span style={{ color: nodeCol(tooltip.n.risk), fontWeight: 600, textShadow: 'none' }}>{tooltip.n.risk.toUpperCase()}</span>
            <span style={{ color: "#475569" }}>Score: {tooltip.n.score.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function GlowStat({ label, value, color }) {
  return (
    <div style={{
      display: "flex", alignItems: "baseline", gap: 5, padding: "4px 10px",
      background: "#ffffff", border: "1px solid #d1d9e6", borderRadius: 6, fontSize: 12
    }}>
      <span style={{
        color, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace",
        textShadow: 'none'
      }}>{value}</span>
      <span style={{ color: "#475569", fontSize: 14 }}>{label}</span>
    </div>
  );
}
