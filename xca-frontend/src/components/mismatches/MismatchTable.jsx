import { useState, useEffect } from "react";
import { C } from "../../constants/colors";
import { riskCol } from "../../constants/riskLevels";
import { fmtINR } from "../../utils/formatters";
import { fetchMismatches, fetchPipelineMismatches, fetchGraphOutputStatic } from "../../services/api";

/**
 * 3-level fallback to get ALL mismatches:
 *   1. GET /api/pipeline/mismatches   (reads graph_output.json via backend)
 *   2. GET /api/reconcile/mismatches  (live reconciliation engine)
 *   3. GET /graph_output.json         (static file from public/)
 */
async function loadAllMismatches() {
  // Level 1 — pipeline endpoint (simple, reads graph_output.json via backend)
  try {
    const data = await fetchPipelineMismatches();
    if (data.mismatches && data.mismatches.length > 0) return data.mismatches;
  } catch (_) { /* try next */ }

  // Level 2 — reconciliation engine (live Neo4j query)
  try {
    const data = await fetchMismatches({ limit: 500 });
    if (data && data.length > 0) return data;
  } catch (_) { /* try next */ }

  // Level 3 — static graph_output.json in public/
  try {
    const data = await fetchGraphOutputStatic();
    if (data.mismatches && data.mismatches.length > 0) return data.mismatches;
  } catch (_) { /* all failed */ }

  return null;
}

export default function MismatchTable({ mismatches: MISMATCHES_PROP, onSelect }) {
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [hoveredRow, setHoveredRow] = useState(null);
  const [liveMismatches, setLiveMismatches] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch ALL mismatches with 3-level fallback
  useEffect(() => {
    setLoading(true);
    loadAllMismatches()
      .then(data => { setLiveMismatches(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  // Use live API data if available, otherwise fall back to static props
  const MISMATCHES = liveMismatches || MISMATCHES_PROP || [];

  const filtered = MISMATCHES.filter(m =>
    (filter === "ALL" || m.risk === filter) &&
    ((m.vendor || '').toLowerCase().includes(search.toLowerCase()) || (m.inv || '').includes(search))
  );
  const total = MISMATCHES.reduce((s, m) => s + (m.delta || 0), 0);

  const statCards = [
    { l: "Total Mismatches", v: MISMATCHES.length, c: "#475569" },
    { l: "ITC at Risk", v: fmtINR(total), c: "#ef4444" },
    { l: "Critical", v: MISMATCHES.filter(m => m.risk === "CRITICAL").length, c: "#dc2626" },
    { l: "High", v: MISMATCHES.filter(m => m.risk === "HIGH").length, c: "#ef4444" },
    { l: "Medium", v: MISMATCHES.filter(m => m.risk === "MEDIUM").length, c: "#f59e0b" },
  ];

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
      {/* Loading indicator */}
      {loading && (
        <div style={{ padding: "24px", textAlign: "center", color: "#94a3b8", fontSize: 14 }}>
          Loading mismatches from database...
        </div>
      )}

      {/* Stats */}
      <div style={{ display: "flex", gap: 12 }}>
        {statCards.map(c => (
          <div key={c.l} style={{
            background: "#fff", border: "1px solid #d1d9e6", borderRadius: 14, padding: "12px 18px",
            display: "flex", gap: 12, alignItems: "center", boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
          }}>
            <span style={{ color: c.c, fontWeight: 800, fontFamily: "'IBM Plex Mono', monospace", fontSize: 18 }}>{c.v}</span>
            <span style={{ color: "#94a3b8", fontSize: 12, fontWeight: 500 }}>{c.l}</span>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div style={{ display: "flex", gap: 10 }}>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search vendor or invoice..."
          style={{
            flex: 1, background: "#fff", border: "1px solid #d1d9e6", borderRadius: 12, padding: "10px 16px",
            color: "#0f172a", fontSize: 14, outline: "none", fontFamily: "inherit", fontWeight: 500,
            transition: "border-color 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
          }}
          onFocus={e => e.target.style.borderColor = "#3b82f6"}
          onBlur={e => e.target.style.borderColor = "#d1d9e6"}
        />
        {["ALL", "CRITICAL", "HIGH", "MEDIUM"].map(f => (
          <button key={f} onClick={() => setFilter(f)} style={{
            padding: "10px 16px", borderRadius: 10, fontSize: 13, cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
            background: filter === f ? (f === "ALL" ? "#f1f4f9" : riskCol(f) + "10") : "#fff",
            border: `1px solid ${filter === f ? (f === "ALL" ? "#b8c5d9" : riskCol(f) + "44") : "#d1d9e6"}`,
            color: filter === f ? (f === "ALL" ? "#0f172a" : riskCol(f)) : "#94a3b8",
            transition: "all 0.2s"
          }}>{f}</button>
        ))}
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflowY: "auto", background: "#fff", borderRadius: 16, border: "1px solid #d1d9e6", boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #f1f4f9" }}>
              {["#", "Vendor", "Invoice", "Type", "GSTR-1", "GSTR-2B", "GSTR-3B", "Delta", "Risk", ""].map(h => (
                <th key={h} style={{
                  padding: "14px 14px", color: "#94a3b8", fontWeight: 600, textAlign: "left",
                  fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px", background: "#fafbfc",
                  position: "sticky", top: 0, zIndex: 1
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((m, i) => (
              <tr key={m.id}
                style={{
                  borderBottom: "1px solid #ffffff", cursor: "pointer", transition: "background 0.15s",
                  background: hoveredRow === m.id ? "#ffffff" : "transparent"
                }}
                onMouseEnter={() => setHoveredRow(m.id)}
                onMouseLeave={() => setHoveredRow(null)}>
                <td style={{ padding: "14px", color: "#b8c5d9", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>{i + 1}</td>
                <td style={{ padding: "14px" }}>
                  <div style={{ color: "#0f172a", fontWeight: 600, fontSize: 13 }}>{m.vendor}</div>
                  <div style={{ color: "#94a3b8", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", marginTop: 2 }}>{m.gstin}</div>
                </td>
                <td style={{ padding: "14px", color: "#475569", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>{m.inv}</td>
                <td style={{ padding: "14px" }}>
                  <span style={{ background: riskCol(m.risk) + "10", color: riskCol(m.risk), padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600 }}>{m.type}</span>
                </td>
                <td style={{ padding: "14px", color: "#475569", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>Rs.{(m.gstr1 / 1000).toFixed(0)}K</td>
                <td style={{ padding: "14px", color: "#475569", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>Rs.{(m.gstr2b / 1000).toFixed(0)}K</td>
                <td style={{ padding: "14px", color: "#475569", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>{m.gstr3b > 0 ? `Rs.${(m.gstr3b / 1000).toFixed(0)}K` : <span style={{ color: "#ef4444", fontWeight: 600 }}>NOT FILED</span>}</td>
                <td style={{ padding: "14px", color: m.delta > 0 ? "#ef4444" : "#b8c5d9", fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, fontSize: 13 }}>
                  {m.delta > 0 ? `-Rs.${(m.delta / 1000).toFixed(0)}K` : "--"}
                </td>
                <td style={{ padding: "14px" }}>
                  <span style={{
                    background: riskCol(m.risk) + "10", color: riskCol(m.risk),
                    padding: "4px 12px", borderRadius: 20, fontSize: 11, fontWeight: 700
                  }}>{m.risk}</span>
                </td>
                <td style={{ padding: "14px" }}>
                  <button onClick={() => onSelect(m)} style={{
                    background: "#ffffff", border: "1px solid #d1d9e6", borderRadius: 8,
                    color: "#475569", padding: "6px 14px", fontSize: 12, cursor: "pointer",
                    fontFamily: "inherit", fontWeight: 600, transition: "all 0.2s"
                  }}>Audit</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
