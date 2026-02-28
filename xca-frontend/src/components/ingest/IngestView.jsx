import { useState } from "react";
import { C } from "../../constants/colors";
import { useIngestPipeline } from "../../hooks/useIngestPipeline";

const API = import.meta.env.VITE_API_URL || '/api';

export default function IngestView({ onComplete }) {
  const { phase, progress, log, setLog, run, graphData } = useIngestPipeline();

  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedCounts, setGeneratedCounts] = useState(null);
  const [hoveredSrc, setHoveredSrc] = useState(null);

  const handleGenerateMockData = async () => {
    setIsGenerating(true);
    setLog(["Connecting to backend pipeline service..."]);

    try {
      const res = await fetch(`${API}/pipeline/generate`, { method: 'POST' });
      const data = await res.json();

      // Animate backend logs one by one into the SYSTEM LOG panel
      const backendLogs = data.logs || [];
      let idx = 0;
      const logTimer = setInterval(() => {
        if (idx < backendLogs.length) {
          setLog(prev => [...prev, backendLogs[idx]]);
          idx++;
        } else {
          clearInterval(logTimer);
          if (data.counts) {
            setGeneratedCounts(data.counts);
          }
          setIsGenerating(false);
        }
      }, 150);
    } catch (err) {
      setLog(prev => [...prev, `ERROR: Failed to connect to backend — ${err.message}`, "Make sure FastAPI is running: cd xca-backend/backend && python main.py"]);
      setIsGenerating(false);
    }
  };

  const sc = generatedCounts || {};

  const sources = [
    { key: "gstr1", label: "GSTR-1", desc: "Outward Supplies", rows: sc.gstr1, color: "#10b981", bg: "linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.02))" },
    { key: "gstr2b", label: "GSTR-2B", desc: "Auto-drafted ITC", rows: sc.gstr2b, color: "#3b82f6", bg: "linear-gradient(135deg, rgba(59,130,246,0.1), rgba(59,130,246,0.02))" },
    { key: "gstr3b", label: "GSTR-3B", desc: "Summary Returns", rows: sc.gstr3b, color: "#8b5cf6", bg: "linear-gradient(135deg, rgba(139,92,246,0.1), rgba(139,92,246,0.02))" },
    { key: "einvoice", label: "e-Invoice", desc: "IRN Registry", rows: sc.einvoice, color: "#f59e0b", bg: "linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.02))" },
    { key: "ewaybill", label: "e-Way Bills", desc: "Goods Movement", rows: sc.ewaybill, color: "#ef4444", bg: "linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.02))" },
    { key: "purchreg", label: "Purchase Reg.", desc: "Internal Records", rows: sc.purchreg, color: "#475569", bg: "linear-gradient(135deg, rgba(100,116,139,0.1), rgba(100,116,139,0.02))" },
  ];

  const summary = graphData?.summary;

  return (
    <div style={{ height: "100%", display: "flex", gap: 32, fontFamily: "'Source Sans 3', sans-serif" }}>
      {/* Left */}
      <div style={{ flex: 1.2, display: "flex", flexDirection: "column" }}>
        <div style={{ marginBottom: 24, padding: "24px", background: "#ffffff", borderRadius: 20, boxShadow: "0 4px 12px rgba(0,0,0,0.05)", border: "1px solid #d1d9e6" }}>
          <div style={{ color: "#0f172a", fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px", marginBottom: 8 }}>Knowledge Graph Data Ingestion</div>
          <div style={{ color: "#475569", fontSize: 13, lineHeight: 1.5 }}>
            Automated pipeline extracting data from GST endpoints, E-Way Bill registries, and internal ERP logs.
            Data is mapped via Neo4j graph algorithms to detect circular trading rings and ITC claim anomalies before final GSTR-3B filing.istries.
          </div>

          <div style={{ marginTop: 20 }}>
            <button
              onClick={handleGenerateMockData}
              disabled={isGenerating || phase === "loading"}
              style={{
                width: "auto",
                padding: "12px 24px",
                background: isGenerating ? "#d1d9e6" : "linear-gradient(135deg, #0f172a, #334155)",
                color: isGenerating ? "#94a3b8" : "#fff",
                border: "none",
                borderRadius: 12,
                fontSize: 14,
                fontWeight: 600,
                cursor: (isGenerating || phase === "loading") ? "not-allowed" : "pointer",
                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                boxShadow: isGenerating ? "none" : "0 4px 12px rgba(15,23,42,0.15), 0 0 0 1px rgba(15,23,42,0.1)",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                transform: isGenerating ? "scale(0.98)" : "scale(1)"
              }}
            >
              {isGenerating ? (
                <>
                  <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="32" strokeLinecap="round" opacity="0.3"></circle>
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"></path>
                  </svg>
                  Simulating Integration...
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                  Fetch Data Sources
                </>
              )}
            </button>
          </div>
        </div>

        {generatedCounts && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24, animation: "fadeIn 0.5s ease-out" }}>
            {sources.map(src => (
              <div
                key={src.key}
                onMouseEnter={() => setHoveredSrc(src.key)}
                onMouseLeave={() => setHoveredSrc(null)}
                style={{
                  background: "#fff",
                  border: `1px solid ${hoveredSrc === src.key ? src.color : "rgba(226, 232, 240, 0.8)"}`,
                  borderRadius: 16,
                  padding: 20,
                  boxShadow: hoveredSrc === src.key ? `0 12px 24px ${src.color}20` : "0 4px 6px -1px rgba(0,0,0,0.02)",
                  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                  transform: hoveredSrc === src.key ? "translateY(-4px)" : "translateY(0)"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    <div style={{
                      width: 40, height: 40,
                      borderRadius: 10,
                      background: src.bg,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: src.color,
                      fontWeight: 800,
                      fontSize: 16
                    }}>
                      {src.label.charAt(0)}
                    </div>
                    <div>
                      <div style={{ color: "#0f172a", fontWeight: 700, fontSize: 15 }}>{src.label}</div>
                      <div style={{ color: "#475569", fontSize: 12, marginTop: 2, fontWeight: 500 }}>{src.desc}</div>
                    </div>
                  </div>
                  <div style={{
                    color: phase === "done" || (progress[src.key] || 0) >= 100 ? "#10b981" : "#475569",
                    fontSize: 12,
                    fontWeight: 600,
                    background: phase === "done" || (progress[src.key] || 0) >= 100 ? "rgba(16,185,129,0.1)" : "#f1f4f9",
                    padding: "4px 8px",
                    borderRadius: 6
                  }}>
                    {phase === "done" || (progress[src.key] || 0) >= 100
                      ? `${src.rows} Ready`
                      : `${src.rows} rows`}
                  </div>
                </div>

                <div style={{ height: 6, background: "#f1f4f9", borderRadius: 8, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", borderRadius: 8,
                    background: `linear-gradient(90deg, ${src.color}, ${src.color}dd)`,
                    width: `${progress[src.key] || 0}%`,
                    transition: "width 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
                    boxShadow: `0 0 10px ${src.color}66`
                  }} />
                </div>

                <div style={{ color: "#94a3b8", fontSize: 11, marginTop: 8, fontWeight: 500, display: "flex", justifyContent: "space-between" }}>
                  <span>{phase === "idle" ? "Pending Ingestion" : progress[src.key] >= 100 ? "Processing Complete" : "Extracting Data..."}</span>
                  <span style={{ color: src.color, fontWeight: 700 }}>{Math.round(progress[src.key] || 0)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {generatedCounts && (
          <button onClick={phase === "done" ? () => onComplete(graphData) : () => run(sources)} disabled={phase === "loading" || isGenerating || !generatedCounts} style={{
            width: "100%",
            padding: "16px",
            borderRadius: 16,
            fontSize: 15,
            fontWeight: 700,
            cursor: (phase === "loading" || isGenerating || !generatedCounts) ? "not-allowed" : "pointer",
            background: phase === "done" ? "#16a34a" : phase === "loading" ? "#f1f4f9" : "#1e40af",
            color: phase === "loading" ? "#475569" : "#fff",
            border: phase === "loading" ? "1px solid #d1d9e6" : "none",
            boxShadow: phase === "loading" ? "none" : phase === "done" ? "0 4px 12px rgba(22,163,74,0.2)" : "0 4px 12px rgba(30,64,175,0.2)",
            transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            letterSpacing: "0.5px",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: 10,
            transform: phase === "loading" ? "scale(0.99)" : "scale(1)",
            marginTop: "auto"
          }}>
            {phase === "idle" ? (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                EXECUTE INGESTION PIPELINE
              </>
            ) : phase === "loading" ? (
              <>
                <svg className="animate-spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>
                BUILDING KNOWLEDGE GRAPH...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                ENTER GRAPH DASHBOARD
              </>
            )}
          </button>
        )}
      </div >

      {/* Right */}
      <div style={{
        flex: 1,
        background: "#ffffff",
        border: "1px solid #d1d9e6",
        borderRadius: 24,
        padding: 24,
        display: "flex",
        flexDirection: "column",
        boxShadow: "0 8px 30px rgba(15,23,42,0.04)",
        position: "relative",
        overflow: "hidden"
      }}>
        {/* Decorative elements */}
        <div style={{ position: "absolute", top: -100, right: -100, width: 250, height: 250, background: "radial-gradient(circle, #f1f4f9 0%, transparent 70%)", borderRadius: "50%", pointerEvents: "none" }}></div>
        <div style={{ position: "absolute", bottom: -50, left: -50, width: 200, height: 200, background: "radial-gradient(circle, rgba(30,64,175,0.03) 0%, transparent 70%)", borderRadius: "50%", pointerEvents: "none" }}></div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20, zIndex: 1 }}>
          <div style={{ display: "flex", gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#ef4444" }}></div>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#f59e0b" }}></div>
            <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#10b981" }}></div>
          </div>
          <div style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, letterSpacing: "1px", marginLeft: 10 }}>TERMINAL</div>
        </div>

        <div style={{ flex: 1, overflowY: "auto", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, zIndex: 1, paddingRight: 8, background: "#f8f9fc", borderRadius: 8, padding: 16, border: "1px solid #d1d9e6" }}>
          {log.length === 0 && <div style={{ color: "#94a3b8" }}>$ Waiting for command...</div>}
          {log.map((l, i) => {
            if (!l) return null;
            const isError = l.includes("CRITICAL") || l.includes("risk") || l.includes("ERROR");
            const isSuccess = l.includes("successfully") || l.includes("Complete") || l.includes("built");

            return (
              <div key={i} style={{
                color: isError ? "#dc2626" : isSuccess ? "#16a34a" : "#475569",
                marginBottom: 8,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
                display: "flex",
                gap: 12
              }}>
                <span style={{ color: "#94a3b8", userSelect: "none" }}>{String(i + 1).padStart(2, "0")}</span>
                <span style={{ flex: 1 }}>{l}</span>
              </div>
            );
          })}
          {(phase === "loading" || isGenerating) && (
            <div style={{ color: "#94a3b8", display: "flex", gap: 12, marginTop: 8 }}>
              <span style={{ color: "#94a3b8" }}>{String(log.length + 1).padStart(2, "0")}</span>
              <span className="animate-pulse" style={{ animation: "pulse 1s infinite cubic-bezier(0.4, 0, 0.6, 1)" }}>...</span>
            </div>
          )}
        </div>

        {phase === "done" && summary && (
          <div style={{
            marginTop: 20,
            padding: 20,
            background: "rgba(22, 163, 74, 0.05)",
            border: "1px solid rgba(22, 163, 74, 0.2)",
            borderRadius: 16,
            backdropFilter: "blur(4px)",
            zIndex: 1,
            animation: "slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
              <div style={{ color: "#16a34a", fontWeight: 700, fontSize: 15 }}>Pipeline Execution Complete</div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 16 }}>
              <div style={{ background: "#ffffff", border: "1px solid #d1d9e6", padding: "10px 14px", borderRadius: 10 }}>
                <div style={{ color: "#475569", fontSize: 11, marginBottom: 4 }}>Graph Size</div>
                <div style={{ color: "#0f172a", fontWeight: 700 }}>{summary.total_nodes} nodes <span style={{ color: "#94a3b8", margin: "0 4px" }}>•</span> {summary.total_edges} edges</div>
              </div>
              <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", padding: "10px 14px", borderRadius: 10 }}>
                <div style={{ color: "#dc2626", fontSize: 11, marginBottom: 4 }}>Risk Assessment</div>
                <div style={{ color: "#dc2626", fontWeight: 700 }}>Rs.{(summary.total_itc_at_risk / 100000).toFixed(1)}L at risk</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
