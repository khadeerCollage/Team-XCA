import { useState, useEffect } from "react";
import { C } from "../../constants/colors";
import { riskCol } from "../../constants/riskLevels";
import { fmtINR } from "../../utils/formatters";
import { runMLPredictions, fetchMLScores, trainMLModel, fetchFeatureImportances } from "../../services/api";

export default function VendorScorecard({ nodes: NODES_PROP, mismatches: MISMATCHES, onSelect }) {
    const [sort, setSort] = useState("score");
    const [hoveredCard, setHoveredCard] = useState(null);
    const [liveNodes, setLiveNodes] = useState(null);
    const [modelUsed, setModelUsed] = useState(null);
    const [importances, setImportances] = useState(null);
    const [mlLoading, setMlLoading] = useState(false);

    // On mount: train model → predict → fetch scores
    useEffect(() => {
        (async () => {
            setMlLoading(true);
            try {
                // 1. Train model on graph features
                await trainMLModel();
                // 2. Run ML predictions
                const result = await runMLPredictions();
                setModelUsed(result.model_used || "xgboost");
                setImportances(result.feature_importances || {});
                // 3. Fetch predicted scores
                const scores = await fetchMLScores(100);
                if (scores && scores.length > 0) setLiveNodes(scores);
            } catch {
                // Silently fall back to static props
                try {
                    const imp = await fetchFeatureImportances();
                    if (imp) setImportances(imp);
                } catch {}
            }
            setMlLoading(false);
        })();
    }, []);

    // Use live ML data if available, otherwise fall back to static props
    const NODES = liveNodes || NODES_PROP;
    const sorted = [...NODES].sort((a, b) => b[sort] - a[sort]);

    // Score breakdown from ML component scores (NOT hardcoded weights)
    const scoreBreakdown = (n) => {
        // If ML component scores available, use them (0-100 → 0-1 normalised)
        if (n.filing_score !== undefined) {
            return [
                { label: "Filing", value: n.filing_score, weight: 0.25, score: (1 - n.filing_score / 100) * 0.25, color: "#f59e0b" },
                { label: "Mismatch", value: n.mismatch_score, weight: 0.25, score: (1 - n.mismatch_score / 100) * 0.25, color: "#ef4444" },
                { label: "Network", value: n.network_score, weight: 0.25, score: (1 - n.network_score / 100) * 0.25, color: "#dc2626" },
                { label: "Physical", value: n.physical_score, weight: 0.25, score: (1 - n.physical_score / 100) * 0.25, color: "#8b5cf6" },
            ];
        }
        // Fallback to old format
        return [
            { label: "Mismatch Rate", value: n.mismatchRate, weight: 0.35, score: n.mismatchRate * 0.35, color: "#ef4444" },
            { label: "Filing Delay", value: n.delay, weight: 0.25, score: Math.min(n.delay / 30, 1) * 0.25, color: "#f59e0b" },
            { label: "ITC Overclaim", value: n.overclaim, weight: 0.20, score: n.overclaim * 0.20, color: "#8b5cf6" },
            { label: "Circular Flag", value: n.circular, weight: 0.15, score: n.circular * 0.15, color: "#dc2626" },
            { label: "New Vendor", value: 0.05, weight: 0.05, score: 0.05 * 0.05, color: "#3b82f6" },
        ];
    };

    // Top 4 feature importances from the trained model
    const topImportances = importances
        ? Object.entries(importances)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5)
        : null;

    return (
        <div style={{ height: "100%", display: "flex", flexDirection: "column", gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
            {/* ML Model Card */}
            <div style={{
                background: "#fff", border: "1px solid #d1d9e6", borderRadius: 16, padding: "20px 24px",
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
            }}>
                {/* <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                    <div style={{ color: "#94a3b8", fontSize: 11, textTransform: "uppercase", letterSpacing: "1px", fontWeight: 600 }}>
                        {modelUsed === "xgboost" ? "XGBoost ML Risk Prediction" : "Predictive Risk Score — Component Analysis"}
                    </div>
                    {modelUsed && (
                        <span style={{
                            background: "#10b98110", color: "#10b981", fontSize: 10, padding: "3px 10px",
                            borderRadius: 12, fontWeight: 700, border: "1px solid #10b98122"
                        }}>
                            {modelUsed === "xgboost" ? "ML Model Active" : "Rule-Based"}
                        </span>
                    )}
                    {mlLoading && (
                        <span style={{ color: "#f59e0b", fontSize: 10, fontWeight: 600 }}>Training...</span>
                    )}
                </div> */}
                {/* <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, color: "#0f172a", lineHeight: 2 }}>
                    {modelUsed === "xgboost" ? (
                        <>
                            <span style={{ color: "#f59e0b", fontWeight: 700 }}>score</span> = XGBoost.predict(
                            <span style={{ color: "#f59e0b" }}> filing_analysis</span>,
                            <span style={{ color: "#ef4444" }}> mismatch_detection</span>,
                            <span style={{ color: "#dc2626" }}> network_topology</span>,
                            <span style={{ color: "#8b5cf6" }}> physical_verification</span>,
                            <span style={{ color: "#3b82f6" }}> 14_graph_features </span>)
                        </>
                    ) : (
                        <>
                            <span style={{ color: "#f59e0b", fontWeight: 700 }}>score</span> = f(
                            <span style={{ color: "#f59e0b" }}> filing_score</span>,
                            <span style={{ color: "#ef4444" }}> mismatch_score</span>,
                            <span style={{ color: "#dc2626" }}> network_score</span>,
                            <span style={{ color: "#8b5cf6" }}> physical_score </span>)
                        </>
                    )}
                </div> */}
                <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                    {[["< 0.3", "Low Risk", "#10b981"], ["0.3-0.6", "Medium Risk", "#f59e0b"], ["> 0.6", "High Risk", "#ef4444"]].map(([r, l, c]) => (
                        <div key={l} style={{
                            background: c + "08", border: `1px solid ${c}22`, borderRadius: 8,
                            padding: "6px 14px", fontSize: 12, display: "flex", alignItems: "center", gap: 8
                        }}>
                            <span style={{ color: c, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700 }}>{r}</span>
                            <span style={{ color: "#94a3b8", fontWeight: 500 }}>{l}</span>
                        </div>
                    ))}
                </div>
            </div> 

            {/* Sort Buttons */}
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ color: "#94a3b8", fontSize: 13, fontWeight: 500 }}>Sort by:</span>
                {[["score", "Risk Score"], ["mismatchRate", "Mismatch Rate"], ["itc", "ITC at Risk"], ["delay", "Filing Delay"]].map(([k, l]) => (
                    <button key={k} onClick={() => setSort(k)} style={{
                        padding: "8px 14px", borderRadius: 8, fontSize: 12, cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
                        background: sort === k ? "#f59e0b10" : "#fff",
                        border: `1px solid ${sort === k ? "#f59e0b44" : "#d1d9e6"}`,
                        color: sort === k ? "#f59e0b" : "#94a3b8",
                        transition: "all 0.2s"
                    }}>{l}</button>
                ))}
            </div>

            {/* Vendor Cards */}
            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 10 }}>
                {sorted.map((n, rank) => {
                    const sc = n.score;
                    const col = sc > 0.6 ? "#ef4444" : sc > 0.3 ? "#f59e0b" : "#10b981";
                    const breakdown = scoreBreakdown(n);
                    const isHovered = hoveredCard === n.id;

                    return (
                        <div key={n.id}
                            onMouseEnter={() => setHoveredCard(n.id)}
                            onMouseLeave={() => setHoveredCard(null)}
                            style={{
                                background: "#fff", border: `1px solid ${isHovered ? col + "44" : "#d1d9e6"}`,
                                borderRadius: 16, padding: "18px 20px",
                                boxShadow: isHovered ? `0 8px 24px ${col}12` : "0 1px 3px rgba(0,0,0,0.04)",
                                transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
                                transform: isHovered ? "translateY(-2px)" : "translateY(0)"
                            }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
                                <div style={{
                                    color: "#b8c5d9", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13,
                                    width: 28, height: 28, borderRadius: 8, background: "#ffffff",
                                    display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700
                                }}>#{rank + 1}</div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ color: "#0f172a", fontWeight: 700, fontSize: 15 }}>{n.label}</div>
                                    <div style={{ color: "#94a3b8", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", marginTop: 2 }}>{n.gstin} · {n.state}</div>
                                </div>
                                <div style={{ textAlign: "right" }}>
                                    <div style={{ color: col, fontFamily: "'IBM Plex Mono', monospace", fontSize: 26, fontWeight: 800 }}>{sc.toFixed(2)}</div>
                                    <span style={{
                                        background: col + "10", color: col, fontSize: 11, padding: "3px 10px",
                                        borderRadius: 12, fontWeight: 700
                                    }}>
                                        {sc > 0.6 ? "HIGH RISK" : sc > 0.3 ? "MEDIUM" : "LOW RISK"}
                                    </span>
                                </div>
                            </div>

                            {/* Score Bar */}
                            <div style={{ height: 6, background: "#f1f4f9", borderRadius: 8, overflow: "hidden", marginBottom: 14 }}>
                                <div style={{
                                    height: "100%", width: `${sc * 100}%`, background: `linear-gradient(90deg, ${col}, ${col}cc)`,
                                    borderRadius: 8, boxShadow: `0 0 10px ${col}44`, transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)"
                                }} />
                            </div>

                            {/* Breakdown */}
                            <div style={{ display: "flex", gap: 10 }}>
                                {breakdown.map(b => (
                                    <div key={b.label} style={{ flex: 1, textAlign: "center" }}>
                                        <div style={{ height: 4, background: "#f1f4f9", borderRadius: 4, overflow: "hidden", marginBottom: 4 }}>
                                            <div style={{
                                                height: "100%", width: `${b.score / b.weight * 100}%`,
                                                background: b.color, borderRadius: 4, transition: "width 0.5s"
                                            }} />
                                        </div>
                                        <div style={{ color: "#94a3b8", fontSize: 10, fontWeight: 500 }}>{b.label.split(" ")[0]}</div>
                                        <div style={{ color: b.color, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700 }}>{(b.score).toFixed(2)}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Bottom Stats */}
                            <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                                {[["ITC", fmtINR(n.itc)], ["Txns", n.tx], ["Delay", `${n.delay}d`], ["Overclaim", `${(n.overclaim * 100).toFixed(0)}%`]].map(([k, v]) => (
                                    <div key={k} style={{
                                        flex: 1, background: "#ffffff", borderRadius: 10, padding: "8px 10px", textAlign: "center"
                                    }}>
                                        <div style={{ color: "#94a3b8", fontSize: 10, fontWeight: 500 }}>{k}</div>
                                        <div style={{ color: "#0f172a", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700 }}>{v}</div>
                                    </div>
                                ))}
                                {MISMATCHES.some(m => m.gstin === n.gstin) && (
                                    <button onClick={() => onSelect(MISMATCHES.find(m => m.gstin === n.gstin))} style={{
                                        padding: "8px 14px", background: "rgba(239,68,68,0.06)",
                                        border: "1px solid rgba(239,68,68,0.2)", borderRadius: 10,
                                        color: "#ef4444", fontSize: 12, cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
                                        transition: "all 0.2s"
                                    }}>Audit</button>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
