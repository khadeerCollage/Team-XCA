export default function Header({ ingested, nodesCount, mismatchesCount, summary, user, logout }) {
    return (
        <div style={{
            borderBottom: "1px solid #d1d9e6",
            padding: "14px 28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "rgba(255,255,255,0.85)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            position: "sticky",
            top: 0,
            zIndex: 50
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                {/* Logo icon */}
                <div style={{
                    width: 36, height: 36, display: "flex", alignItems: "center", justifyContent: "center"
                }}>
                    <img src="/emblem.svg" alt="Emblem" style={{ height: "100%", width: "auto", objectFit: "contain" }} />
                </div>
                <div>
                    <div className="serif-header" style={{
                        fontWeight: 700,
                        fontSize: 18,
                        letterSpacing: "-0.3px",
                        color: "#0f172a",
                    }}>
                        GST Reconciliation Engine
                    </div>
                    <div style={{
                        color: "#475569",
                        fontSize: 11,
                        letterSpacing: "0.5px",
                        fontWeight: 500,
                        marginTop: 1
                    }}>
                        Knowledge Graph · ITC Validation · HackWithAI · PS #76
                    </div>
                </div>
            </div>
            <div style={{ display: "flex", gap: 12, fontSize: 12, alignItems: "center", fontWeight: 500, fontFamily: "'Source Sans 3', sans-serif" }}>
                <StatusPill label="FY 2025–26" color="#475569" bg="#f1f4f9" />
                <StatusPill
                    label={ingested ? "GRAPH READY" : "AWAITING DATA"}
                    color={ingested ? "#10b981" : "#f59e0b"}
                    bg={ingested ? "rgba(16,185,129,0.1)" : "rgba(245,158,11,0.1)"}
                    dot
                />
                <StatusPill label={`${nodesCount} Taxpayers`} color="#3b82f6" bg="rgba(59,130,246,0.08)" />
                <StatusPill label={`${mismatchesCount} Mismatches`} color="#ef4444" bg="rgba(239,68,68,0.08)" />
                {summary && (
                    <>
                        <StatusPill label={`${summary.critical_count || 0} Critical`} color="#dc2626" bg="rgba(220,38,38,0.08)" />
                        <StatusPill label={`Rs.${((summary.total_itc_at_risk || 0) / 100000).toFixed(1)}L at Risk`} color="#f59e0b" bg="rgba(245,158,11,0.08)" />
                    </>
                )}

                {/* Divider */}
                <div style={{ width: 1, height: 28, background: "#d1d9e6", margin: "0 4px" }} />

                {/* User info */}
                {user && (
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>

                        <div>
                            <div style={{ fontSize: 12, fontWeight: 600, color: "#0f172a" }}>{user.name}</div>
                            <div style={{ fontSize: 10, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px" }}>{user.role}</div>
                        </div>
                        <button onClick={logout} style={{
                            background: "#ffffff", border: "1px solid #d1d9e6", borderRadius: 8,
                            padding: "6px 10px", cursor: "pointer", color: "#94a3b8",
                            display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600,
                            fontFamily: "inherit", transition: "all 0.2s", marginLeft: 4
                        }}
                            onMouseEnter={e => { e.currentTarget.style.background = "#fef2f2"; e.currentTarget.style.color = "#ef4444"; e.currentTarget.style.borderColor = "#fca5a5"; }}
                            onMouseLeave={e => { e.currentTarget.style.background = "#ffffff"; e.currentTarget.style.color = "#94a3b8"; e.currentTarget.style.borderColor = "#d1d9e6"; }}
                        >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
                            Logout
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

function StatusPill({ label, color, bg, dot }) {
    return (
        <span style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            padding: "5px 12px",
            borderRadius: 20,
            background: bg,
            color: color,
            fontSize: 12,
            fontWeight: 600,
            transition: "all 0.2s"
        }}>
            {dot && (
                <span style={{
                    width: 7,
                    height: 7,
                    borderRadius: "50%",
                    background: color,
                    display: "inline-block",
                    boxShadow: `0 0 6px ${color}88`
                }}></span>
            )}
            {label}
        </span>
    );
}
