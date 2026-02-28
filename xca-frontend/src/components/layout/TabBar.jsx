export default function TabBar({ tab, setTab, ingested }) {
    const tabs = [
        { id: "ingest", label: "Ingest", icon: "↓" },
        { id: "graph", label: "Graph", icon: "◉", locked: !ingested },
        { id: "mismatches", label: "Mismatches", icon: "⚠", locked: !ingested },
        { id: "vendor", label: "Vendors", icon: "◇", locked: !ingested },
        { id: "audit", label: "Audit", icon: "▤", locked: !ingested },
        { id: "schema", label: "Schema", icon: "⬡" },
    ];

    return (
        <div style={{ display: "flex", background: "#f8f9fc", padding: "0 24px", borderBottom: "1px solid #d1d9e6", gap: 0 }}>
            {tabs.map(t => {
                const active = tab === t.id;
                return (
                    <button key={t.id} onClick={() => !t.locked && setTab(t.id)} style={{
                        padding: "8px 14px",
                        background: "transparent",
                        cursor: t.locked ? "not-allowed" : "pointer",
                        border: "none",
                        borderBottom: `2px solid ${active ? "#1e40af" : "transparent"}`,
                        color: t.locked ? "#94a3b8" : active ? "#0f172a" : "#475569",
                        fontSize: 14, fontWeight: active ? 600 : 400,
                        fontFamily: "inherit",
                        transition: "color 0.15s, border-color 0.15s",
                        display: "flex", alignItems: "center", gap: 5
                    }}
                        onMouseEnter={e => { if (!t.locked && !active) e.target.style.color = "#475569"; }}
                        onMouseLeave={e => { if (!t.locked && !active) e.target.style.color = "#475569"; }}>
                        <span style={{ fontSize: 14, opacity: active ? 1 : 0.5 }}>{t.icon}</span>
                        {t.label}
                    </button>
                );
            })}
        </div>
    );
}
