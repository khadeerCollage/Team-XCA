import { useState, useEffect } from "react";
import { useAuth } from "./contexts/AuthContext";
import LoginScreen from "./components/auth/LoginScreen";

import Header from "./components/layout/Header";
import TabBar from "./components/layout/TabBar";
import IngestView from "./components/ingest/IngestView";
import GraphView from "./components/graph/GraphView";
import MismatchTable from "./components/mismatches/MismatchTable";
import VendorScorecard from "./components/vendor/VendorScorecard";
import AuditTrail from "./components/audit/AuditTrail";
import SchemaView from "./components/schema/SchemaView";
import { runReconciliation, fetchReconcileSummary, fetchGraphOutputStatic } from "./services/api";

export default function App() {
    const { isAuthenticated, user, logout } = useAuth();

    // Show login screen if not authenticated
    if (!isAuthenticated) {
        return <LoginScreen />;
    }

    return <Dashboard user={user} logout={logout} />;
}

function Dashboard({ user, logout }) {
    const [tab, setTab] = useState("ingest");
    const [mismatch, setMismatch] = useState(null);
    const [ingested, setIngested] = useState(false);
    const [graphData, setGraphData] = useState(null);
    const [summary, setSummary] = useState(null);

    const handleSelect = (m) => { setMismatch(m); setTab("audit"); };
    const handleIngestDone = (data) => {
        setGraphData(data);
        setIngested(true);
        setTab("graph");
    };

    // ── Auto-load graph data on startup ───────────────────────
    // Try backend first (GET /pipeline/graph-data), fall back to static file
    useEffect(() => {
        async function autoLoad() {
            try {
                // Try pipeline API first
                const res = await fetch('/api/pipeline/graph-data');
                if (res.ok) {
                    const data = await res.json();
                    if (data.nodes && data.nodes.length > 0) {
                        setGraphData(data);
                        setIngested(true);
                        return;
                    }
                }
            } catch (_) { /* backend may be down */ }

            // Fall back to static graph_output.json in public/
            try {
                const data = await fetchGraphOutputStatic();
                if (data.nodes && data.nodes.length > 0) {
                    setGraphData(data);
                    setIngested(true);
                }
            } catch (_) { /* no data available yet — user needs to ingest */ }
        }
        autoLoad();
    }, []);

    // After ingestion completes → trigger reconciliation engine + fetch summary
    useEffect(() => {
        if (!ingested) return;
        // Run reconciliation (fast, rule-based — no AI)
        runReconciliation(null, false)
            .then(() => fetchReconcileSummary())
            .then(s => setSummary(s))
            .catch(err => console.warn("Reconciliation summary fetch failed:", err));
    }, [ingested]);

    const nodes = graphData?.nodes || [];
    const edges = graphData?.edges || [];
    const mismatches = graphData?.mismatches || [];

    return (
        <div style={{ background: "#f8f9fc", minHeight: "100vh", fontFamily: "'Source Sans 3', sans-serif", color: "#0f172a", display: "flex", flexDirection: "column" }}>
            <Header ingested={ingested} nodesCount={nodes.length} mismatchesCount={summary?.total_mismatches ?? mismatches.length} summary={summary} user={user} logout={logout} />
            <TabBar tab={tab} setTab={setTab} ingested={ingested} />
            <div style={{ flex: 1, padding: "18px 24px", overflow: "hidden", height: "calc(100vh - 116px)" }}>
                {tab === "ingest" && <IngestView onComplete={handleIngestDone} />}
                {tab === "graph" && <GraphView nodes={nodes} edges={edges} mismatches={mismatches} rings={graphData?.rings || []} onSelectMismatch={handleSelect} setTab={setTab} />}
                {tab === "mismatches" && <MismatchTable mismatches={mismatches} onSelect={handleSelect} />}
                {tab === "vendor" && <VendorScorecard nodes={nodes} mismatches={mismatches} onSelect={handleSelect} />}
                {tab === "audit" && <AuditTrail mismatch={mismatch} nodes={nodes} />}
                {tab === "schema" && <SchemaView />}
            </div>
        </div>
    );
}
