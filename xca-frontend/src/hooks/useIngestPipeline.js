import { useState, useEffect } from "react";
import { loadGraph } from "../services/graphLoader";

export function useIngestPipeline() {
    const [phase, setPhase] = useState("idle"); // idle | loading | done
    const [progress, setProgress] = useState({});
    const [log, setLog] = useState([]);
    const [graphData, setGraphData] = useState(null);

    // Pre-fetch static graph_output.json as fallback (non-blocking)
    useEffect(() => {
        loadGraph().then(data => setGraphData(data)).catch(() => { });
    }, []);

    const run = (sources) => {
        setPhase("loading");
        setProgress({});
        setLog([]);

        // Animate progress bars immediately (visual feedback while backend works)
        sources.forEach((src, i) => {
            const delay = i * 300;
            const steps = 20;
            for (let s = 1; s <= steps; s++) {
                setTimeout(() => {
                    setProgress(prev => ({ ...prev, [src.key]: (s / steps) * 100 }));
                }, delay + s * 120);
            }
        });

        // Call the real backend pipeline endpoint
        fetch('/api/pipeline/run', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                // Animate backend logs into SYSTEM LOG one by one
                const backendLogs = data.logs || [];
                let logIdx = 0;

                const logTimer = setInterval(() => {
                    if (logIdx < backendLogs.length) {
                        setLog(prev => [...prev, backendLogs[logIdx]]);
                        logIdx++;
                    } else {
                        clearInterval(logTimer);
                    }
                }, 180);

                // Wait for both progress bars and log animations to finish
                const progressDone = sources.length * 300 + 20 * 120;
                const logsDone = backendLogs.length * 180;
                const waitMs = Math.max(progressDone, logsDone) + 400;

                setTimeout(() => {
                    if (data.graph_data) {
                        setGraphData(data.graph_data);
                    }
                    setPhase("done");
                }, waitMs);
            })
            .catch(err => {
                console.error("Pipeline failed:", err);
                setLog(prev => [
                    ...prev,
                    `ERROR: ${err.message}`,
                    "Falling back to static graph_output.json..."
                ]);

                // Fallback: try loading pre-built static graph_output.json
                loadGraph()
                    .then(data => {
                        setGraphData(data);
                        setLog(prev => [...prev, "Fallback data loaded successfully."]);
                    })
                    .catch(() => {
                        setLog(prev => [...prev, "ERROR: No graph data available."]);
                    })
                    .finally(() => setPhase("done"));
            });
    };

    return { phase, progress, log, setLog, run, graphData };
}
