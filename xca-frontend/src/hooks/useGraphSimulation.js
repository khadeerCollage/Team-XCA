import { useEffect, useRef } from "react";
import { nodeCol } from "../constants/riskLevels";

export function useGraphSimulation(NODES, EDGES, cvRef, setSelected) {
    const animRef = useRef(null);
    const dragRef = useRef(null);
    const nsRef = useRef([]);
    const zoomRef = useRef({ scale: 1, ox: 0, oy: 0 }); // zoom & pan state

    useEffect(() => {
        const cv = cvRef.current; if (!cv) return;
        const W = cv.offsetWidth, H = cv.offsetHeight;
        cv.width = W; cv.height = H;
        const ctx = cv.getContext("2d");

        // Reset zoom on data change
        zoomRef.current = { scale: 1, ox: 0, oy: 0 };

        const ns = NODES.map((n, i) => ({
            ...n,
            x: W / 2 + Math.cos((i / NODES.length) * Math.PI * 2) * 200,
            y: H / 2 + Math.sin((i / NODES.length) * Math.PI * 2) * 200,
            vx: 0, vy: 0,
        }));
        const es = EDGES.map(e => ({
            ...e,
            source: ns.find(n => n.id === e.s),
            target: ns.find(n => n.id === e.t),
        }));
        nsRef.current = ns;

        // Zoom handler
        const onWheel = (e) => {
            e.preventDefault();
            const z = zoomRef.current;
            const rect = cv.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.3, Math.min(5, z.scale * delta));

            // Zoom towards mouse position
            z.ox = mx - (mx - z.ox) * (newScale / z.scale);
            z.oy = my - (my - z.oy) * (newScale / z.scale);
            z.scale = newScale;
        };
        cv.addEventListener("wheel", onWheel, { passive: false });

        let tick = 0;
        function step() {
            tick++;
            const a = Math.max(0.001, 1 - tick / 200);
            ns.forEach(n => { n.vx *= 0.75; n.vy *= 0.75; });
            for (let i = 0; i < ns.length; i++)
                for (let j = i + 1; j < ns.length; j++) {
                    const dx = ns[j].x - ns[i].x, dy = ns[j].y - ns[i].y;
                    const d = Math.sqrt(dx * dx + dy * dy) || 1;
                    const f = (2000 / (d * d)) * a;
                    ns[i].vx -= (dx / d) * f; ns[i].vy -= (dy / d) * f;
                    ns[j].vx += (dx / d) * f; ns[j].vy += (dy / d) * f;
                }
            es.forEach(e => {
                const dx = e.target.x - e.source.x, dy = e.target.y - e.source.y;
                const d = Math.sqrt(dx * dx + dy * dy) || 1;
                const f = ((d - 160) / d) * 0.12 * a;
                e.source.vx += dx * f; e.source.vy += dy * f;
                e.target.vx -= dx * f; e.target.vy -= dy * f;
            });
            ns.forEach(n => {
                n.vx += (W / 2 - n.x) * 0.008 * a;
                n.vy += (H / 2 - n.y) * 0.008 * a;
                if (!n._drag) { n.x += n.vx; n.y += n.vy; }
                n.x = Math.max(36, Math.min(W - 36, n.x));
                n.y = Math.max(36, Math.min(H - 36, n.y));
            });
        }

        function draw() {
            const z = zoomRef.current;
            ctx.clearRect(0, 0, W, H);

            // Apply zoom transform
            ctx.save();
            ctx.translate(z.ox, z.oy);
            ctx.scale(z.scale, z.scale);

            // Grid dots
            ctx.fillStyle = "#0f172a08";
            const gridStep = 30;
            const startX = -z.ox / z.scale, startY = -z.oy / z.scale;
            const endX = (W - z.ox) / z.scale, endY = (H - z.oy) / z.scale;
            for (let x = Math.floor(startX / gridStep) * gridStep; x < endX; x += gridStep)
                for (let y = Math.floor(startY / gridStep) * gridStep; y < endY; y += gridStep) {
                    ctx.beginPath(); ctx.arc(x, y, 0.8 / z.scale, 0, Math.PI * 2); ctx.fill();
                }

            es.forEach(e => {
                const dx = e.target.x - e.source.x, dy = e.target.y - e.source.y;
                const d = Math.sqrt(dx * dx + dy * dy) || 1;
                ctx.beginPath();
                ctx.moveTo(e.source.x, e.source.y);
                ctx.lineTo(e.target.x, e.target.y);
                ctx.strokeStyle = e.ok ? "#00d48a33" : "#ff3b5c55";
                ctx.lineWidth = (e.ok ? 1 : 1.8) / z.scale;
                if (!e.ok) ctx.setLineDash([5 / z.scale, 4 / z.scale]); else ctx.setLineDash([]);
                ctx.stroke(); ctx.setLineDash([]);
                const r = 20, ax = e.target.x - (dx / d) * r, ay = e.target.y - (dy / d) * r;
                const ang = Math.atan2(dy, dx);
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(ax - 9 * Math.cos(ang - 0.45), ay - 9 * Math.sin(ang - 0.45));
                ctx.lineTo(ax - 9 * Math.cos(ang + 0.45), ay - 9 * Math.sin(ang + 0.45));
                ctx.closePath();
                ctx.fillStyle = e.ok ? "#00d48a55" : "#ff3b5c77";
                ctx.fill();
            });

            ns.forEach(n => {
                const r = 13 + Math.sqrt(n.tx) * 0.9;
                const col = nodeCol(n.risk);
                if (n.risk === "high") {
                    const g = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, r + 18);
                    g.addColorStop(0, col + "44"); g.addColorStop(1, col + "00");
                    ctx.beginPath(); ctx.arc(n.x, n.y, r + 18, 0, Math.PI * 2);
                    ctx.fillStyle = g; ctx.fill();
                }
                ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
                ctx.fillStyle = col + "1a"; ctx.fill();
                ctx.strokeStyle = col; ctx.lineWidth = (n._sel ? 3 : 1.5) / z.scale; ctx.stroke();
                ctx.fillStyle = "#0f172a";
                ctx.font = `${n._sel ? "bold " : ""}${13 / z.scale}px 'IBM Plex Mono', monospace`;
                ctx.textAlign = "center";
                const shortLabel = n.label.split(" ")[0];
                ctx.fillText(shortLabel, n.x, n.y + r + 14);
                if (n.risk === "high") {
                    ctx.fillStyle = col + "cc";
                    ctx.font = `bold ${11 / z.scale}px monospace`;
                    ctx.fillText(n.score.toFixed(2), n.x, n.y + 4);
                }
            });

            ctx.restore();

            // Zoom indicator (outside transform)
            ctx.fillStyle = "#47556988";
            ctx.font = "11px monospace";
            ctx.textAlign = "right";
            ctx.fillText(`${Math.round(z.scale * 100)}%`, W - 12, H - 10);
        }

        function loop() { step(); draw(); animRef.current = requestAnimationFrame(loop); }
        loop();
        return () => {
            cancelAnimationFrame(animRef.current);
            cv.removeEventListener("wheel", onWheel);
        };
    }, [NODES, EDGES, cvRef]);

    // Convert screen coords to graph coords (accounting for zoom)
    const screenToGraph = (sx, sy) => {
        const z = zoomRef.current;
        return { x: (sx - z.ox) / z.scale, y: (sy - z.oy) / z.scale };
    };

    const didDrag = useRef(false);
    const panRef = useRef(null); // for panning empty canvas

    const onDown = (n) => {
        if (n) {
            n._drag = true; dragRef.current = n; didDrag.current = false;
        } else {
            // Start panning
            panRef.current = { startOx: zoomRef.current.ox, startOy: zoomRef.current.oy, sx: 0, sy: 0 };
        }
    };
    const onUp = (n) => {
        if (panRef.current) { panRef.current = null; return; }
        if (dragRef.current) {
            dragRef.current._drag = false;
            const wasDrag = didDrag.current;
            dragRef.current = null;
            didDrag.current = false;
            if (!wasDrag) {
                if (n) { nsRef.current.forEach(nd => nd._sel = false); n._sel = true; setSelected(n); }
            }
            return;
        }
        if (n) { nsRef.current.forEach(nd => nd._sel = false); n._sel = true; setSelected(n); }
    };
    const onMoveDrag = (x, y) => {
        if (dragRef.current) {
            const g = screenToGraph(x, y);
            dragRef.current.x = g.x; dragRef.current.y = g.y;
            didDrag.current = true;
            return true;
        }
        if (panRef.current) {
            const p = panRef.current;
            if (p.sx === 0 && p.sy === 0) { p.sx = x; p.sy = y; }
            zoomRef.current.ox = p.startOx + (x - p.sx);
            zoomRef.current.oy = p.startOy + (y - p.sy);
            return true;
        }
        return false;
    };

    return { nsRef, onDown, onUp, onMoveDrag, screenToGraph };
}
