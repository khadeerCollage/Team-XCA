export const getNodeAt = (nodes, x, y) => nodes.find(n => {
    const r = 13 + Math.sqrt(n.tx) * 0.9;
    return Math.hypot(n.x - x, n.y - y) < r + 6;
});

export const buildAdjacency = (nodes, edges) => {
    const adj = {};
    nodes.forEach(n => adj[n.id] = []);
    edges.forEach(e => {
        if (adj[e.source.id]) adj[e.source.id].push(e.target.id);
    });
    return adj;
};

export const detectCycles = (adj) => {
    // offline stub port of Python cycle detection
    return [];
};
