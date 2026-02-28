export const fmtINR = v => v >= 100000 ? "₹" + (v / 100000).toFixed(1) + "L" : "₹" + (v / 1000).toFixed(0) + "K";
export const formatDate = d => new Date(d).toLocaleDateString("en-IN");
export const safeFloat = v => parseFloat(v) || 0;
