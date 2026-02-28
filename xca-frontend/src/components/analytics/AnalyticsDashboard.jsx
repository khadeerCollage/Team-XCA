import { useMemo } from 'react';
import {
    PieChart, Pie, Cell, Tooltip as RechartsTooltip, ResponsiveContainer,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
    ScatterChart, Scatter, ZAxis,
} from 'recharts';
import { fmtINR } from '../../utils/formatters';

export default function AnalyticsDashboard({ nodes, mismatches }) {
    // 1. Donut Chart - Risk Distribution (Total ITC by Risk)
    const riskDistribution = useMemo(() => {
        const data = [
            { name: 'High Risk', value: 0, color: '#ef4444' }, // red
            { name: 'Medium Risk', value: 0, color: '#f59e0b' }, // amber
            { name: 'Low Risk', value: 0, color: '#10b981' }, // green
        ];
        nodes.forEach(n => {
            if (n.risk === 'high') data[0].value += n.itc;
            else if (n.risk === 'medium') data[1].value += n.itc;
            else data[2].value += n.itc;
        });
        return data;
    }, [nodes]);

    // 2. Bar Chart - State-wise Risk
    const stateRisk = useMemo(() => {
        const states = {};
        nodes.forEach(n => {
            if (!states[n.state]) states[n.state] = { state: n.state, highRiskItc: 0, vendorCount: 0 };
            if (n.risk === 'high') states[n.state].highRiskItc += n.itc;
            states[n.state].vendorCount += 1;
        });
        // Sort by high risk ITC
        return Object.values(states).sort((a, b) => b.highRiskItc - a.highRiskItc).slice(0, 8);
    }, [nodes]);

    // 3. Scatter Plot - Score vs ITC at Risk
    const scatterData = useMemo(() => {
        return nodes.map(n => ({
            vendor: n.label,
            score: n.score,
            itc: n.itc,
            color: n.risk === 'high' ? '#ef4444' : n.risk === 'medium' ? '#f59e0b' : '#10b981'
        }));
    }, [nodes]);

    // 4. Stacked Bar - Mismatch Types Over Time (Mocked monthly data preserving total mismatch context)
    const trendData = useMemo(() => {
        return [
            { month: 'Apr', valueDelta: 12, circular: 2, nonFiler: 4 },
            { month: 'May', valueDelta: 15, circular: 3, nonFiler: 5 },
            { month: 'Jun', valueDelta: 10, circular: 4, nonFiler: 3 },
            { month: 'Jul', valueDelta: 18, circular: 6, nonFiler: 4 },
            { month: 'Aug', valueDelta: 14, circular: 8, nonFiler: 2 },
            { month: 'Sep', valueDelta: 8, circular: 10, nonFiler: 2 } // Circular increasing!
        ];
    }, []);

    const CustomTooltip = ({ active, payload, label, formatter }) => {
        if (active && payload && payload.length) {
            return (
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 6, padding: '10px 14px', boxShadow: '0 8px 24px rgba(0,0,0,0.4)' }}>
                    {label && <div style={{ color: '#fafafa', fontWeight: 600, marginBottom: 8, fontSize: 13 }}>{label}</div>}
                    {payload.map((entry, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 4 }}>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: entry.color }} />
                            <span style={{ color: '#a1a1aa' }}>{entry.name || 'Value'}:</span>
                            <span style={{ color: '#fafafa', fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>
                                {formatter ? formatter(entry.value) : entry.value}
                            </span>
                        </div>
                    ))}
                </div>
            );
        }
        return null;
    };

    const ScatterTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 6, padding: '10px 14px', boxShadow: '0 8px 24px rgba(0,0,0,0.4)' }}>
                    <div style={{ color: '#fafafa', fontWeight: 600, marginBottom: 8, fontSize: 13 }}>{data.vendor}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, marginBottom: 4 }}>
                        <span style={{ color: '#a1a1aa' }}>Risk Score:</span>
                        <span style={{ color: data.color, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{data.score.toFixed(2)}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                        <span style={{ color: '#a1a1aa' }}>ITC at Risk:</span>
                        <span style={{ color: '#fafafa', fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{fmtINR(data.itc)}</span>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div style={{ height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16, fontFamily: "'Source Sans 3', sans-serif" }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

                {/* 1. Donut Chart */}
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, padding: 20 }}>
                    <div style={{ color: '#fafafa', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Risk Distribution</div>
                    <div style={{ color: '#52525b', fontSize: 12, marginBottom: 16 }}>Total ITC apportioned by risk tier</div>
                    <div style={{ height: 260 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie data={riskDistribution} innerRadius={70} outerRadius={100} paddingAngle={4} dataKey="value" stroke="none">
                                    {riskDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <RechartsTooltip content={<CustomTooltip formatter={(val) => fmtINR(val)} />} />
                                <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: 12, color: '#a1a1aa' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* 2. Bar Chart */}
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, padding: 20 }}>
                    <div style={{ color: '#fafafa', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>State-wise Risk Exposure</div>
                    <div style={{ color: '#52525b', fontSize: 12, marginBottom: 16 }}>High-risk ITC concentrated by geographical region</div>
                    <div style={{ height: 260 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={stateRisk} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                <XAxis dataKey="state" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
                                <YAxis stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val / 100000).toFixed(0)}L`} />
                                <RechartsTooltip content={<CustomTooltip formatter={(val) => fmtINR(val)} />} cursor={{ fill: '#27272a' }} />
                                <Bar dataKey="highRiskItc" name="High Risk ITC" fill="#ef4444" radius={[4, 4, 0, 0]} barSize={24} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* 3. Scatter Plot */}
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, padding: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                            <div style={{ color: '#fafafa', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Risk Matrix Quadrant</div>
                            <div style={{ color: '#52525b', fontSize: 12, marginBottom: 16 }}>Score vs ITC. Top Right = Act Immediately</div>
                        </div>
                    </div>
                    <div style={{ height: 260, position: 'relative' }}>
                        <div style={{ position: 'absolute', top: 0, right: 0, bottom: 0, left: 0, border: '1px solid #27272a', borderRadius: 4, pointerEvents: 'none' }} />
                        <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, borderTop: '1px dashed #3f3f46', pointerEvents: 'none' }} />
                        <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, borderLeft: '1px dashed #3f3f46', pointerEvents: 'none' }} />
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <XAxis type="number" dataKey="score" name="Risk Score" domain={[0, 1]} stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
                                <YAxis type="number" dataKey="itc" name="ITC at Risk" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val / 100000).toFixed(0)}L`} />
                                <ZAxis type="number" range={[40, 40]} />
                                <RechartsTooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                                <Scatter data={scatterData}>
                                    {scatterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.6} stroke={entry.color} strokeWidth={1} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* 4. Stacked Bar */}
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, padding: 20 }}>
                    <div style={{ color: '#fafafa', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Anomaly Trends</div>
                    <div style={{ color: '#52525b', fontSize: 12, marginBottom: 16 }}>Fraud typologies distribution over last 6 months</div>
                    <div style={{ height: 260 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={trendData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                <XAxis dataKey="month" stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
                                <YAxis stroke="#52525b" fontSize={11} tickLine={false} axisLine={false} />
                                <RechartsTooltip content={<CustomTooltip />} cursor={{ fill: '#27272a' }} />
                                <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: 12, color: '#a1a1aa' }} />
                                <Bar dataKey="circular" name="Circular Trading" stackId="a" fill="#ef4444" barSize={32} />
                                <Bar dataKey="nonFiler" name="Non-Filer GSTR-3B" stackId="a" fill="#f59e0b" />
                                <Bar dataKey="valueDelta" name="Value Mis-declaration" stackId="a" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

            </div>
        </div>
    );
}
