import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';

export default function LoginScreen() {
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [lockTimer, setLockTimer] = useState(0);
    const [showPassword, setShowPassword] = useState(false);
    const usernameRef = useRef(null);

    useEffect(() => { usernameRef.current?.focus(); }, []);
    useEffect(() => {
        if (lockTimer <= 0) return;
        const t = setInterval(() => { setLockTimer(prev => { if (prev <= 1) { clearInterval(t); setError(''); return 0; } return prev - 1; }); }, 1000);
        return () => clearInterval(t);
    }, [lockTimer]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (loading || lockTimer > 0) return;
        if (!username.trim() || !password) { setError('Enter both fields'); return; }
        setLoading(true); setError('');
        const result = await login(username, password);
        if (!result.success) { setError(result.error); if (result.locked) setLockTimer(result.remainingSec || 60); setPassword(''); }
        setLoading(false);
    };

    const isLocked = lockTimer > 0;
    const inputStyle = {
        width: '100%', padding: '10px 12px', background: '#ffffff', border: '1px solid #d1d9e6',
        borderRadius: 6, fontSize: 13, color: '#0f172a', outline: 'none', fontFamily: 'inherit'
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8f9fc', fontFamily: "'Source Sans 3', sans-serif" }}>
            {/* Ambient glow */}
            <div style={{ position: 'fixed', top: '20%', left: '50%', transform: 'translateX(-50%)', width: 400, height: 400, background: 'radial-gradient(circle, rgba(30,64,175,0.08) 0%, transparent 70%)', pointerEvents: 'none' }} />

            <div style={{
                width: 380, background: '#ffffff', border: '1px solid #d1d9e6', borderRadius: 12, padding: '40px 32px',
                position: 'relative', zIndex: 1, animation: 'slideUp 0.4s ease-out'
            }}>
                <div style={{ textAlign: 'center', marginBottom: 32 }}>
                    <div style={{
                        width: 56, height: 56, margin: '0 auto 16px', display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                        <img src="/emblem.svg" alt="Emblem" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    </div>
                    <h1 className="serif-header" style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px', letterSpacing: '-0.3px' }}>GST Reconciliation Engine</h1>
                    <p style={{ color: '#475569', fontSize: 14, margin: 0 }}>Sign in to access the audit dashboard</p>
                </div>

                <form onSubmit={handleSubmit}>
                    <div style={{ marginBottom: 12 }}>
                        <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#475569', marginBottom: 4 }}>Username</label>
                        <input ref={usernameRef} type="text" value={username} onChange={e => setUsername(e.target.value)}
                            placeholder="admin" disabled={isLocked} autoComplete="username" style={inputStyle}
                            onFocus={e => e.target.style.borderColor = '#1e40af'} onBlur={e => e.target.style.borderColor = '#d1d9e6'} />
                    </div>
                    <div style={{ marginBottom: 16 }}>
                        <label style={{ display: 'block', fontSize: 14, fontWeight: 500, color: '#475569', marginBottom: 4 }}>Password</label>
                        <div style={{ position: 'relative' }}>
                            <input type={showPassword ? 'text' : 'password'} value={password} onChange={e => setPassword(e.target.value)}
                                placeholder="••••••••" disabled={isLocked} autoComplete="current-password"
                                style={{ ...inputStyle, paddingRight: 40 }}
                                onFocus={e => e.target.style.borderColor = '#1e40af'} onBlur={e => e.target.style.borderColor = '#d1d9e6'} />
                            <button type="button" onClick={() => setShowPassword(!showPassword)}
                                style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#475569', fontSize: 14, fontFamily: 'inherit' }}>
                                {showPassword ? 'HIDE' : 'SHOW'}
                            </button>
                        </div>
                    </div>

                    {error && (
                        <div style={{ background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.2)', borderRadius: 6, padding: '8px 12px', marginBottom: 12, color: '#dc2626', fontSize: 14 }}>
                            {isLocked ? `🔒 Locked — ${lockTimer}s` : error}
                        </div>
                    )}

                    <button type="submit" disabled={loading || isLocked} style={{
                        width: '100%', padding: '10px', background: loading || isLocked ? '#d1d9e6' : '#1e40af',
                        color: loading || isLocked ? '#475569' : '#fff', border: 'none', borderRadius: 6,
                        fontSize: 13, fontWeight: 600, cursor: loading || isLocked ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
                        transition: 'all 0.15s',
                        boxShadow: loading || isLocked ? 'none' : '0 0 20px rgba(30,64,175,0.2)'
                    }}>
                        {loading ? 'Signing in...' : isLocked ? `Locked (${lockTimer}s)` : 'Sign in'}
                    </button>
                </form>

                <div style={{ marginTop: 16, textAlign: 'center', color: '#94a3b8', fontSize: 14 }}>
                    Let the people know the facts, and the country will be safe.
                </div>
            </div>
        </div>
    );
}
