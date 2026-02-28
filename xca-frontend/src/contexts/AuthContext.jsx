import { createContext, useContext, useState, useCallback } from 'react';

const AuthContext = createContext(null);

// Hardcoded credentials for hackathon demo
// In production, this would be a backend API call
const VALID_USERS = [
    { username: 'admin', passwordHash: 'admin123', role: 'auditor', name: 'GST Auditor' },
];

const MAX_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 60 * 1000; // 1 minute lockout
const PROGRESSIVE_DELAY_BASE = 1000; // 1s base delay, increases with each attempt

export function AuthProvider({ children }) {
    const [user, setUser] = useState(() => {
        // Restore session from sessionStorage (cleared on tab close)
        const saved = sessionStorage.getItem('gst_auth_session');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                // Session expires after 4 hours
                if (Date.now() - parsed.loginTime < 4 * 60 * 60 * 1000) {
                    return parsed;
                }
                sessionStorage.removeItem('gst_auth_session');
            } catch { /* invalid session */ }
        }
        return null;
    });

    const [attempts, setAttempts] = useState(0);
    const [lockUntil, setLockUntil] = useState(0);
    const [lastAttemptTime, setLastAttemptTime] = useState(0);

    const login = useCallback(async (username, password) => {
        const now = Date.now();

        // Check lockout
        if (now < lockUntil) {
            const remainingSec = Math.ceil((lockUntil - now) / 1000);
            return {
                success: false,
                error: `Account locked. Try again in ${remainingSec}s`,
                locked: true,
                remainingSec
            };
        }

        // Progressive delay — force wait between attempts
        const delay = attempts * PROGRESSIVE_DELAY_BASE;
        if (delay > 0 && now - lastAttemptTime < delay) {
            return {
                success: false,
                error: `Too fast. Wait ${Math.ceil((delay - (now - lastAttemptTime)) / 1000)}s before trying again`,
                rateLimit: true
            };
        }

        setLastAttemptTime(now);

        // Simulate network delay (prevents timing attacks in demo)
        await new Promise(r => setTimeout(r, 300 + Math.random() * 200));

        // Validate credentials
        const matched = VALID_USERS.find(
            u => u.username === username.toLowerCase().trim() && u.passwordHash === password
        );

        if (!matched) {
            const newAttempts = attempts + 1;
            setAttempts(newAttempts);

            if (newAttempts >= MAX_ATTEMPTS) {
                const lockTime = now + LOCKOUT_DURATION_MS;
                setLockUntil(lockTime);
                setAttempts(0);
                return {
                    success: false,
                    error: `Too many failed attempts. Locked for 60 seconds.`,
                    locked: true,
                    remainingSec: 60
                };
            }

            return {
                success: false,
                error: `Invalid credentials. ${MAX_ATTEMPTS - newAttempts} attempts remaining.`,
                attemptsLeft: MAX_ATTEMPTS - newAttempts
            };
        }

        // Success
        const session = {
            username: matched.username,
            role: matched.role,
            name: matched.name,
            loginTime: now,
            sessionId: crypto.randomUUID ? crypto.randomUUID() : `${now}-${Math.random().toString(36).slice(2)}`
        };

        sessionStorage.setItem('gst_auth_session', JSON.stringify(session));
        setUser(session);
        setAttempts(0);
        setLockUntil(0);

        return { success: true, user: session };
    }, [attempts, lockUntil, lastAttemptTime]);

    const logout = useCallback(() => {
        sessionStorage.removeItem('gst_auth_session');
        setUser(null);
        setAttempts(0);
    }, []);

    return (
        <AuthContext.Provider value={{ user, login, logout, isAuthenticated: !!user }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
}
