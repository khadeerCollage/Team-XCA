import { C } from './colors';

export const riskCol = r => ({
    CRITICAL: C.critical,
    HIGH: C.red,
    MEDIUM: C.amber,
    LOW: C.green
}[r] || C.text3);

export const nodeCol = r => ({
    high: C.red,
    medium: C.amber,
    low: C.green
}[r] || C.text3);
