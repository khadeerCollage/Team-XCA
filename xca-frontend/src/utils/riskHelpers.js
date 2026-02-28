import { C } from '../constants/colors';

export const getRiskColor = r => ({
    CRITICAL: C.critical,
    HIGH: C.red,
    MEDIUM: C.amber,
    LOW: C.green
}[r] || C.text3);

export const getRiskLabel = r => String(r).toUpperCase();
