"""
Report Templates — centralized template strings for audit report generation.
Separated from logic for easy maintenance and future multi-language support.
"""

# ─── Main report templates ────────────────────────────────────────────

REPORT_VALID = (
    "Invoice #{invoice_number} claimed by {buyer_name} (GSTIN: {buyer_gstin_short}) "
    "for Rs.{itc_amount} ITC from {seller_name} is VALID.\n"
    "All {passed_count} verification checkpoints passed. "
    "{irn_line}{eway_line}"
    "Chain: {chain_summary}.\n"
    "Risk Level: {risk_level}. Risk Score: {risk_score}/100.\n"
    "Final Decision: {decision}."
)

REPORT_INVALID = (
    "Invoice #{invoice_number} claimed by {buyer_name} (GSTIN: {buyer_gstin_short}) "
    "for Rs.{itc_amount} ITC from {seller_name} is INVALID.\n"
    "{explanation}\n"
    "Chain broken at {break_point} checkpoint (Step {break_step}).\n"
    "Chain: {chain_summary}.\n"
    "Final Decision: {decision}."
)

REPORT_CRITICAL = (
    "Invoice #{invoice_number} claimed by {buyer_name} (GSTIN: {buyer_gstin_short}) "
    "for Rs.{itc_amount} ITC from {seller_name} is INVALID.\n"
    "CRITICAL FAILURE: All verification checkpoints have failed.\n"
    "{explanation}\n"
    "Chain: {chain_summary}.\n"
    "Final Decision: {decision}."
)


# ─── Checkpoint status symbols ────────────────────────────────────────

SYMBOL_PASS = "PASS"
SYMBOL_FAIL = "FAIL"
SYMBOL_SKIP = "SKIP"


# ─── Explanation fragments per checkpoint failure ─────────────────────

FAILURE_EXPLANATIONS = {
    "GSTR1": "Seller has not filed GSTR-1. Invoice not visible on GST portal.",
    "GSTR3B": "Seller has not filed GSTR-3B. Tax payment could not be verified.",
    "TAX": "Tax has not been paid by the seller to the government.",
    "IRN": "Invoice Registration Number (IRN) is invalid or not found on IRP.",
    "EWB": "e-Way Bill verification failed. Transport compliance not confirmed.",
}

# ─── Checkpoint display names for chain summary ──────────────────────

CHECKPOINT_DISPLAY = {
    "GSTR1": "GSTR1",
    "GSTR3B": "GSTR3B",
    "TAX": "TAX",
    "IRN": "IRN",
    "EWB": "EWB",
}
