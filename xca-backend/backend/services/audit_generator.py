"""
services/audit_generator.py
─────────────────────────────
LLM-powered audit trail generation.
Sends mismatch data to OpenAI and gets a plain English explanation.
"""

import os
from schemas.mismatch import MismatchRecord
from dotenv import load_dotenv

load_dotenv()

def generate_audit_note(mismatch: MismatchRecord) -> str:
    """
    Temporary bypass of OpenAI to force rule-based generation.
    Returns a 3-sentence plain-English audit note explaining what happened.
    """
    return _fallback_audit_note(mismatch)


def _fallback_audit_note(mismatch: MismatchRecord) -> str:
    """
    Rule-based audit note when OpenAI is not available.
    Generates readable notes without any API call.
    """
    delta = abs(mismatch.delta)

    if mismatch.chain_broken_at == "IRN":
        return (
            f"Invoice {mismatch.invoice_no} filed by {mismatch.seller_name} "
            f"does not have a valid IRN registered on the NIC e-Invoice portal, "
            f"which is mandatory for invoices above ₹5 lakh. "
            f"The ITC of ₹{mismatch.itc_blocked:,.2f} claimed by {mismatch.buyer_name} "
            f"is therefore ineligible. Recommended action: {mismatch.seller_name} must "
            f"register the invoice on the e-Invoice portal immediately or the ITC claim will be rejected."
        )

    if mismatch.chain_broken_at == "e-Way Bill":
        return (
            f"Invoice {mismatch.invoice_no} from {mismatch.seller_name} to "
            f"{mismatch.buyer_name} involves physical goods worth ₹{mismatch.gstr1_value:,.2f} "
            f"but no e-Way Bill was found for this shipment. "
            f"Without an e-Way Bill, the physical movement of goods cannot be verified, "
            f"making the ITC claim of ₹{mismatch.itc_blocked:,.2f} suspicious. "
            f"Recommended action: {mismatch.seller_name} must generate an e-Way Bill or "
            f"provide proof of delivery to validate this transaction."
        )

    if mismatch.chain_broken_at == "GSTR-2B":
        return (
            f"Invoice {mismatch.invoice_no} shows a discrepancy of ₹{delta:,.2f} between "
            f"{mismatch.seller_name}'s GSTR-1 (₹{mismatch.gstr1_value:,.2f}) and "
            f"{mismatch.buyer_name}'s GSTR-2B (₹{mismatch.gstr2b_value:,.2f}). "
            f"This is likely a data entry error on the seller's side or a partial amendment "
            f"that has not been reconciled. "
            f"Recommended action: {mismatch.seller_name} should file a GSTR-1 amendment for "
            f"the correct invoice value to allow {mismatch.buyer_name} to claim the full ITC."
        )

    return (
        f"A {mismatch.mismatch_type.value} mismatch was detected on invoice "
        f"{mismatch.invoice_no} between {mismatch.seller_name} and {mismatch.buyer_name} "
        f"with a financial impact of ₹{mismatch.itc_blocked:,.2f} in blocked ITC. "
        f"The verification chain broke at the {mismatch.chain_broken_at} stage. "
        f"Recommended action: Both parties should review their filings for the {mismatch.period} "
        f"period and file corrections as needed."
    )


def batch_generate_audit_notes(
    mismatches: list[MismatchRecord],
    use_ai: bool = True
) -> list[MismatchRecord]:
    """
    Add audit notes to a list of mismatches.
    use_ai=False → use rule-based fallback (no API calls).
    """
    for m in mismatches:
        if use_ai:
            m.audit_note = generate_audit_note(m)
        else:
            m.audit_note = _fallback_audit_note(m)
    return mismatches
