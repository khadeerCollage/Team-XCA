"""
Comprehensive tests for the GST Audit Report Engine.
Covers all acceptance criteria from the PRD:
    ✓ Valid chain
    ✓ GSTR-1 failure
    ✓ GSTR-3B failure
    ✓ Tax payment failure
    ✓ IRN failure
    ✓ e-Way Bill failure
    ✓ All checkpoints fail (critical)
    ✓ Multiple failures
    ✓ No route (e-Way skipped)
    ✓ Risk classification correctness
    ✓ Input validation / edge cases
"""

import pytest
from app.core.models import ValidationInput, RiskLevel, Decision, CheckpointStatus
from app.core.engine import generate_audit_report
from app.core.validator import evaluate_chain, find_break_point, is_chain_valid
from app.core.risk import determine_risk_level, calculate_risk_score, generate_decision


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _base_input(**overrides) -> ValidationInput:
    """Factory for valid input — override specific fields for test scenarios."""
    defaults = {
        "invoice_number": "INV-2024-882",
        "buyer_name": "Ananya Textiles",
        "buyer_gstin": "27ABCDE1234F1Z5",
        "seller_name": "Vikram Fabrics",
        "seller_gstin": "29ABCDE5678G1Z9",
        "itc_amount": 60000,
        "gstr1_filed": True,
        "gstr3b_filed": True,
        "tax_paid": True,
        "irn_valid": True,
        "eway_bill_verified": True,
        "route": "BLR-MUM",
    }
    defaults.update(overrides)
    return ValidationInput(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. VALID CHAIN — All checkpoints pass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestValidChain:
    def test_all_pass_chain_is_valid(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.chain_valid is True

    def test_all_pass_risk_is_low(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.LOW

    def test_all_pass_decision_approved(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.decision == Decision.APPROVED

    def test_all_pass_risk_score_100(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.risk_score == 100

    def test_all_pass_no_break_point(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.break_point is None
        assert result.break_step is None

    def test_all_pass_report_contains_valid(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert "VALID" in result.report
        assert "ITC APPROVED" in result.report
        assert "INV-2024-882" in result.report
        assert "Ananya Textiles" in result.report
        assert "BLR-MUM" in result.report

    def test_all_pass_checkpoints_count(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert len(result.checkpoints) == 5
        assert all(cp.status == CheckpointStatus.PASS for cp in result.checkpoints)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GSTR-1 FAILURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGSTR1Failure:
    def test_gstr1_fail_chain_broken(self):
        data = _base_input(gstr1_filed=False)
        result = generate_audit_report(data)
        assert result.chain_valid is False

    def test_gstr1_only_fail_risk_medium(self):
        """GSTR-1 alone is a minor failure → MEDIUM risk."""
        data = _base_input(gstr1_filed=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_gstr1_fail_break_point(self):
        data = _base_input(gstr1_filed=False)
        result = generate_audit_report(data)
        assert result.break_point == "GSTR1"
        assert result.break_step == 1

    def test_gstr1_fail_report_contains_invalid(self):
        data = _base_input(gstr1_filed=False)
        result = generate_audit_report(data)
        assert "INVALID" in result.report
        assert "GSTR1" in result.report or "GSTR-1" in result.report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GSTR-3B FAILURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGSTR3BFailure:
    def test_gstr3b_fail_high_risk(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.HIGH

    def test_gstr3b_fail_rejected(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert result.decision == Decision.REJECTED

    def test_gstr3b_fail_break_at_step_2(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert result.break_point == "GSTR3B"
        assert result.break_step == 2

    def test_gstr3b_fail_report_text(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert "INVALID" in result.report
        assert "ITC REJECTED" in result.report
        assert "GSTR-3B" in result.report or "GSTR3B" in result.report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TAX PAYMENT FAILURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTaxPaymentFailure:
    def test_tax_unpaid_high_risk(self):
        data = _base_input(tax_paid=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.HIGH

    def test_tax_unpaid_rejected(self):
        data = _base_input(tax_paid=False)
        result = generate_audit_report(data)
        assert result.decision == Decision.REJECTED

    def test_tax_unpaid_break_at_step_3(self):
        data = _base_input(tax_paid=False)
        result = generate_audit_report(data)
        assert result.break_point == "TAX"
        assert result.break_step == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. IRN FAILURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIRNFailure:
    def test_irn_invalid_high_risk(self):
        data = _base_input(irn_valid=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.HIGH

    def test_irn_invalid_rejected(self):
        data = _base_input(irn_valid=False)
        result = generate_audit_report(data)
        assert result.decision == Decision.REJECTED

    def test_irn_invalid_break_at_step_4(self):
        data = _base_input(irn_valid=False)
        result = generate_audit_report(data)
        assert result.break_point == "IRN"
        assert result.break_step == 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. e-WAY BILL FAILURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEWayBillFailure:
    def test_eway_fail_medium_risk(self):
        """e-Way Bill alone is a minor failure → MEDIUM risk."""
        data = _base_input(eway_bill_verified=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.MEDIUM

    def test_eway_fail_still_approved(self):
        """MEDIUM risk → ITC still APPROVED (with warning)."""
        data = _base_input(eway_bill_verified=False)
        result = generate_audit_report(data)
        assert result.decision == Decision.APPROVED

    def test_eway_fail_break_at_step_5(self):
        data = _base_input(eway_bill_verified=False)
        result = generate_audit_report(data)
        assert result.break_point == "EWB"
        assert result.break_step == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. e-WAY BILL SKIPPED (no route)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEWaySkipped:
    def test_no_route_eway_skipped(self):
        data = _base_input(route=None)
        result = generate_audit_report(data)
        eway_cp = next(cp for cp in result.checkpoints if cp.name == "EWB")
        assert eway_cp.status == CheckpointStatus.SKIPPED

    def test_no_route_chain_still_valid(self):
        data = _base_input(route=None)
        result = generate_audit_report(data)
        assert result.chain_valid is True
        assert result.decision == Decision.APPROVED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ALL CHECKPOINTS FAIL — CRITICAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCriticalFailure:
    def test_all_fail_critical_risk(self):
        data = _base_input(
            gstr1_filed=False,
            gstr3b_filed=False,
            tax_paid=False,
            irn_valid=False,
            eway_bill_verified=False,
        )
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.CRITICAL

    def test_all_fail_rejected(self):
        data = _base_input(
            gstr1_filed=False,
            gstr3b_filed=False,
            tax_paid=False,
            irn_valid=False,
            eway_bill_verified=False,
        )
        result = generate_audit_report(data)
        assert result.decision == Decision.REJECTED

    def test_all_fail_risk_score_zero(self):
        data = _base_input(
            gstr1_filed=False,
            gstr3b_filed=False,
            tax_paid=False,
            irn_valid=False,
            eway_bill_verified=False,
        )
        result = generate_audit_report(data)
        assert result.risk_score == 0

    def test_all_fail_report_critical(self):
        data = _base_input(
            gstr1_filed=False,
            gstr3b_filed=False,
            tax_paid=False,
            irn_valid=False,
            eway_bill_verified=False,
        )
        result = generate_audit_report(data)
        assert "CRITICAL" in result.report
        assert "ITC REJECTED" in result.report

    def test_all_fail_first_break_is_gstr1(self):
        data = _base_input(
            gstr1_filed=False,
            gstr3b_filed=False,
            tax_paid=False,
            irn_valid=False,
            eway_bill_verified=False,
        )
        result = generate_audit_report(data)
        assert result.break_point == "GSTR1"
        assert result.break_step == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. MULTIPLE FAILURES (not all)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultipleFailures:
    def test_gstr3b_and_tax_fail(self):
        data = _base_input(gstr3b_filed=False, tax_paid=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.HIGH
        assert result.decision == Decision.REJECTED
        # First break is GSTR3B (step 2)
        assert result.break_point == "GSTR3B"

    def test_gstr1_and_eway_fail_medium(self):
        """Only minor checkpoints fail → MEDIUM (no major checkpoint hit)."""
        data = _base_input(gstr1_filed=False, eway_bill_verified=False)
        result = generate_audit_report(data)
        assert result.risk_level == RiskLevel.MEDIUM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. RISK SCORE CALCULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRiskScore:
    def test_score_all_pass(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert result.risk_score == 100

    def test_score_gstr1_fail(self):
        data = _base_input(gstr1_filed=False)
        result = generate_audit_report(data)
        assert result.risk_score == 85  # 100 - 15

    def test_score_gstr3b_fail(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert result.risk_score == 75  # 100 - 25

    def test_score_tax_fail(self):
        data = _base_input(tax_paid=False)
        result = generate_audit_report(data)
        assert result.risk_score == 70  # 100 - 30

    def test_score_irn_fail(self):
        data = _base_input(irn_valid=False)
        result = generate_audit_report(data)
        assert result.risk_score == 80  # 100 - 20

    def test_score_eway_fail(self):
        data = _base_input(eway_bill_verified=False)
        result = generate_audit_report(data)
        assert result.risk_score == 90  # 100 - 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. INPUT VALIDATION / EDGE CASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInputValidation:
    def test_reject_negative_amount(self):
        with pytest.raises(Exception):
            _base_input(itc_amount=-1000)

    def test_reject_zero_amount(self):
        with pytest.raises(Exception):
            _base_input(itc_amount=0)

    def test_reject_invalid_gstin_length(self):
        with pytest.raises(Exception):
            _base_input(buyer_gstin="ABC123")

    def test_reject_empty_invoice_number(self):
        with pytest.raises(Exception):
            _base_input(invoice_number="")

    def test_reject_unknown_field(self):
        """Extra fields must be rejected (model_config extra='forbid')."""
        with pytest.raises(Exception):
            data = _base_input()
            # Pass unknown field via dict unpacking to trigger validation error
            ValidationInput(**{**data.dict(), "unknown_field": "should_fail"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. CHAIN SUMMARY FORMAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChainSummary:
    def test_all_pass_summary(self):
        data = _base_input()
        result = generate_audit_report(data)
        assert "GSTR1 ✓" in result.chain_summary
        assert "GSTR3B ✓" in result.chain_summary
        assert "TAX ✓" in result.chain_summary
        assert "IRN ✓" in result.chain_summary
        assert "EWB ✓" in result.chain_summary

    def test_gstr3b_fail_summary(self):
        data = _base_input(gstr3b_filed=False)
        result = generate_audit_report(data)
        assert "GSTR3B ✗" in result.chain_summary
        assert "GSTR1 ✓" in result.chain_summary

    def test_eway_skipped_summary(self):
        data = _base_input(route=None)
        result = generate_audit_report(data)
        assert "EWB ○" in result.chain_summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. DETERMINISM — same input → same output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeterminism:
    def test_same_input_same_output(self):
        data = _base_input(gstr3b_filed=False)
        result1 = generate_audit_report(data)
        result2 = generate_audit_report(data)
        assert result1.report == result2.report
        assert result1.risk_level == result2.risk_level
        assert result1.risk_score == result2.risk_score
        assert result1.decision == result2.decision
        assert result1.chain_summary == result2.chain_summary
