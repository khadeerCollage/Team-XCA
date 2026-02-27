"""
Standalone test suite for the Vendor Risk Scorer.

Usage:
    cd backend
    python -m risk_scorer.test_risk_scorer
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def test_imports():
    print("\n" + "=" * 56)
    print("  TEST 1: Import Verification")
    print("=" * 56)

    modules = [
        ("risk_scorer.filing_scorer", "FilingScorer"),
        ("risk_scorer.dispute_scorer", "DisputeScorer"),
        ("risk_scorer.network_scorer", "NetworkScorer"),
        ("risk_scorer.physical_scorer", "PhysicalScorer"),
        ("risk_scorer.vendor_risk_engine", "VendorRiskEngine"),
        ("risk_scorer.itc_decision", "ITCDecisionEngine"),
        ("risk_scorer.explainer", "generate_vendor_explanation"),
    ]

    for mod_name, cls_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            getattr(mod, cls_name)
            print(f"  ✅ {mod_name:40s} → {cls_name}")
        except Exception as e:
            print(f"  ❌ {mod_name}: {e}")
            return False

    return True


def test_risk_band_logic():
    print("\n" + "=" * 56)
    print("  TEST 2: Risk Band Classification")
    print("=" * 56)

    from risk_scorer.vendor_risk_engine import VendorRiskEngine

    engine = VendorRiskEngine.__new__(VendorRiskEngine)

    test_cases = [
        (95, "SAFE", "green"),
        (80, "SAFE", "green"),
        (79, "MODERATE", "yellow"),
        (50, "MODERATE", "yellow"),
        (49, "HIGH_RISK", "orange"),
        (30, "HIGH_RISK", "orange"),
        (29, "FRAUD", "red"),
        (5, "FRAUD", "red"),
        (0, "FRAUD", "red"),
    ]

    all_pass = True
    for score, expected_cat, expected_color in test_cases:
        cat, color = engine._classify_risk(score)
        passed = cat == expected_cat and color == expected_color
        status = "✅" if passed else "❌"
        print(f"  {status} Score {score:3d} → {cat:10s} ({color})")
        if not passed:
            all_pass = False

    return all_pass


def test_itc_decision_logic():
    print("\n" + "=" * 56)
    print("  TEST 3: ITC Decision Logic")
    print("=" * 56)

    from risk_scorer.itc_decision import ITCDecisionEngine, ITCDecision

    engine = ITCDecisionEngine.__new__(ITCDecisionEngine)

    test_cases = [
        (90, "SAFE", ITCDecision.AUTO_APPROVE),
        (80, "SAFE", ITCDecision.AUTO_APPROVE),
        (65, "MODERATE", ITCDecision.MANUAL_REVIEW),
        (50, "MODERATE", ITCDecision.MANUAL_REVIEW),
        (40, "HIGH_RISK", ITCDecision.BLOCK),
        (30, "HIGH_RISK", ITCDecision.BLOCK),
        (20, "FRAUD", ITCDecision.BLOCK_AND_AUDIT),
        (5, "FRAUD", ITCDecision.BLOCK_AND_AUDIT),
    ]

    all_pass = True
    for score, category, expected in test_cases:
        decision = engine._make_decision(score, category)
        passed = decision == expected
        status = "✅" if passed else "❌"
        print(f"  {status} Score {score:3d} → {decision:20s}")
        if not passed:
            all_pass = False

    return all_pass


def test_explainer():
    print("\n" + "=" * 56)
    print("  TEST 4: Explanation Generator")
    print("=" * 56)

    from risk_scorer.explainer import generate_vendor_explanation

    # Mock safe vendor
    safe_vendor = {
        "gstin": "29AAAAA1234A1Z5",
        "business_name": "Vikram Textiles Pvt Ltd",
        "final_score": 88.5,
        "risk_category": "SAFE",
        "filing_score": 95.0,
        "dispute_score": 85.0,
        "network_score": 90.0,
        "physical_score": 84.0,
        "ml_adjustment": 0.0,
        "base_score": 88.5,
        "gnn_fraud_probability": 0.08,
        "top_risk_factors": ["All compliant"],
        "filing_breakdown": {},
        "dispute_breakdown": {"total_invoices": 50},
    }

    result = generate_vendor_explanation(safe_vendor)
    assert "SAFE" in result["summary"]
    assert result["score"] == 88.5
    print(f"  ✅ Safe vendor explanation generated")
    print(f"     Summary: {result['summary'][:100]}...")

    # Mock fraud vendor
    fraud_vendor = {
        "gstin": "29XYZAB1234C1Z5",
        "business_name": "Ghost Suppliers Ltd",
        "final_score": 8.0,
        "risk_category": "FRAUD",
        "filing_score": 10.0,
        "dispute_score": 5.0,
        "network_score": 12.0,
        "physical_score": 5.0,
        "ml_adjustment": -8.0,
        "base_score": 16.0,
        "gnn_fraud_probability": 0.95,
        "top_risk_factors": [
            "GSTR-3B not filed for 5/6 months",
            "Part of circular trading cycle",
            "80% invoice mismatch rate",
        ],
        "filing_breakdown": {},
        "dispute_breakdown": {"total_invoices": 30},
    }

    result2 = generate_vendor_explanation(fraud_vendor)
    assert "FRAUD" in result2["summary"] or "ALERT" in result2["summary"]
    assert "AUDIT" in result2["recommendation"].upper()
    print(f"  ✅ Fraud vendor explanation generated")
    print(f"     Summary: {result2['summary'][:100]}...")

    return True


def test_full_pipeline():
    print("\n" + "=" * 56)
    print("  TEST 5: Full Pipeline (requires databases)")
    print("=" * 56)

    try:
        from risk_scorer.vendor_risk_engine import VendorRiskEngine

        engine = VendorRiskEngine()
        results = engine.score_all_vendors()

        assert len(results) > 0, "No vendors scored"

        # Check score ranges
        for r in results:
            assert 0 <= r.final_score <= 100, f"Score out of range: {r.final_score}"
            assert r.risk_category in ("SAFE", "MODERATE", "HIGH_RISK", "FRAUD")

        # Find extremes
        safest = max(results, key=lambda r: r.final_score)
        riskiest = min(results, key=lambda r: r.final_score)

        print(f"\n  Vendors scored: {len(results)}")
        print(f"  Safest:  {safest.business_name} → {safest.final_score:.1f} ({safest.risk_category})")
        print(f"  Riskiest: {riskiest.business_name} → {riskiest.final_score:.1f} ({riskiest.risk_category})")

        # Test ITC decisions
        from risk_scorer.itc_decision import ITCDecisionEngine

        scores_dict = [r.to_dict() for r in results]
        itc_engine = ITCDecisionEngine(vendor_scores=scores_dict)
        itc_results = itc_engine.decide_all()

        assert len(itc_results) > 0, "No ITC decisions made"
        print(f"  ITC decisions: {len(itc_results)}")

        engine.close()
        itc_engine.close()

        print("\n  ✅ Full pipeline test PASSED")
        return True

    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Vendor Risk Scorer — Test Suite                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = {}
    results["imports"] = test_imports()

    if results["imports"]:
        results["risk_bands"] = test_risk_band_logic()
        results["itc_logic"] = test_itc_decision_logic()
        results["explainer"] = test_explainer()
        results["full_pipeline"] = test_full_pipeline()

    print("\n" + "=" * 56)
    print("  TEST SUMMARY")
    print("=" * 56)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s} : {status}")

    all_ok = all(results.values())
    print("=" * 56)
    print("  🎉 ALL PASSED" if all_ok else "  ⚠️  Some tests failed")
    print("=" * 56 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())