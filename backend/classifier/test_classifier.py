"""
Standalone test for the hybrid mismatch classifier.

Usage:
    cd backend
    python -m classifier.test_classifier
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

    try:
        import xgboost
        print(f"  [OK]  xgboost         : {xgboost.__version__}")
    except ImportError as e:
        print(f"  [FAIL]  xgboost: {e}")
        return False

    try:
        from classifier.rule_engine import RuleEngine, MismatchType, RiskLevel
        print("  [OK]  rule_engine      : OK")
    except ImportError as e:
        print(f"  [FAIL]  rule_engine: {e}")
        return False

    try:
        from classifier.feature_engineering import FeatureExtractor, get_feature_names
        print("  [OK]  feature_eng      : OK")
    except ImportError as e:
        print(f"  [FAIL]  feature_engineering: {e}")
        return False

    try:
        from classifier.xgboost_model import train_xgboost, predict_xgboost
        print("  [OK]  xgboost_model    : OK")
    except ImportError as e:
        print(f"  [FAIL]  xgboost_model: {e}")
        return False

    try:
        from classifier.hybrid_classifier import HybridClassifier
        print("  [OK]  hybrid_classifier: OK")
    except ImportError as e:
        print(f"  [FAIL]  hybrid_classifier: {e}")
        return False

    try:
        from classifier.evaluator import evaluate_classifier
        print("  [OK]  evaluator        : OK")
    except ImportError as e:
        print(f"  [FAIL]  evaluator: {e}")
        return False

    return True


def test_rule_engine_logic():
    print("\n" + "=" * 56)
    print("  TEST 2: Rule Engine Logic (unit tests)")
    print("=" * 56)

    from classifier.rule_engine import RuleEngine, MismatchType

    # Test risk resolution
    engine = RuleEngine()

    # No mismatches
    level, score = engine._resolve_risk([])
    assert level == "LOW", f"Expected LOW, got {level}"
    assert score < 0.1, f"Expected <0.1, got {score}"
    print("  [OK]  No mismatch -> LOW")

    # Single Type A
    level, score = engine._resolve_risk([MismatchType.TYPE_A_MISSING_GSTR1])
    assert level == "HIGH", f"Expected HIGH, got {level}"
    assert score >= 0.7, f"Expected >=0.7, got {score}"
    print(f"  [OK]  Type A -> HIGH (score={score:.2f})")

    # Type C -> CRITICAL
    level, score = engine._resolve_risk([MismatchType.TYPE_C_FAKE_IRN])
    assert level == "CRITICAL", f"Expected CRITICAL, got {level}"
    assert score >= 0.9, f"Expected >=0.9, got {score}"
    print(f"  [OK]  Type C -> CRITICAL (score={score:.2f})")

    # Multiple mismatches (compounding)
    level, score = engine._resolve_risk([
        MismatchType.TYPE_A_MISSING_GSTR1,
        MismatchType.TYPE_C_FAKE_IRN,
        MismatchType.TYPE_D_NO_EWAY,
    ])
    assert level == "CRITICAL"
    assert score > 0.95
    print(f"  [OK]  Multiple -> CRITICAL (score={score:.4f})")

    print("\n  All rule engine logic tests passed!")
    return True


def test_xgboost_standalone():
    print("\n" + "=" * 56)
    print("  TEST 3: XGBoost Standalone (dummy data)")
    print("=" * 56)

    import numpy as np
    import pandas as pd
    from classifier.feature_engineering import get_feature_names
    from classifier.xgboost_model import train_xgboost, predict_xgboost

    feature_names = get_feature_names()
    n_samples = 200

    np.random.seed(42)

    # Simulate features
    data = {}
    data["invoice_number"] = [f"INV-TEST-{i:04d}" for i in range(n_samples)]
    for fn in feature_names:
        data[fn] = np.random.rand(n_samples).astype(np.float32)

    features_df = pd.DataFrame(data)

    # Simulate labels (30% risky)
    labels = np.zeros(n_samples, dtype=int)
    risky_idx = np.random.choice(n_samples, size=60, replace=False)
    labels[risky_idx] = 1

    labels_df = pd.DataFrame({
        "invoice_number": data["invoice_number"],
        "binary_label": labels,
    })

    # Train
    metrics = train_xgboost(
        features_df,
        labels_df,
        model_path="test_xgb_model.pkl",
        scaler_path="test_xgb_scaler.pkl",
    )

    assert metrics["accuracy"] > 0.5, f"Accuracy too low: {metrics['accuracy']}"
    print(f"\n  Accuracy : {metrics['accuracy']}")
    print(f"  F1       : {metrics['f1']}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")

    # Predict
    preds_df = predict_xgboost(
        features_df,
        model_path="test_xgb_model.pkl",
        scaler_path="test_xgb_scaler.pkl",
    )

    assert len(preds_df) == n_samples
    assert "ml_risk_probability" in preds_df.columns
    print(f"\n  Predictions: {len(preds_df)} invoices")
    print(f"  Flagged high-risk: {int(preds_df['ml_risk_label'].sum())}")

    # Cleanup
    import os
    os.remove("test_xgb_model.pkl")
    os.remove("test_xgb_scaler.pkl")

    print("\n  [OK]  XGBoost standalone test PASSED")
    return True


def test_evaluator():
    print("\n" + "=" * 56)
    print("  TEST 4: Evaluator")
    print("=" * 56)

    import numpy as np
    from classifier.evaluator import evaluate_classifier

    np.random.seed(42)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.6, 0.85, 0.15, 0.95, 0.1])

    metrics = evaluate_classifier(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "recall_analysis" in metrics
    assert "confusion_matrix_details" in metrics

    print("  [OK]  Evaluator test PASSED")
    return True


def test_full_pipeline():
    print("\n" + "=" * 56)
    print("  TEST 5: Full Pipeline (requires databases)")
    print("=" * 56)

    try:
        from classifier.hybrid_classifier import HybridClassifier

        hc = HybridClassifier()

        # Train
        print("\n  Training...")
        metrics = hc.train()
        assert metrics["status"] == "success"
        print(f"  -> {metrics['rule_engine_stats']}")

        # Classify
        print("\n  Classifying...")
        results = hc.classify()
        assert len(results) > 0
        print(f"  -> {len(results)} invoices classified")

        # Show sample
        for r in results[:3]:
            print(
                f"     {r['invoice_number']} | "
                f"{r['mismatch_type']:<20s} | "
                f"{r['risk_level']:<10s} | "
                f"score={r['risk_probability']:.4f}"
            )

        hc.close()
        print("\n  [OK]  Full pipeline test PASSED")
        return True

    except Exception as e:
        print(f"\n  [FAIL]  Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("============================================================")
    print("   GST Mismatch Classifier - Test Suite                 ")
    print("============================================================")

    results = {}
    results["imports"] = test_imports()

    if results["imports"]:
        results["rule_logic"] = test_rule_engine_logic()
        results["xgboost"] = test_xgboost_standalone()
        results["evaluator"] = test_evaluator()
        results["full_pipeline"] = test_full_pipeline()

    print("\n" + "=" * 56)
    print("  TEST SUMMARY")
    print("=" * 56)
    for name, passed in results.items():
        status = "[OK]  PASS" if passed else "[FAIL]  FAIL"
        print(f"  {name:20s} : {status}")

    all_ok = all(results.values())
    print("=" * 56)
    print("   ALL PASSED" if all_ok else "    Some tests failed")
    print("=" * 56 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())