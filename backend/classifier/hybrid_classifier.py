"""
Hybrid Mismatch Classifier
==========================
Combines the Rule Engine and the XGBoost ML model to detect
anomalies and determine risk levels for invoices.
"""

import pandas as pd
import numpy as np

from .rule_engine import RuleEngine, MismatchType, RiskLevel
from .feature_engineering import FeatureExtractor
from .xgboost_model import train_xgboost, predict_xgboost
from .evaluator import evaluate_classifier

class HybridClassifier:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.feature_extractor = FeatureExtractor()

    def _get_mock_data(self, n_samples=200):
        """Generate mock data for the hackathon baseline."""
        np.random.seed(42)
        feature_names = self.feature_extractor._feature_names

        data = {"invoice_number": [f"INV-TEST-{i:04d}" for i in range(n_samples)]}
        for fn in feature_names:
            data[fn] = np.random.rand(n_samples).astype(np.float32)

        features_df = pd.DataFrame(data)

        # 30% anomaly
        labels = np.zeros(n_samples, dtype=int)
        risky_idx = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
        labels[risky_idx] = 1

        labels_df = pd.DataFrame({
            "invoice_number": data["invoice_number"],
            "binary_label": labels,
        })
        return features_df, labels_df

    def train(self):
        """Train the underlying ML component of the hybrid classifier."""
        features_df, labels_df = self._get_mock_data()
        
        xgb_metrics = train_xgboost(
            features_df,
            labels_df,
            model_path="xgb_model.pkl",
            scaler_path="xgb_scaler.pkl",
        )

        return {
            "status": "success",
            "rule_engine_stats": "Rule Engine requires no training.",
            "xgboost_metrics": xgb_metrics,
        }

    def classify(self):
        """Classify invoices using the hybrid approach."""
        features_df, _ = self._get_mock_data(n_samples=50)

        # ML Prediction
        preds_df = predict_xgboost(
            features_df,
            model_path="xgb_model.pkl",
            scaler_path="xgb_scaler.pkl"
        )
        
        results = []
        for _, row in preds_df.iterrows():
            mismatches = []
            
            # Simulated Rule Checks
            if row["filing_delay_days"] > 0.8:
                mismatches.append(MismatchType.TYPE_A_MISSING_GSTR1)
            if row["itc_ratio"] > 0.9:
                mismatches.append(MismatchType.TYPE_B_ITC_OVERCLAIM)
            if row["supplier_pagerank"] > 0.85:
                mismatches.append(MismatchType.TYPE_E_CIRCULAR)

            # Rule Output
            rule_level, rule_score = self.rule_engine.evaluate(mismatches)
            
            # ML Output
            ml_prob = row["ml_risk_probability"]
            
            # Hybrid Logic
            final_prob = max(rule_score, ml_prob)
            
            if final_prob >= 0.90:
                final_level = "CRITICAL"
            elif final_prob >= 0.70:
                final_level = "HIGH"
            elif final_prob >= 0.40:
                final_level = "MEDIUM"
            else:
                final_level = "LOW"
            
            results.append({
                "invoice_number": row["invoice_number"],
                "mismatch_type": ", ".join([m.value for m in mismatches]) if mismatches else "None",
                "risk_level": final_level,
                "risk_probability": float(final_prob),
                "rule_score": float(rule_score),
                "ml_probability": float(ml_prob)
            })
            
        return results
    
    def close(self):
        pass