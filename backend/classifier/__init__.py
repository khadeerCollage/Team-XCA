"""
GST Mismatch Classifier Layer
================================
Hybrid Rule-Engine + XGBoost classification system for
GST invoice discrepancy detection and risk scoring.

Modules:
    - rule_engine: Deterministic rule-based mismatch classification
    - feature_engineering: Invoice-level feature extraction from Neo4j + PostgreSQL
    - xgboost_model: XGBoost classifier for risk probability scoring
    - hybrid_classifier: Combined rule + ML decision logic
    - evaluator: Precision, Recall, F1, ROC-AUC evaluation
"""

from .rule_engine import RuleEngine, apply_rules
from .hybrid_classifier import HybridClassifier, classify_all_invoices
from .xgboost_model import train_xgboost, predict_xgboost
from .evaluator import evaluate_classifier

__all__ = [
    "RuleEngine",
    "apply_rules",
    "HybridClassifier",
    "classify_all_invoices",
    "train_xgboost",
    "predict_xgboost",
    "evaluate_classifier",
]