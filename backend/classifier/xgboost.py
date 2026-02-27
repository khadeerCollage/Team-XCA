"""
XGBoost Risk Classifier
=========================
Supervised classifier that learns hidden fraud / mismatch patterns
from invoice-level features.

Why XGBoost:
    • Gradient boosting excels on tabular / structured data
    • Built-in handling of missing values
    • Feature importance for explainability
    • Regularisation (L1/L2) prevents overfitting on small datasets
    • Fast training — suitable for hackathon demo
    • Handles class imbalance via scale_pos_weight
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost is required. Install with: pip install xgboost")

from .feature_engineering import FEATURE_NAMES, FeatureExtractor, get_feature_names

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_checkpoint.pkl")
DEFAULT_SCALER_PATH = os.path.join(MODEL_DIR, "xgboost_scaler.pkl")

# ──────────────────────────────────────────────
# Label generation (for hackathon synthetic data)
# ──────────────────────────────────────────────
def generate_labels_from_rules(rule_results: List[dict]) -> pd.DataFrame:
    """
    Convert rule engine outputs into supervised training labels.

    Label scheme:
        0 = LOW risk    (no mismatches or minor ones)
        1 = HIGH risk   (any mismatch detected)

    For multi-class extension:
        0 = No mismatch
        1 = Missing GSTR-1
        2 = Amount Mismatch
        3 = Fake IRN
        4 = No e-Way Bill

    We train a **binary classifier** (risk / no risk) because it is
    more robust with limited hackathon data.
    """
    rows = []
    for r in rule_results:
        inv = r["invoice_number"]
        has_mismatch = 1 if len(r.get("mismatch_types", [])) > 0 else 0

        # Multi-class label (highest priority mismatch)
        if "Fake/Missing IRN" in r.get("mismatch_types", []):
            multi_label = 3
        elif "Missing GSTR-1" in r.get("mismatch_types", []):
            multi_label = 1
        elif "No e-Way Bill" in r.get("mismatch_types", []):
            multi_label = 4
        elif "Amount Mismatch" in r.get("mismatch_types", []):
            multi_label = 2
        else:
            multi_label = 0

        rows.append({
            "invoice_number": inv,
            "binary_label": has_mismatch,
            "multi_label": multi_label,
            "rule_risk_score": r.get("rule_risk_score", 0.0),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_xgboost(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    test_size: float = 0.2,
) -> Dict:
    """
    Train an XGBoost binary classifier on invoice features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Columns: ['invoice_number'] + FEATURE_NAMES
    labels_df : pd.DataFrame
        Columns: ['invoice_number', 'binary_label', ...]
    model_path : str
        Where to save the trained model.
    scaler_path : str
        Where to save the fitted scaler.
    test_size : float
        Fraction held out for testing.

    Returns
    -------
    dict
        Training results with metrics.
    """
    np.random.seed(42)

    # Merge
    merged = features_df.merge(labels_df, on="invoice_number", how="inner")
    if merged.empty:
        raise ValueError("No matching invoices between features and labels")

    feature_cols = get_feature_names()
    X = merged[feature_cols].values.astype(np.float32)
    y = merged["binary_label"].values.astype(int)

    # Handle NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split (stratified)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        logger.warning("Stratification failed; falling back to random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

    # Class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    print("\n" + "=" * 56)
    print("  XGBoost Mismatch Classifier — Training")
    print("=" * 56)
    print(f"  Total invoices : {len(y)}")
    print(f"  Train / Test   : {len(y_train)} / {len(y_test)}")
    print(f"  Positive (risk): {n_pos} ({100*n_pos/max(len(y_train),1):.1f}%)")
    print(f"  Negative (ok)  : {n_neg} ({100*n_neg/max(len(y_train),1):.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    print("-" * 56)

    # Model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,        # L1 regularisation
        reg_lambda=1.0,       # L2 regularisation
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Evaluate
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    print("\n  Top-5 Feature Importances:")
    for fname, imp in feat_imp[:5]:
        print(f"    {fname:35s} : {imp:.4f}")

    print("=" * 56)

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  Model saved  : {model_path}")
    print(f"  Scaler saved : {scaler_path}\n")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "feature_importance": [
            {"feature": fn, "importance": round(float(imp), 4)}
            for fn, imp in feat_imp
        ],
        "train_size": len(y_train),
        "test_size": len(y_test),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "model_path": model_path,
    }


# ──────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────
def load_xgboost(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> Tuple[xgb.XGBClassifier, StandardScaler]:
    """Load saved XGBoost model and scaler."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No XGBoost checkpoint at {model_path}. "
            "Train first via POST /api/classifier/train"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_xgboost(
    features_df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> pd.DataFrame:
    """
    Run XGBoost inference on invoice features.

    Returns DataFrame with columns:
        invoice_number, ml_risk_probability, ml_risk_label
    """
    model, scaler = load_xgboost(model_path, scaler_path)

    feature_cols = get_feature_names()
    X = features_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    result = pd.DataFrame({
        "invoice_number": features_df["invoice_number"].values,
        "ml_risk_probability": probabilities.round(4),
        "ml_risk_label": predictions,
    })

    logger.info(
        "XGBoost predictions — %d invoices, %d flagged high-risk",
        len(result),
        int(predictions.sum()),
    )
    return result