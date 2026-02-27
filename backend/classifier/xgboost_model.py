"""
XGBoost Machine Learning Component
==================================
Trains and predicts on extracted features to capture non-linear
patterns and weak signals that deterministic rules miss.
"""

import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_xgboost(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_path: str = "xgb_model.pkl",
    scaler_path: str = "xgb_scaler.pkl",
):
    """Train the XGBoost classifier and save artifacts."""
    feature_cols = [c for c in features_df.columns if c != "invoice_number"]

    X = features_df[feature_cols].copy()
    y = labels_df["binary_label"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_scaled, y)

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Metrics
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
    }


def predict_xgboost(
    features_df: pd.DataFrame,
    model_path: str = "xgb_model.pkl",
    scaler_path: str = "xgb_scaler.pkl",
) -> pd.DataFrame:
    """Predict probabilities for a batch of invoices."""
    feature_cols = [c for c in features_df.columns if c != "invoice_number"]
    X = features_df[feature_cols].copy()

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Fallback if no model exists (e.g. before initial training)
        features_df["ml_risk_probability"] = 0.5
        features_df["ml_risk_label"] = 0
        return features_df

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    result_df = features_df.copy()
    result_df["ml_risk_probability"] = probs
    result_df["ml_risk_label"] = preds

    return result_df
