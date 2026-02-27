"""
Model Evaluator Component
=========================
Analyses predictions vs ground-truth and outputs standard metrics.
"""

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

def evaluate_classifier(y_true, y_pred, y_prob):
    """Calculate evaluation metrics for the given arrays."""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "recall_analysis": float(rec),
        "precision": float(prec),
        "confusion_matrix_details": cm
    }