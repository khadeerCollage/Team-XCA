"""
GNN Training Pipeline
======================
Handles end-to-end model training with:
    • Class-weighted BCE loss (handles fraud / legit imbalance)
    • Adam optimiser with weight decay
    • Early stopping on validation F1
    • Optimal threshold search
    • Model checkpointing
"""

import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preparation import (
    get_feature_names,
    get_gstin_to_idx,
    get_idx_to_gstin,
    prepare_pyg_data,
)
from .gnn_model import GSTFraudGNN, count_parameters, create_model, model_summary

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("torch_geometric required")

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "model_checkpoint.pt")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def _set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────
def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Return a per-sample weight tensor to counter class imbalance.
    Fraud samples get weight = num_legit / num_fraud.
    """
    num_pos = max(int(labels.sum().item()), 1)
    num_neg = max(len(labels) - num_pos, 1)
    pos_weight = num_neg / num_pos

    weights = torch.where(
        labels == 1.0,
        torch.tensor(pos_weight, dtype=torch.float),
        torch.tensor(1.0, dtype=torch.float),
    )
    logger.info("Class weights — pos_weight: %.2f  (pos=%d, neg=%d)", pos_weight, num_pos, num_neg)
    return weights


# ──────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────
def train_epoch(
    model: GSTFraudGNN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    class_weights: torch.Tensor,
) -> float:
    """Run one training epoch.  Returns mean weighted loss."""
    model.train()
    optimizer.zero_grad()

    probs = model(data.x, data.edge_index)
    per_sample_loss = criterion(probs[data.train_mask], data.y[data.train_mask])
    weighted = per_sample_loss * class_weights[data.train_mask]
    loss = weighted.mean()

    loss.backward()
    optimizer.step()
    return loss.item()


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: GSTFraudGNN,
    data: Data,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate model on the given mask.

    Returns dict with keys:
        loss, accuracy, precision, recall, f1, roc_auc,
        probabilities, predictions
    """
    model.eval()
    probs = model(data.x, data.edge_index)

    mask_probs = probs[mask].cpu().numpy()
    mask_labels = data.y[mask].cpu().numpy()

    preds = (mask_probs >= threshold).astype(int)

    # Loss (unweighted)
    bce = nn.BCELoss(reduction="mean")
    loss_val = bce(probs[mask], data.y[mask]).item()

    acc = accuracy_score(mask_labels, preds)
    prec = precision_score(mask_labels, preds, zero_division=0)
    rec = recall_score(mask_labels, preds, zero_division=0)
    f1 = f1_score(mask_labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(mask_labels, mask_probs)
    except ValueError:
        auc = 0.0

    return {
        "loss": loss_val,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "probabilities": mask_probs.tolist(),
        "predictions": preds.tolist(),
    }


# ──────────────────────────────────────────────
# Optimal threshold search
# ──────────────────────────────────────────────
@torch.no_grad()
def find_optimal_threshold(model: GSTFraudGNN, data: Data, mask: torch.Tensor) -> float:
    """Try thresholds 0.25–0.65 and return the one maximising F1 on *mask*."""
    model.eval()
    probs = model(data.x, data.edge_index)[mask].cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.66, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    logger.info("Optimal threshold: %.2f (F1=%.4f)", best_t, best_f1)
    return round(float(best_t), 2)


# ──────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────
def save_model(
    model: GSTFraudGNN,
    threshold: float,
    feature_names: List[str],
    path: str = DEFAULT_MODEL_PATH,
) -> None:
    """Persist model weights + metadata."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "feature_names": feature_names,
            "input_dim": model._input_dim,
        },
        path,
    )
    logger.info("Model saved to %s", path)


def load_model(path: str = DEFAULT_MODEL_PATH) -> Tuple[GSTFraudGNN, float, List[str]]:
    """Load model from checkpoint. Returns (model, threshold, feature_names)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model checkpoint at {path}. Train first via POST /api/ml/train"
        )
    ckpt = torch.load(path, map_location=DEVICE)
    model = create_model(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt["threshold"], ckpt["feature_names"]


# ──────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────
def train_model(
    data: Data,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 30,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict:
    """
    Complete training pipeline.

    Returns
    -------
    dict
        train_metrics, val_metrics, test_metrics,
        optimal_threshold, training_history, model_path
    """
    _set_seeds(42)

    data = data.to(DEVICE)

    model = create_model(input_dim=data.num_node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCELoss(reduction="none")  # per-sample — we weight manually
    class_weights = compute_class_weights(data.y).to(DEVICE)

    feature_names = get_feature_names()

    print("\n" + "=" * 56)
    print("  GST Fraud Detection GNN — Training")
    print("=" * 56)
    print(f"  Nodes:   {data.num_nodes}  |  Edges: {data.num_edges}  |  Features: {data.num_node_features}")
    fraud_n = int(data.y.sum().item())
    legit_n = data.num_nodes - fraud_n
    print(f"  Labels:  {fraud_n} fraud ({100*fraud_n/data.num_nodes:.0f}%)  |  "
          f"{legit_n} legit ({100*legit_n/data.num_nodes:.0f}%)")
    print(f"  Train: {int(data.train_mask.sum())}  |  Val: {int(data.val_mask.sum())}  |  "
          f"Test: {int(data.test_mask.sum())}")
    print(f"  Device: {DEVICE}")
    print()
    print(model_summary(model))
    print("-" * 56)

    history: List[dict] = []
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion, class_weights)

        if epoch % 10 == 0 or epoch == 1:
            val_m = evaluate(model, data, data.val_mask)
            entry = {
                "epoch": epoch,
                "train_loss": round(loss, 4),
                "val_loss": round(val_m["loss"], 4),
                "val_f1": val_m["f1"],
                "val_recall": val_m["recall"],
            }
            history.append(entry)

            print(
                f"  Epoch {epoch:03d} | Train Loss: {loss:.4f} | "
                f"Val Loss: {val_m['loss']:.4f} | Val F1: {val_m['f1']:.4f} | "
                f"Val Recall: {val_m['recall']:.4f}"
            )

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_epoch = epoch
                no_improve = 0
                # Save best weights temporarily
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 10  # we evaluate every 10 epochs

            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience {patience})")
                break

    elapsed = time.time() - t0

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Optimal threshold
    opt_threshold = find_optimal_threshold(model, data, data.val_mask)

    # Final evaluation
    train_m = evaluate(model, data, data.train_mask, opt_threshold)
    val_m = evaluate(model, data, data.val_mask, opt_threshold)
    test_m = evaluate(model, data, data.test_mask, opt_threshold)

    print("-" * 56)
    print(f"  Optimal Threshold: {opt_threshold}")
    print("-" * 56)
    print("  TEST RESULTS:")
    print(f"    Accuracy  : {test_m['accuracy']}")
    print(f"    Precision : {test_m['precision']}")
    print(f"    Recall    : {test_m['recall']}")
    print(f"    F1 Score  : {test_m['f1']}")
    print(f"    ROC-AUC   : {test_m['roc_auc']}")
    print("=" * 56)
    print(f"  Training time: {elapsed:.1f}s")

    # Save
    save_model(model, opt_threshold, feature_names, model_path)
    print(f"  Model saved to: {model_path}\n")

    return {
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "optimal_threshold": opt_threshold,
        "training_history": history,
        "model_path": model_path,
        "best_epoch": best_epoch,
        "training_time_seconds": round(elapsed, 2),
        "model_parameters": count_parameters(model),
    }


# ──────────────────────────────────────────────
# End-to-end (called by API)
# ──────────────────────────────────────────────
def train_and_evaluate(driver, session) -> Dict:
    """
    One-call entry point:
        1. Build PyG Data from graph + SQL
        2. Train GNN
        3. Return results dict
    """
    data = prepare_pyg_data(driver, session)
    results = train_model(data)
    return results