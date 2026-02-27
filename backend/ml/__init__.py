"""
GST Fraud Detection — ML Layer
===============================
Graph Neural Network based fraud detection for GST taxpayer network.

Modules:
    - graph_features: Extract structural, behavioral, and network features from Neo4j/PostgreSQL
    - data_preparation: Build PyTorch Geometric Data objects for GNN training
    - gnn_model: GraphSAGE-based GNN architecture
    - train: Training pipeline with class-weighted loss and early stopping
    - predict: Inference pipeline producing fraud probabilities
    - explainer: Gradient-based explainability and natural language audit reports
"""

from .gnn_model import GSTFraudGNN, create_model
from .train import train_and_evaluate, train_model
from .predict import predict_all, predict_single
from .explainer import generate_explanation

__all__ = [
    "GSTFraudGNN",
    "create_model",
    "train_and_evaluate",
    "train_model",
    "predict_all",
    "predict_single",
    "generate_explanation",
]