"""
GNN Model Architecture — GST Fraud Detection
===============================================
A 3-layer GraphSAGE network for node-level binary classification
(fraud vs legitimate taxpayer).

Architecture
------------
    SAGEConv(18 → 64) → BN → ReLU → Dropout(0.3)
    SAGEConv(64 → 32) → BN → ReLU → Dropout(0.3)
    SAGEConv(32 → 16) → ReLU
    Linear(16 → 1)   → Sigmoid

Why GraphSAGE:
    • Inductive — handles new taxpayers without retraining the full graph.
    • Sample-and-aggregate — scalable to the national GST network.
    • Neighbourhood-aware — fraud signal propagates from risky suppliers.

Why 3 layers:
    • 1-hop: direct supplier/customer behaviour
    • 2-hop: supplier's supplier (supply-chain awareness)
    • 3-hop: extended fraud-ring detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, BatchNorm
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install with:\n"
        "  pip install torch-geometric torch-scatter torch-sparse"
    )


class GSTFraudGNN(nn.Module):
    """
    Graph Neural Network for GST taxpayer fraud classification.

    Parameters
    ----------
    input_dim : int
        Number of node features (default 18).
    hidden1, hidden2, hidden3 : int
        Hidden layer dimensions.
    dropout : float
        Dropout probability applied after layers 1 and 2.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden1: int = 64,
        hidden2: int = 32,
        hidden3: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        # GraphSAGE convolution layers
        self.conv1 = SAGEConv(input_dim, hidden1)
        self.bn1 = BatchNorm(hidden1)

        self.conv2 = SAGEConv(hidden1, hidden2)
        self.bn2 = BatchNorm(hidden2)

        self.conv3 = SAGEConv(hidden2, hidden3)

        # Classification head
        self.classifier = nn.Linear(hidden3, 1)

        self.dropout = dropout
        self._input_dim = input_dim

    # ────────────────────────────────
    # Forward pass
    # ────────────────────────────────
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute fraud probability for every node.

        Parameters
        ----------
        x : Tensor [N, input_dim]
            Node feature matrix.
        edge_index : Tensor [2, E]
            Edge index.

        Returns
        -------
        Tensor [N]
            Fraud probability ∈ [0, 1] per node.
        """
        h = self._encode(x, edge_index)
        logits = self.classifier(h).squeeze(-1)
        return torch.sigmoid(logits)

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return raw logits (before sigmoid) — useful for BCEWithLogitsLoss."""
        h = self._encode(x, edge_index)
        return self.classifier(h).squeeze(-1)

    # ────────────────────────────────
    # Encoder (shared by forward + get_embeddings)
    # ────────────────────────────────
    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return 16-D node embeddings from the penultimate layer."""
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = F.relu(h)

        return h

    # ────────────────────────────────
    # Embedding / explainability helpers
    # ────────────────────────────────
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Return 16-D node embeddings (penultimate layer output).
        Useful for t-SNE visualisation, clustering, and explainability.
        """
        self.eval()
        with torch.no_grad():
            return self._encode(x, edge_index)

    def get_layer_outputs(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """
        Return intermediate representations from every layer.
        Useful for per-layer analysis / debugging.
        """
        self.eval()
        with torch.no_grad():
            h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
            h3 = F.relu(self.conv3(h2, edge_index))
            out = torch.sigmoid(self.classifier(h3).squeeze(-1))
        return {"layer1": h1, "layer2": h2, "layer3": h3, "output": out}


# ──────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────
def create_model(input_dim: int = 18, **kwargs) -> GSTFraudGNN:
    """Instantiate a GSTFraudGNN with sensible defaults."""
    return GSTFraudGNN(input_dim=input_dim, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: GSTFraudGNN) -> str:
    """Return a human-readable architecture summary string."""
    lines = [
        "GSTFraudGNN Architecture",
        "─" * 40,
        f"  SAGEConv({model._input_dim} → 64) → BN → ReLU → Dropout({model.dropout})",
        f"  SAGEConv(64 → 32) → BN → ReLU → Dropout({model.dropout})",
        "  SAGEConv(32 → 16) → ReLU",
        "  Linear(16 → 1) → Sigmoid",
        "─" * 40,
        f"  Trainable parameters: {count_parameters(model):,}",
    ]
    return "\n".join(lines)