"""
PyTorch Geometric Data Preparation
====================================
Converts the Neo4j knowledge graph + extracted features into a
``torch_geometric.data.Data`` object for GNN training / inference.

Pipeline:
    1. Extract taxpayer nodes from Neo4j
    2. Build node ↔ integer-index mappings
    3. Extract edges, make bidirectional
    4. Build normalised feature matrix (18-D)
    5. Generate fraud labels (simulated for hackathon)
    6. Create stratified train / val / test masks
    7. Assemble PyG Data object
"""

import logging
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "torch_geometric is required.  Install with:\n"
        "  pip install torch-geometric torch-scatter torch-sparse"
    )

from .graph_features import (
    FEATURE_NAMES,
    build_networkx_graph,
    extract_all_features,
    get_feature_names,
    get_neo4j_driver,
    get_sql_session,
    precompute_graph_metrics,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Module-level caches (re-used by predict.py)
# ──────────────────────────────────────────────
_gstin_to_idx: Dict[str, int] = {}
_idx_to_gstin: Dict[int, str] = {}
_feature_scaler: StandardScaler = None
_node_properties: Dict[str, dict] = {}

EXPECTED_FILING_PERIODS = 6


# ──────────────────────────────────────────────
# Public accessors for cached mappings
# ──────────────────────────────────────────────
def get_gstin_to_idx() -> Dict[str, int]:
    """Return the current gstin → index mapping."""
    return dict(_gstin_to_idx)


def get_idx_to_gstin() -> Dict[int, str]:
    """Return the current index → gstin mapping."""
    return dict(_idx_to_gstin)


def get_feature_scaler() -> StandardScaler:
    """Return the fitted StandardScaler (or None if not yet fitted)."""
    return _feature_scaler


def get_node_properties() -> Dict[str, dict]:
    """Return cached node property dicts keyed by gstin."""
    return dict(_node_properties)


# ──────────────────────────────────────────────
# Step 1 — Extract taxpayer nodes
# ──────────────────────────────────────────────
def extract_taxpayer_nodes(driver) -> List[dict]:
    """
    Query all Taxpayer nodes from Neo4j and return their properties.
    """
    nodes = []
    with driver.session() as session:
        result = session.run(
            """
            MATCH (t:Taxpayer)
            RETURN t.gstin              AS gstin,
                   t.name               AS name,
                   t.compliance_rating  AS rating,
                   t.business_type      AS btype,
                   t.is_active          AS active,
                   t.registration_date  AS reg_date,
                   t.filing_frequency   AS freq
            ORDER BY t.gstin
            """
        )
        for rec in result:
            nodes.append({
                "gstin": rec["gstin"],
                "name": rec["name"] or "Unknown",
                "compliance_rating": rec["rating"] or "C",
                "business_type": rec["btype"] or "Unknown",
                "is_active": rec["active"] if rec["active"] is not None else True,
                "registration_date": str(rec["reg_date"]) if rec["reg_date"] else "",
                "filing_frequency": rec["freq"] or "Monthly",
            })
    logger.info("Extracted %d taxpayer nodes from Neo4j", len(nodes))
    return nodes


# ──────────────────────────────────────────────
# Step 2 — Build index mappings
# ──────────────────────────────────────────────
def build_node_mapping(nodes: List[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create gstin ↔ integer index bi-directional mappings.
    Also caches node properties in module-level ``_node_properties``.
    """
    global _gstin_to_idx, _idx_to_gstin, _node_properties
    _gstin_to_idx = {}
    _idx_to_gstin = {}
    _node_properties = {}

    for idx, node in enumerate(nodes):
        gstin = node["gstin"]
        _gstin_to_idx[gstin] = idx
        _idx_to_gstin[idx] = gstin
        _node_properties[gstin] = node

    logger.info("Node mapping built — %d nodes indexed", len(_gstin_to_idx))
    return dict(_gstin_to_idx), dict(_idx_to_gstin)


# ──────────────────────────────────────────────
# Step 3 — Extract edges
# ──────────────────────────────────────────────
def extract_edges(driver) -> List[Tuple[str, str]]:
    """
    Extract all SUPPLIES_TO and CLAIMED_ITC_FROM edges.
    De-duplicate edge pairs.
    """
    edge_set = set()
    with driver.session() as session:
        for rel_type in ("SUPPLIES_TO", "CLAIMED_ITC_FROM"):
            result = session.run(
                f"""
                MATCH (a:Taxpayer)-[:{rel_type}]->(b:Taxpayer)
                RETURN a.gstin AS source, b.gstin AS target
                """
            )
            for rec in result:
                edge_set.add((rec["source"], rec["target"]))

    edges = sorted(edge_set)
    logger.info("Extracted %d unique directed edges", len(edges))
    return edges


# ──────────────────────────────────────────────
# Step 4 — Build edge_index tensor
# ──────────────────────────────────────────────
def build_edge_index(
    edges: List[Tuple[str, str]], gstin_to_idx: Dict[str, int]
) -> torch.LongTensor:
    """
    Convert (source_gstin, target_gstin) pairs into a PyG edge_index
    tensor of shape ``[2, 2*E]`` (bidirectional).
    """
    sources, targets = [], []
    for src, tgt in edges:
        if src in gstin_to_idx and tgt in gstin_to_idx:
            s_idx = gstin_to_idx[src]
            t_idx = gstin_to_idx[tgt]
            # forward
            sources.append(s_idx)
            targets.append(t_idx)
            # reverse (make undirected for message passing)
            sources.append(t_idx)
            targets.append(s_idx)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    logger.info("edge_index shape: %s", list(edge_index.shape))
    return edge_index


# ──────────────────────────────────────────────
# Step 5 — Build node feature matrix
# ──────────────────────────────────────────────
def build_feature_matrix(
    driver, session, gstin_to_idx: Dict[str, int]
) -> torch.FloatTensor:
    """
    Extract 18-D feature vectors for all nodes, normalise with
    StandardScaler, and return as a FloatTensor of shape ``[N, 18]``.
    """
    global _feature_scaler

    features_df = extract_all_features(driver, session)

    # Ensure ordering matches gstin_to_idx
    ordered_gstins = [g for g, _ in sorted(gstin_to_idx.items(), key=lambda x: x[1])]
    features_df = features_df.set_index("gstin").reindex(ordered_gstins)

    # Fill any missing nodes with zeros
    features_df = features_df.fillna(0.0)

    feature_cols = get_feature_names()
    X = features_df[feature_cols].values.astype(np.float32)

    # Normalise
    _feature_scaler = StandardScaler()
    X_scaled = _feature_scaler.fit_transform(X)

    tensor = torch.tensor(X_scaled, dtype=torch.float)
    logger.info("Feature matrix shape: %s", list(tensor.shape))
    return tensor


# ──────────────────────────────────────────────
# Step 6 — Generate labels
# ──────────────────────────────────────────────
def generate_labels(
    nodes: List[dict],
    driver,
    session,
    gstin_to_idx: Dict[str, int],
) -> torch.FloatTensor:
    """
    Simulate ground-truth fraud labels for hackathon data.

    A node is labelled FRAUD (1.0) if ANY of these hold:
        a) compliance_rating == 'D'
        b) is_active == False AND has at least one GSTR-1 filing
        c) participates in a graph cycle (circular trading)
        d) itc_overclaim_ratio > 0.30
        e) gstr1_filing_rate > 0 AND gstr3b_filing_rate == 0
        f) irn_coverage < 0.30 AND transaction_volume > median
    """
    from sqlalchemy import text as sa_text

    num_nodes = len(nodes)
    labels = np.zeros(num_nodes, dtype=np.float32)

    # Pre-fetch per-node features that we already have
    features_df = extract_all_features(driver, session).set_index("gstin")

    # Detect cycles once
    G = build_networkx_graph(driver)
    import networkx as nx

    nodes_in_cycles: set = set()
    try:
        for cycle in nx.simple_cycles(G, length_bound=5):
            for n in cycle:
                nodes_in_cycles.add(n)
    except Exception:
        pass

    # Median transaction volume for condition (f)
    if "transaction_volume" in features_df.columns:
        median_vol = features_df["transaction_volume"].median()
    else:
        median_vol = 0.0

    for node in nodes:
        gstin = node["gstin"]
        idx = gstin_to_idx.get(gstin)
        if idx is None:
            continue

        feat = features_df.loc[gstin] if gstin in features_df.index else pd.Series(dtype=float)

        is_fraud = False

        # (a) compliance_rating == 'D'
        if node.get("compliance_rating") == "D":
            is_fraud = True

        # (b) inactive but has GSTR-1 filings
        if not node.get("is_active", True):
            gstr1_rate = feat.get("gstr1_filing_rate", 0)
            if gstr1_rate > 0:
                is_fraud = True

        # (c) cycle participation
        if gstin in nodes_in_cycles:
            is_fraud = True

        # (d) ITC overclaim > 30%
        overclaim = feat.get("itc_overclaim_ratio", 0)
        if overclaim > 0.30:
            is_fraud = True

        # (e) files GSTR-1 but never GSTR-3B
        g1 = feat.get("gstr1_filing_rate", 0)
        g3b = feat.get("gstr3b_filing_rate", 0)
        if g1 > 0 and g3b == 0:
            is_fraud = True

        # (f) low IRN coverage + high volume
        irn_cov = feat.get("irn_coverage", 1.0)
        vol = feat.get("transaction_volume", 0)
        if irn_cov < 0.3 and vol > median_vol:
            is_fraud = True

        labels[idx] = 1.0 if is_fraud else 0.0

    pos = int(labels.sum())
    neg = num_nodes - pos
    logger.info("Labels generated — FRAUD: %d (%.1f%%) | LEGIT: %d (%.1f%%)",
                pos, 100 * pos / max(num_nodes, 1), neg, 100 * neg / max(num_nodes, 1))

    return torch.tensor(labels, dtype=torch.float)


# ──────────────────────────────────────────────
# Step 7 — Train / val / test masks
# ──────────────────────────────────────────────
def create_masks(
    num_nodes: int, labels: torch.Tensor
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """
    Stratified 60 / 20 / 20 split.
    Falls back to random split if stratification is impossible.
    """
    indices = np.arange(num_nodes)
    y = labels.numpy().astype(int)

    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=42, stratify=y
        )
        y_temp = y[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        # Too few samples for stratification — fall back
        logger.warning("Stratification failed; using random split.")
        np.random.seed(42)
        np.random.shuffle(indices)
        n_train = int(0.6 * num_nodes)
        n_val = int(0.2 * num_nodes)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    logger.info("Masks — train: %d | val: %d | test: %d",
                int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
    return train_mask, val_mask, test_mask


# ──────────────────────────────────────────────
# Master orchestrator
# ──────────────────────────────────────────────
def prepare_pyg_data(driver, session) -> Data:
    """
    End-to-end data preparation pipeline.

    Returns a ``torch_geometric.data.Data`` object ready for GNN
    training with attributes:  x, edge_index, y, train_mask,
    val_mask, test_mask.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Step 1
    nodes = extract_taxpayer_nodes(driver)
    if len(nodes) < 3:
        raise ValueError(
            f"Only {len(nodes)} nodes in graph. Need ≥ 3 for GNN training. "
            "Run data ingestion first."
        )

    # Step 2
    gstin_to_idx, idx_to_gstin = build_node_mapping(nodes)

    # Step 3
    edges = extract_edges(driver)

    # Step 4
    edge_index = build_edge_index(edges, gstin_to_idx)

    # Step 5
    x = build_feature_matrix(driver, session, gstin_to_idx)

    # Step 6
    y = generate_labels(nodes, driver, session, gstin_to_idx)

    # Step 7
    train_mask, val_mask, test_mask = create_masks(len(nodes), y)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print("\n" + "=" * 50)
    print("  PyG Data Object — Summary")
    print("=" * 50)
    print(f"  Nodes          : {data.num_nodes}")
    print(f"  Edges          : {data.num_edges} (bidirectional)")
    print(f"  Features       : {data.num_node_features}")
    print(f"  Fraud nodes    : {int(y.sum())} / {len(y)}  "
          f"({100 * y.sum().item() / len(y):.1f}%)")
    print(f"  Train / Val / Test : "
          f"{int(train_mask.sum())} / {int(val_mask.sum())} / {int(test_mask.sum())}")
    print("=" * 50 + "\n")

    return data