"""
Graph Feature Extraction Engine
================================
Extracts 18 numerical features per Taxpayer node from:
    - Neo4j Knowledge Graph (structural + network features)
    - PostgreSQL database (behavioral / compliance features)

Features are used as the node feature matrix (X) for the GNN.
"""

import logging
import math
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration (adjust to match your config.py)
# ──────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"

EXPECTED_FILING_PERIODS = 6  # 6 months of data

BUSINESS_TYPE_ENCODING = {
    "Manufacturer": 0.0,
    "Service Provider": 0.33,
    "Trader": 0.66,
}
BUSINESS_TYPE_DEFAULT = 1.0

COMPLIANCE_RATING_ENCODING = {
    "A": 0.0,
    "B": 0.33,
    "C": 0.66,
    "D": 1.0,
}
COMPLIANCE_RATING_DEFAULT = 0.5

FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "pagerank",
    "betweenness_centrality",
    "clustering_coefficient",
    "connected_component_size",
    "gstr1_filing_rate",
    "gstr3b_filing_rate",
    "filing_gap",
    "itc_overclaim_ratio",
    "irn_coverage",
    "eway_coverage",
    "transaction_volume",
    "transaction_frequency",
    "avg_neighbor_risk",
    "risky_neighbor_ratio",
    "cycle_participation",
    "business_type_encoded",
]


def get_feature_names() -> list:
    """Return the ordered list of 18 feature names."""
    return list(FEATURE_NAMES)


# ──────────────────────────────────────────────
# Connection helpers
# ──────────────────────────────────────────────
def get_neo4j_driver():
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_sql_engine():
    """Create and return a SQLAlchemy engine."""
    return create_engine(POSTGRES_URL)


def get_sql_session():
    """Create and return a new SQLAlchemy session."""
    engine = get_sql_engine()
    Session = sessionmaker(bind=engine)
    return Session()


# ──────────────────────────────────────────────
# Build NetworkX graph from Neo4j
# ──────────────────────────────────────────────
def build_networkx_graph(driver) -> nx.DiGraph:
    """
    Extract the full Taxpayer→SUPPLIES_TO→Taxpayer graph from Neo4j
    and build a NetworkX DiGraph.

    Node attributes stored: name, compliance_rating, business_type, is_active
    Edge attributes stored: total_value, transaction_count
    """
    G = nx.DiGraph()

    with driver.session() as session:
        # ── nodes ──
        result = session.run(
            """
            MATCH (t:Taxpayer)
            RETURN t.gstin       AS gstin,
                   t.name        AS name,
                   t.compliance_rating AS rating,
                   t.business_type     AS btype,
                   t.is_active         AS active
            """
        )
        for record in result:
            G.add_node(
                record["gstin"],
                name=record["name"] or "",
                compliance_rating=record["rating"] or "C",
                business_type=record["btype"] or "Unknown",
                is_active=record["active"] if record["active"] is not None else True,
            )

        # ── edges from SUPPLIES_TO ──
        result = session.run(
            """
            MATCH (a:Taxpayer)-[r:SUPPLIES_TO]->(b:Taxpayer)
            RETURN a.gstin AS source,
                   b.gstin AS target,
                   r.total_value       AS total_value,
                   r.transaction_count AS tx_count
            """
        )
        for record in result:
            G.add_edge(
                record["source"],
                record["target"],
                total_value=record["total_value"] or 0,
                transaction_count=record["tx_count"] or 0,
            )

        # ── edges from CLAIMED_ITC_FROM (add if not already present) ──
        result = session.run(
            """
            MATCH (a:Taxpayer)-[r:CLAIMED_ITC_FROM]->(b:Taxpayer)
            RETURN a.gstin AS source,
                   b.gstin AS target,
                   r.total_amount  AS total_amount,
                   r.invoice_count AS inv_count
            """
        )
        for record in result:
            src, tgt = record["source"], record["target"]
            if not G.has_edge(src, tgt):
                G.add_edge(
                    src,
                    tgt,
                    total_value=record["total_amount"] or 0,
                    transaction_count=record["inv_count"] or 0,
                )

    logger.info("NetworkX graph built — nodes: %d, edges: %d", G.number_of_nodes(), G.number_of_edges())
    return G


# ──────────────────────────────────────────────
# Pre-compute expensive graph metrics ONCE
# ──────────────────────────────────────────────
def precompute_graph_metrics(G: nx.DiGraph) -> dict:
    """
    Compute PageRank, betweenness centrality, clustering coefficient,
    weakly-connected components, and cycle memberships for ALL nodes.

    Returns a dict keyed by gstin with sub-dicts of metric values.
    """
    metrics: dict = defaultdict(dict)

    # --- PageRank ---
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pr = {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes()}
    for n, v in pr.items():
        metrics[n]["pagerank"] = v

    # --- Betweenness centrality ---
    try:
        bc = nx.betweenness_centrality(G)
    except Exception:
        bc = {n: 0.0 for n in G.nodes()}
    for n, v in bc.items():
        metrics[n]["betweenness_centrality"] = v

    # --- Clustering coefficient (on undirected projection) ---
    G_undirected = G.to_undirected()
    cc = nx.clustering(G_undirected)
    for n, v in cc.items():
        metrics[n]["clustering_coefficient"] = v

    # --- Weakly connected component sizes ---
    comp_map: dict = {}
    for comp in nx.weakly_connected_components(G):
        size = len(comp)
        for n in comp:
            comp_map[n] = size
    for n in G.nodes():
        metrics[n]["connected_component_size"] = comp_map.get(n, 1)

    # --- Cycle participation (length ≤ 5) ---
    nodes_in_cycles: set = set()
    try:
        for cycle in nx.simple_cycles(G, length_bound=5):
            for n in cycle:
                nodes_in_cycles.add(n)
    except Exception:
        logger.warning("Cycle detection failed or timed out; marking none.")
    for n in G.nodes():
        metrics[n]["cycle_participation"] = 1.0 if n in nodes_in_cycles else 0.0

    logger.info(
        "Graph metrics pre-computed — cycles found involving %d nodes", len(nodes_in_cycles)
    )
    return dict(metrics)


# ──────────────────────────────────────────────
# Structural features (per node, from NetworkX)
# ──────────────────────────────────────────────
def compute_structural_features(
    G: nx.DiGraph, gstin: str, precomputed: dict
) -> dict:
    """
    Return 6 structural features for *gstin*:
        in_degree, out_degree, pagerank, betweenness_centrality,
        clustering_coefficient, connected_component_size
    """
    node_metrics = precomputed.get(gstin, {})
    return {
        "in_degree": float(G.in_degree(gstin)) if G.has_node(gstin) else 0.0,
        "out_degree": float(G.out_degree(gstin)) if G.has_node(gstin) else 0.0,
        "pagerank": node_metrics.get("pagerank", 0.0),
        "betweenness_centrality": node_metrics.get("betweenness_centrality", 0.0),
        "clustering_coefficient": node_metrics.get("clustering_coefficient", 0.0),
        "connected_component_size": float(node_metrics.get("connected_component_size", 1)),
    }


# ──────────────────────────────────────────────
# Behavioral features (per node, from Neo4j + SQL)
# ──────────────────────────────────────────────
def compute_behavioral_features(driver, session, gstin: str) -> dict:
    """
    Return 8 behavioural / compliance features for *gstin*:
        gstr1_filing_rate, gstr3b_filing_rate, filing_gap,
        itc_overclaim_ratio, irn_coverage, eway_coverage,
        transaction_volume, transaction_frequency
    """

    # ── GSTR-1 filing rate (Neo4j) ──
    with driver.session() as neo_sess:
        rec = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[:FILED_GSTR1]->(g:GSTR1Filing)
            WHERE g.status = 'Filed'
            RETURN count(g) AS cnt
            """,
            gstin=gstin,
        ).single()
        gstr1_filed = rec["cnt"] if rec else 0

    gstr1_filing_rate = min(gstr1_filed / EXPECTED_FILING_PERIODS, 1.0)

    # ── GSTR-3B filing rate (Neo4j) ──
    with driver.session() as neo_sess:
        rec = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[:FILED_GSTR3B]->(g:GSTR3BFiling)
            WHERE g.status = 'Filed'
            RETURN count(g) AS cnt
            """,
            gstin=gstin,
        ).single()
        gstr3b_filed = rec["cnt"] if rec else 0

    gstr3b_filing_rate = min(gstr3b_filed / EXPECTED_FILING_PERIODS, 1.0)

    filing_gap = gstr1_filing_rate - gstr3b_filing_rate  # positive ⇒ shell-company indicator

    # ── ITC overclaim ratio (PostgreSQL) ──
    try:
        row = session.execute(
            text(
                """
                SELECT COALESCE(SUM(cgst_amount + sgst_amount + igst_amount), 0) AS available
                FROM gstr1_records
                WHERE buyer_gstin = :gstin AND return_status = 'Filed'
                """
            ),
            {"gstin": gstin},
        ).fetchone()
        itc_available = float(row[0]) if row else 0.0

        row = session.execute(
            text(
                """
                SELECT COALESCE(SUM(total_itc_claimed), 0) AS claimed
                FROM gstr3b_records
                WHERE gstin = :gstin
                """
            ),
            {"gstin": gstin},
        ).fetchone()
        itc_claimed = float(row[0]) if row else 0.0

        if itc_available > 0:
            itc_overclaim_ratio = max((itc_claimed - itc_available) / itc_available, 0.0)
        else:
            itc_overclaim_ratio = 1.0 if itc_claimed > 0 else 0.0
    except Exception as exc:
        logger.warning("ITC overclaim query failed for %s: %s", gstin, exc)
        itc_overclaim_ratio = 0.0

    # ── IRN coverage (PostgreSQL) ──
    try:
        row = session.execute(
            text(
                """
                SELECT COUNT(*) AS total,
                       COUNT(irn.irn_number) AS with_irn
                FROM invoices inv
                LEFT JOIN irn_records irn
                    ON inv.invoice_number = irn.invoice_number
                   AND irn.status = 'Active'
                WHERE inv.seller_gstin = :gstin
                """
            ),
            {"gstin": gstin},
        ).fetchone()
        total_inv = int(row[0]) if row else 0
        with_irn = int(row[1]) if row else 0
        irn_coverage = with_irn / total_inv if total_inv > 0 else 1.0
    except Exception as exc:
        logger.warning("IRN coverage query failed for %s: %s", gstin, exc)
        irn_coverage = 1.0

    # ── e-Way coverage (PostgreSQL, invoices > ₹50 000) ──
    try:
        row = session.execute(
            text(
                """
                SELECT COUNT(*) AS total,
                       COUNT(ew.eway_bill_number) AS with_eway
                FROM invoices inv
                LEFT JOIN eway_bills ew
                    ON inv.invoice_number = ew.invoice_number
                   AND ew.status = 'Active'
                WHERE inv.seller_gstin = :gstin
                  AND inv.total_value > 50000
                """
            ),
            {"gstin": gstin},
        ).fetchone()
        total_qualifying = int(row[0]) if row else 0
        with_eway = int(row[1]) if row else 0
        eway_coverage = with_eway / total_qualifying if total_qualifying > 0 else 1.0
    except Exception as exc:
        logger.warning("eWay coverage query failed for %s: %s", gstin, exc)
        eway_coverage = 1.0

    # ── Transaction volume & frequency (Neo4j) ──
    with driver.session() as neo_sess:
        rec = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[r:SUPPLIES_TO]-(other:Taxpayer)
            RETURN COALESCE(SUM(r.total_value), 0) AS vol,
                   COUNT(DISTINCT other.gstin)      AS freq
            """,
            gstin=gstin,
        ).single()
        raw_volume = float(rec["vol"]) if rec else 0.0
        transaction_frequency = float(rec["freq"]) if rec else 0.0

    transaction_volume = math.log1p(raw_volume)  # log-scale normalisation

    return {
        "gstr1_filing_rate": gstr1_filing_rate,
        "gstr3b_filing_rate": gstr3b_filing_rate,
        "filing_gap": filing_gap,
        "itc_overclaim_ratio": min(itc_overclaim_ratio, 5.0),  # cap at 5x
        "irn_coverage": irn_coverage,
        "eway_coverage": eway_coverage,
        "transaction_volume": transaction_volume,
        "transaction_frequency": transaction_frequency,
    }


# ──────────────────────────────────────────────
# Network risk features (per node)
# ──────────────────────────────────────────────
def compute_network_risk_features(
    driver, G: nx.DiGraph, gstin: str, precomputed: dict
) -> dict:
    """
    Return 4 network-risk features for *gstin*:
        avg_neighbor_risk, risky_neighbor_ratio,
        cycle_participation, business_type_encoded
    """
    # ── Neighbor risk from Neo4j ──
    with driver.session() as neo_sess:
        result = neo_sess.run(
            """
            MATCH (t:Taxpayer {gstin: $gstin})-[:SUPPLIES_TO|CLAIMED_ITC_FROM]-(n:Taxpayer)
            RETURN n.compliance_rating AS rating
            """,
            gstin=gstin,
        )
        ratings = [r["rating"] for r in result if r["rating"] is not None]

    if ratings:
        encoded = [COMPLIANCE_RATING_ENCODING.get(r, COMPLIANCE_RATING_DEFAULT) for r in ratings]
        avg_neighbor_risk = sum(encoded) / len(encoded)
        risky_count = sum(1 for r in ratings if r == "D")
        risky_neighbor_ratio = risky_count / len(ratings)
    else:
        avg_neighbor_risk = 0.5
        risky_neighbor_ratio = 0.0

    # ── Cycle participation (pre-computed) ──
    node_metrics = precomputed.get(gstin, {})
    cycle_participation = node_metrics.get("cycle_participation", 0.0)

    # ── Business type encoding ──
    btype = G.nodes[gstin].get("business_type", "Unknown") if G.has_node(gstin) else "Unknown"
    business_type_encoded = BUSINESS_TYPE_ENCODING.get(btype, BUSINESS_TYPE_DEFAULT)

    return {
        "avg_neighbor_risk": avg_neighbor_risk,
        "risky_neighbor_ratio": risky_neighbor_ratio,
        "cycle_participation": cycle_participation,
        "business_type_encoded": business_type_encoded,
    }


# ──────────────────────────────────────────────
# Master extractor — all features, all nodes
# ──────────────────────────────────────────────
def extract_all_features(driver, session) -> pd.DataFrame:
    """
    For every Taxpayer node in Neo4j, compute the 18-dimensional
    feature vector.

    Returns
    -------
    pd.DataFrame
        Columns: ['gstin'] + FEATURE_NAMES  (19 columns total)
        One row per taxpayer, sorted by gstin.
    """
    G = build_networkx_graph(driver)

    if G.number_of_nodes() == 0:
        logger.error("Graph has no nodes — cannot extract features.")
        return pd.DataFrame(columns=["gstin"] + FEATURE_NAMES)

    precomputed = precompute_graph_metrics(G)

    all_gstins = sorted(G.nodes())
    rows = []

    for idx, gstin in enumerate(all_gstins):
        structural = compute_structural_features(G, gstin, precomputed)
        behavioral = compute_behavioral_features(driver, session, gstin)
        network = compute_network_risk_features(driver, G, gstin, precomputed)

        row = {"gstin": gstin}
        row.update(structural)
        row.update(behavioral)
        row.update(network)
        rows.append(row)

        if (idx + 1) % 10 == 0:
            logger.info("Features extracted for %d / %d nodes", idx + 1, len(all_gstins))

    df = pd.DataFrame(rows)
    # ensure column order matches FEATURE_NAMES
    df = df[["gstin"] + FEATURE_NAMES]
    logger.info("Feature extraction complete — shape: %s", df.shape)
    return df


def detect_cycles(driver) -> list:
    """
    Detect circular trading cycles (length ≤ 5) in the SUPPLIES_TO graph.

    Returns
    -------
    list[list[str]]
        Each inner list is an ordered list of GSTINs forming a cycle.
    """
    G = build_networkx_graph(driver)
    cycles = []
    try:
        for cycle in nx.simple_cycles(G, length_bound=5):
            cycles.append(list(cycle))
    except Exception as exc:
        logger.warning("Cycle detection error: %s", exc)
    logger.info("Detected %d cycles in graph", len(cycles))
    return cycles