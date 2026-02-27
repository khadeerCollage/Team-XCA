"""
Network Connections Scorer
============================
Evaluates the quality and riskiness of a vendor's supply-chain
network based on graph topology in Neo4j.

Sub-scores:
    A. Circular trading penalty        (30% of component)
    B. Risky neighbour concentration   (30% of component)
    C. Network diversity               (25% of component)
    D. Transaction pattern score       (15% of component)

Output: 0–100 (higher = healthier network = safer)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"


@dataclass
class NetworkScoreBreakdown:
    """Detailed breakdown of network connections score."""
    gstin: str
    final_score: float = 100.0
    circular_trading_score: float = 100.0
    risky_neighbour_score: float = 100.0
    network_diversity_score: float = 100.0
    transaction_pattern_score: float = 100.0

    # Raw metrics
    in_degree: int = 0
    out_degree: int = 0
    total_connections: int = 0
    unique_counterparties: int = 0

    # Circular trading
    cycle_count: int = 0
    cycle_gstins: List[List[str]] = field(default_factory=list)

    # Neighbour risk
    avg_neighbour_risk: float = 0.0
    high_risk_neighbours: int = 0
    total_neighbours: int = 0
    risky_neighbour_ratio: float = 0.0

    # Concentration
    top_counterparty_share: float = 0.0

    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gstin": self.gstin,
            "final_score": round(self.final_score, 2),
            "circular_trading_score": round(self.circular_trading_score, 2),
            "risky_neighbour_score": round(self.risky_neighbour_score, 2),
            "network_diversity_score": round(self.network_diversity_score, 2),
            "transaction_pattern_score": round(self.transaction_pattern_score, 2),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "total_connections": self.total_connections,
            "unique_counterparties": self.unique_counterparties,
            "cycle_count": self.cycle_count,
            "avg_neighbour_risk": round(self.avg_neighbour_risk, 4),
            "high_risk_neighbours": self.high_risk_neighbours,
            "total_neighbours": self.total_neighbours,
            "risky_neighbour_ratio": round(self.risky_neighbour_ratio, 4),
            "top_counterparty_share": round(self.top_counterparty_share, 4),
            "details": self.details,
        }


class NetworkScorer:
    """
    Scores vendor based on supply-chain network topology.

    Score = 0.30 × CircularTradingScore
          + 0.30 × RiskyNeighbourScore
          + 0.25 × NetworkDiversityScore
          + 0.15 × TransactionPatternScore

    Uses Neo4j graph queries only (no SQL needed).
    """

    def __init__(self, driver=None):
        self._driver = driver
        self._owns_driver = False

        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

    def close(self):
        if self._owns_driver and self._driver:
            self._driver.close()

    # ────────────────────────────────────────
    # Data Gathering
    # ────────────────────────────────────────

    def _get_degree_stats(self, gstin: str) -> Dict:
        """Get in-degree (received) and out-degree (issued) for this vendor."""
        with self._driver.session() as sess:
            out_result = sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
                RETURN count(inv) AS out_degree
                """,
                gstin=gstin,
            ).single()

            in_result = sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:RECEIVED]->(inv:Invoice)
                RETURN count(inv) AS in_degree
                """,
                gstin=gstin,
            ).single()

            counterparty_result = sess.run(
                """
                OPTIONAL MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)<-[:RECEIVED]-(buyer:Taxpayer)
                WITH collect(DISTINCT buyer.gstin) AS buyers
                OPTIONAL MATCH (t2:Taxpayer {gstin: $gstin})-[:RECEIVED]->(inv2:Invoice)<-[:ISSUED]-(seller:Taxpayer)
                WITH buyers, collect(DISTINCT seller.gstin) AS sellers
                WITH [x IN buyers + sellers WHERE x IS NOT NULL] AS all_partners
                RETURN size(all_partners) AS unique_counterparties
                """,
                gstin=gstin,
            ).single()

        out_deg = out_result["out_degree"] if out_result else 0
        in_deg = in_result["in_degree"] if in_result else 0
        unique = counterparty_result["unique_counterparties"] if counterparty_result else 0

        return {
            "out_degree": out_deg,
            "in_degree": in_deg,
            "total_connections": out_deg + in_deg,
            "unique_counterparties": unique,
        }

    def _detect_circular_trading(self, gstin: str) -> Dict:
        """
        Detect circular trading paths that pass through this vendor.
        Cycles of length 3-5 in the TRANSACTS_WITH graph.
        """
        with self._driver.session() as sess:
            result = sess.run(
                """
                MATCH path = (a:Taxpayer {gstin: $gstin})-[:TRANSACTS_WITH*3..5]->(a)
                RETURN [n IN nodes(path) | n.gstin] AS cycle
                LIMIT 10
                """,
                gstin=gstin,
            )
            cycles = [record["cycle"] for record in result]

        return {
            "cycle_count": len(cycles),
            "cycles": cycles,
        }

    def _get_neighbour_risk(self, gstin: str) -> Dict:
        """
        Evaluate the risk profile of this vendor's direct counterparties.
        """
        with self._driver.session() as sess:
            result = sess.run(
                """
                OPTIONAL MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)<-[:RECEIVED]-(buyer:Taxpayer)
                WITH collect(DISTINCT {gstin: buyer.gstin, risk: coalesce(buyer.risk_score, 0.0)}) AS buyers
                OPTIONAL MATCH (t2:Taxpayer {gstin: $gstin})-[:RECEIVED]->(inv2:Invoice)<-[:ISSUED]-(seller:Taxpayer)
                WITH buyers, collect(DISTINCT {gstin: seller.gstin, risk: coalesce(seller.risk_score, 0.0)}) AS sellers
                WITH [x IN buyers + sellers WHERE x.gstin IS NOT NULL] AS all_neighbours
                UNWIND all_neighbours AS n
                WITH DISTINCT n.gstin AS neighbour_gstin, n.risk AS risk
                RETURN
                    count(*) AS total,
                    avg(risk) AS avg_risk,
                    sum(CASE WHEN risk > 0.6 THEN 1 ELSE 0 END) AS high_risk_count
                """,
                gstin=gstin,
            ).single()

        if result and result["total"] and result["total"] > 0:
            return {
                "total": result["total"],
                "avg_risk": result["avg_risk"] or 0.0,
                "high_risk_count": result["high_risk_count"] or 0,
            }
        return {"total": 0, "avg_risk": 0.0, "high_risk_count": 0}

    def _get_concentration_risk(self, gstin: str) -> float:
        """
        Check if transactions are concentrated with a single counterparty.
        Returns the share (0-1) of total value from the top counterparty.
        """
        with self._driver.session() as sess:
            result = sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:ISSUED]->(inv:Invoice)
                WITH inv.buyer_gstin AS partner, sum(inv.taxable_value) AS partner_value
                ORDER BY partner_value DESC
                WITH collect({partner: partner, value: partner_value}) AS partners
                WITH partners, reduce(s = 0.0, p IN partners | s + p.value) AS total_value
                WITH partners[0].value AS top_value, total_value
                RETURN CASE WHEN total_value > 0
                       THEN toFloat(top_value) / total_value
                       ELSE 0.0 END AS top_share
                """,
                gstin=gstin,
            ).single()

        return result["top_share"] if result and result["top_share"] else 0.0

    # ────────────────────────────────────────
    # Scoring
    # ────────────────────────────────────────

    def score_vendor(self, gstin: str) -> NetworkScoreBreakdown:
        """Compute network connections score for one vendor."""
        breakdown = NetworkScoreBreakdown(gstin=gstin)

        # ── Gather data ──
        degree = self._get_degree_stats(gstin)
        breakdown.in_degree = degree["in_degree"]
        breakdown.out_degree = degree["out_degree"]
        breakdown.total_connections = degree["total_connections"]
        breakdown.unique_counterparties = degree["unique_counterparties"]

        # If vendor has no transactions, assign neutral score
        if degree["total_connections"] == 0:
            breakdown.final_score = 75.0
            breakdown.details.append("No transaction history — neutral network score")
            return breakdown

        # ── Sub-score A: Circular Trading (30%) ──
        circular = self._detect_circular_trading(gstin)
        breakdown.cycle_count = circular["cycle_count"]
        breakdown.cycle_gstins = circular["cycles"]

        if circular["cycle_count"] == 0:
            breakdown.circular_trading_score = 100.0
        elif circular["cycle_count"] <= 2:
            breakdown.circular_trading_score = 30.0
            breakdown.details.append(
                f"WARNING: {circular['cycle_count']} circular trading pattern(s) detected"
            )
        else:
            breakdown.circular_trading_score = 0.0
            breakdown.details.append(
                f"CRITICAL: {circular['cycle_count']} circular trading rings — "
                f"strong fraud signal"
            )

        # ── Sub-score B: Risky Neighbour Concentration (30%) ──
        neighbours = self._get_neighbour_risk(gstin)
        breakdown.total_neighbours = neighbours["total"]
        breakdown.avg_neighbour_risk = neighbours["avg_risk"]
        breakdown.high_risk_neighbours = neighbours["high_risk_count"]

        if neighbours["total"] > 0:
            breakdown.risky_neighbour_ratio = (
                neighbours["high_risk_count"] / neighbours["total"]
            )
            breakdown.risky_neighbour_score = max(
                0.0, (1.0 - breakdown.risky_neighbour_ratio) * 100
            )
            if neighbours["avg_risk"] > 0.6:
                breakdown.risky_neighbour_score *= 0.7
                breakdown.details.append(
                    f"High-risk network: avg neighbour risk score "
                    f"{neighbours['avg_risk']:.2f}"
                )
            if neighbours["high_risk_count"] > 0:
                breakdown.details.append(
                    f"{neighbours['high_risk_count']}/{neighbours['total']} "
                    f"counterparties are high-risk"
                )
        else:
            breakdown.risky_neighbour_score = 80.0

        # ── Sub-score C: Network Diversity (25%) ──
        unique = degree["unique_counterparties"]
        if unique >= 5:
            breakdown.network_diversity_score = 100.0
        elif unique >= 3:
            breakdown.network_diversity_score = 75.0
        elif unique >= 2:
            breakdown.network_diversity_score = 50.0
            breakdown.details.append(
                "Low network diversity — only 2 counterparties"
            )
        elif unique == 1:
            breakdown.network_diversity_score = 20.0
            breakdown.details.append(
                "Single counterparty — possible shell company link"
            )
        else:
            breakdown.network_diversity_score = 0.0

        # Concentration penalty
        concentration = self._get_concentration_risk(gstin)
        breakdown.top_counterparty_share = concentration
        if concentration > 0.8:
            breakdown.network_diversity_score *= 0.5
            breakdown.details.append(
                f"{concentration*100:.0f}% of transactions with single partner — "
                f"high concentration risk"
            )
        elif concentration > 0.6:
            breakdown.network_diversity_score *= 0.75
            breakdown.details.append(
                f"{concentration*100:.0f}% of transactions concentrated with top partner"
            )

        # ── Sub-score D: Transaction Pattern (15%) ──
        if degree["out_degree"] > 0 and degree["in_degree"] > 0:
            ratio = min(degree["out_degree"], degree["in_degree"]) / max(
                degree["out_degree"], degree["in_degree"]
            )
            breakdown.transaction_pattern_score = min(100.0, ratio * 100 + 30)
        elif degree["out_degree"] > 0 or degree["in_degree"] > 0:
            breakdown.transaction_pattern_score = 40.0
            direction = "sells" if degree["out_degree"] > 0 else "buys"
            breakdown.details.append(
                f"One-sided transactions — vendor only {direction}"
            )
        else:
            breakdown.transaction_pattern_score = 0.0

        # ── Final weighted score ──
        breakdown.final_score = (
            0.30 * breakdown.circular_trading_score
            + 0.30 * breakdown.risky_neighbour_score
            + 0.25 * breakdown.network_diversity_score
            + 0.15 * breakdown.transaction_pattern_score
        )
        breakdown.final_score = max(0.0, min(100.0, breakdown.final_score))

        if not breakdown.details:
            breakdown.details.append("Healthy network — no significant risks detected")

        return breakdown

    def score_all_vendors(self) -> List[NetworkScoreBreakdown]:
        """Score all vendors in the database."""
        with self._driver.session() as sess:
            result = sess.run(
                "MATCH (t:Taxpayer) RETURN t.gstin AS gstin ORDER BY t.gstin"
            )
            gstins = [r["gstin"] for r in result]

        results = []
        for i, gstin in enumerate(gstins):
            score = self.score_vendor(gstin)
            results.append(score)
            if (i + 1) % 20 == 0:
                logger.info("Network scores computed: %d/%d", i + 1, len(gstins))

        return results