"""
database/neo4j_connection.py
"""

from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
import os
import warnings

# Suppress non-critical Neo4j DNS deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j")

load_dotenv()

# ── Singleton driver instance ──────────────────────────────────
_driver: Driver | None = None


def get_driver() -> Driver:
    """
    Returns the shared Neo4j driver.
    Creates it on first call (lazy init).
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER",     "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password123"),
            ),
        )
    return _driver


def close_driver():
    """Call this on app shutdown to close the connection pool."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_query(query: str, params: dict = {}) -> list[dict]:
    """
    Execute a Cypher query and return results as a list of dicts.

    Usage:
        results = run_query(
            "MATCH (t:Taxpayer {gstin: $gstin}) RETURN t",
            {"gstin": "27XXXXX"}
        )
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]


def run_write_query(query: str, params: dict = {}) -> None:
    """
    Execute a write Cypher query (CREATE / MERGE / SET / DELETE).
    Use for mutations — does not return results.
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(query, params)
