"""
Invoice-Level Feature Engineering
===================================
Extracts numerical features from each invoice by combining data from
PostgreSQL tables and the Neo4j knowledge graph.

These features are the input (X) for the XGBoost classifier.

Feature vector per invoice (14 dimensions):
    1.  seller_compliance_score
    2.  buyer_compliance_score
    3.  tax_difference_ratio
    4.  irn_missing_flag
    5.  irn_cancelled_flag
    6.  eway_missing_flag
    7.  eway_required_flag
    8.  filing_delay_days
    9.  seller_historical_mismatch_rate
    10. buyer_historical_mismatch_rate
    11. seller_gstr3b_filing_rate
    12. invoice_value_log
    13. connected_high_risk_neighbors
    14. seller_transaction_frequency
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gstneo4j"
POSTGRES_URL = "postgresql://gstuser:gstpass@localhost:5432/gstdb"

EXPECTED_FILING_PERIODS = 6

COMPLIANCE_SCORE_MAP = {
    "A": 1.0,
    "B": 0.75,
    "C": 0.50,
    "D": 0.25,
}

FEATURE_NAMES = [
    "seller_compliance_score",
    "buyer_compliance_score",
    "tax_difference_ratio",
    "irn_missing_flag",
    "irn_cancelled_flag",
    "eway_missing_flag",
    "eway_required_flag",
    "filing_delay_days",
    "seller_historical_mismatch_rate",
    "buyer_historical_mismatch_rate",
    "seller_gstr3b_filing_rate",
    "invoice_value_log",
    "connected_high_risk_neighbors",
    "seller_transaction_frequency",
]


def get_feature_names() -> List[str]:
    """Return ordered list of feature names."""
    return list(FEATURE_NAMES)


class FeatureExtractor:
    """
    Extracts a 14-dimensional feature vector for each invoice.
    """

    def __init__(self, driver=None, session=None):
        self._driver = driver
        self._session = session
        self._owns_driver = False
        self._owns_session = False

        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

        if self._session is None:
            engine = create_engine(POSTGRES_URL)
            Session = sessionmaker(bind=engine)
            self._session = Session()
            self._owns_session = True

        # Caches
        self._compliance_cache: Dict[str, float] = {}
        self._gstr3b_rate_cache: Dict[str, float] = {}
        self._mismatch_rate_cache: Dict[str, float] = {}
        self._risk_neighbor_cache: Dict[str, int] = {}
        self._tx_freq_cache: Dict[str, int] = {}

    def close(self):
        if self._owns_session and self._session:
            self._session.close()
        if self._owns_driver and self._driver:
            self._driver.close()

    # ────────────────────────────────────────
    # Individual feature extractors
    # ────────────────────────────────────────
    def _get_compliance_score(self, gstin: str) -> float:
        """
        Feature: seller_compliance_score / buyer_compliance_score
        Source: taxpayers.compliance_rating → numerical encoding
        """
        if gstin in self._compliance_cache:
            return self._compliance_cache[gstin]

        row = self._session.execute(
            text("SELECT compliance_rating FROM taxpayers WHERE gstin = :g"),
            {"g": gstin},
        ).fetchone()

        rating = row[0] if row else "C"
        score = COMPLIANCE_SCORE_MAP.get(rating, 0.50)
        self._compliance_cache[gstin] = score
        return score

    def _get_tax_difference_ratio(
        self, seller_gstin: str, invoice_number: str,
        inv_taxable: float, inv_cgst: float, inv_sgst: float, inv_igst: float,
    ) -> float:
        """
        Feature: tax_difference_ratio
        = abs(invoice_total_tax - gstr1_total_tax) / max(invoice_total_tax, 1)

        If GSTR-1 not found → ratio = 1.0 (maximum mismatch)
        """
        inv_total_tax = inv_cgst + inv_sgst + inv_igst

        row = self._session.execute(
            text("""
                SELECT cgst_amount, sgst_amount, igst_amount
                FROM gstr1_records
                WHERE seller_gstin = :s AND invoice_number = :inv
                LIMIT 1
            """),
            {"s": seller_gstin, "inv": invoice_number},
        ).fetchone()

        if row is None:
            return 1.0

        gstr1_total_tax = float(row[0] or 0) + float(row[1] or 0) + float(row[2] or 0)
        denominator = max(inv_total_tax, 1.0)
        ratio = abs(inv_total_tax - gstr1_total_tax) / denominator
        return min(ratio, 5.0)  # cap at 5x

    def _get_irn_flags(self, invoice_number: str) -> Tuple[float, float]:
        """
        Features: irn_missing_flag, irn_cancelled_flag
        """
        row = self._session.execute(
            text("SELECT status FROM irn_records WHERE invoice_number = :inv LIMIT 1"),
            {"inv": invoice_number},
        ).fetchone()

        if row is None:
            return 1.0, 0.0  # missing, not cancelled

        status = (row[0] or "").lower()
        if status == "cancelled":
            return 0.0, 1.0  # present but cancelled
        if status in ("active", "valid"):
            return 0.0, 0.0  # present and valid

        return 1.0, 0.0  # unknown status → treat as missing

    def _get_eway_flags(self, invoice_number: str, invoice_value: float) -> Tuple[float, float]:
        """
        Features: eway_missing_flag, eway_required_flag
        """
        eway_required = 1.0 if invoice_value > 50000 else 0.0

        row = self._session.execute(
            text("SELECT status FROM eway_bills WHERE invoice_number = :inv LIMIT 1"),
            {"inv": invoice_number},
        ).fetchone()

        if row is None:
            eway_missing = 1.0 if eway_required else 0.0
        else:
            status = (row[0] or "").lower()
            eway_missing = 0.0 if status in ("active", "valid") else 1.0

        return eway_missing, eway_required

    def _get_filing_delay_days(self, seller_gstin: str, invoice_number: str) -> float:
        """
        Feature: filing_delay_days
        = (gstr1_filing_date - invoice_date).days

        If GSTR-1 not filed → return 180 (6 months penalty default)
        """
        row = self._session.execute(
            text("""
                SELECT g.filing_date, i.invoice_date
                FROM gstr1_records g
                JOIN invoices i ON g.invoice_number = i.invoice_number
                WHERE g.seller_gstin = :s AND g.invoice_number = :inv
                LIMIT 1
            """),
            {"s": seller_gstin, "inv": invoice_number},
        ).fetchone()

        if row is None or row[0] is None or row[1] is None:
            return 180.0

        try:
            from datetime import datetime

            filing_dt = row[0] if isinstance(row[0], datetime) else datetime.strptime(str(row[0]), "%Y-%m-%d")
            invoice_dt = row[1] if isinstance(row[1], datetime) else datetime.strptime(str(row[1]), "%Y-%m-%d")
            delay = (filing_dt - invoice_dt).days
            return max(float(delay), 0.0)
        except Exception:
            return 30.0

    def _get_historical_mismatch_rate(self, gstin: str) -> float:
        """
        Feature: seller_historical_mismatch_rate / buyer_historical_mismatch_rate

        = (invoices with any mismatch involving this GSTIN) / total invoices

        Computed from: irn_records and eway_bills status checks
        """
        if gstin in self._mismatch_rate_cache:
            return self._mismatch_rate_cache[gstin]

        # Count total invoices involving this GSTIN
        row = self._session.execute(
            text("""
                SELECT COUNT(*) FROM invoices
                WHERE seller_gstin = :g OR buyer_gstin = :g
            """),
            {"g": gstin},
        ).fetchone()
        total = int(row[0]) if row else 0

        if total == 0:
            self._mismatch_rate_cache[gstin] = 0.0
            return 0.0

        # Count invoices with issues
        row2 = self._session.execute(
            text("""
                SELECT COUNT(DISTINCT inv.invoice_number)
                FROM invoices inv
                LEFT JOIN irn_records irn ON inv.invoice_number = irn.invoice_number
                LEFT JOIN gstr1_records g1 ON inv.invoice_number = g1.invoice_number
                    AND inv.seller_gstin = g1.seller_gstin
                WHERE (inv.seller_gstin = :g OR inv.buyer_gstin = :g)
                  AND (irn.irn_number IS NULL
                       OR irn.status = 'Cancelled'
                       OR g1.invoice_number IS NULL
                       OR g1.return_status != 'Filed')
            """),
            {"g": gstin},
        ).fetchone()
        mismatch_count = int(row2[0]) if row2 else 0

        rate = mismatch_count / total
        self._mismatch_rate_cache[gstin] = rate
        return rate

    def _get_gstr3b_filing_rate(self, gstin: str) -> float:
        """
        Feature: seller_gstr3b_filing_rate
        = filed periods / expected periods
        """
        if gstin in self._gstr3b_rate_cache:
            return self._gstr3b_rate_cache[gstin]

        with self._driver.session() as neo_sess:
            rec = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:FILED_GSTR3B]->(g:GSTR3BFiling)
                WHERE g.status = 'Filed'
                RETURN count(g) AS cnt
                """,
                gstin=gstin,
            ).single()
            filed = rec["cnt"] if rec else 0

        rate = min(filed / EXPECTED_FILING_PERIODS, 1.0)
        self._gstr3b_rate_cache[gstin] = rate
        return rate

    def _get_connected_high_risk_neighbors(self, gstin: str) -> int:
        """
        Feature: connected_high_risk_neighbors
        Count of directly connected taxpayers with compliance_rating = 'D'
        """
        if gstin in self._risk_neighbor_cache:
            return self._risk_neighbor_cache[gstin]

        with self._driver.session() as neo_sess:
            rec = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:SUPPLIES_TO|CLAIMED_ITC_FROM]-(n:Taxpayer)
                WHERE n.compliance_rating = 'D'
                RETURN count(DISTINCT n.gstin) AS cnt
                """,
                gstin=gstin,
            ).single()
            count = rec["cnt"] if rec else 0

        self._risk_neighbor_cache[gstin] = count
        return count

    def _get_transaction_frequency(self, gstin: str) -> int:
        """
        Feature: seller_transaction_frequency
        Number of distinct trading partners
        """
        if gstin in self._tx_freq_cache:
            return self._tx_freq_cache[gstin]

        with self._driver.session() as neo_sess:
            rec = neo_sess.run(
                """
                MATCH (t:Taxpayer {gstin: $gstin})-[:SUPPLIES_TO|CLAIMED_ITC_FROM]-(n:Taxpayer)
                RETURN count(DISTINCT n.gstin) AS cnt
                """,
                gstin=gstin,
            ).single()
            count = rec["cnt"] if rec else 0

        self._tx_freq_cache[gstin] = count
        return count

    # ────────────────────────────────────────
    # Single invoice feature vector
    # ────────────────────────────────────────
    def extract_invoice_features(self, invoice: dict) -> Dict[str, float]:
        """
        Build 14-D feature vector for a single invoice.

        Parameters
        ----------
        invoice : dict
            Keys: invoice_number, seller_gstin, buyer_gstin,
                  taxable_value, cgst, sgst, igst, total_value
        """
        inv_num = invoice["invoice_number"]
        seller = invoice["seller_gstin"]
        buyer = invoice["buyer_gstin"]
        taxable = float(invoice.get("taxable_value", 0))
        cgst = float(invoice.get("cgst", 0))
        sgst = float(invoice.get("sgst", 0))
        igst = float(invoice.get("igst", 0))
        total = float(invoice.get("total_value", 0))

        irn_missing, irn_cancelled = self._get_irn_flags(inv_num)
        eway_missing, eway_required = self._get_eway_flags(inv_num, total)

        return {
            "seller_compliance_score": self._get_compliance_score(seller),
            "buyer_compliance_score": self._get_compliance_score(buyer),
            "tax_difference_ratio": self._get_tax_difference_ratio(
                seller, inv_num, taxable, cgst, sgst, igst
            ),
            "irn_missing_flag": irn_missing,
            "irn_cancelled_flag": irn_cancelled,
            "eway_missing_flag": eway_missing,
            "eway_required_flag": eway_required,
            "filing_delay_days": self._get_filing_delay_days(seller, inv_num),
            "seller_historical_mismatch_rate": self._get_historical_mismatch_rate(seller),
            "buyer_historical_mismatch_rate": self._get_historical_mismatch_rate(buyer),
            "seller_gstr3b_filing_rate": self._get_gstr3b_filing_rate(seller),
            "invoice_value_log": math.log1p(total),
            "connected_high_risk_neighbors": float(
                self._get_connected_high_risk_neighbors(seller)
            ),
            "seller_transaction_frequency": float(
                self._get_transaction_frequency(seller)
            ),
        }

    # ────────────────────────────────────────
    # All invoices
    # ────────────────────────────────────────
    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract feature matrix for every invoice in the database.

        Returns
        -------
        pd.DataFrame
            Columns: ['invoice_number'] + FEATURE_NAMES (15 columns)
        """
        inv_df = pd.read_sql(
            "SELECT invoice_number, seller_gstin, buyer_gstin, "
            "taxable_value, cgst, sgst, igst, total_value FROM invoices "
            "ORDER BY invoice_number",
            self._session.bind,
        )

        if inv_df.empty:
            logger.warning("No invoices found")
            return pd.DataFrame(columns=["invoice_number"] + FEATURE_NAMES)

        rows = []
        total = len(inv_df)
        for idx, inv_row in inv_df.iterrows():
            features = self.extract_invoice_features(inv_row.to_dict())
            features["invoice_number"] = inv_row["invoice_number"]
            rows.append(features)

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                logger.info("Features extracted for %d / %d invoices", idx + 1, total)

        df = pd.DataFrame(rows)
        df = df[["invoice_number"] + FEATURE_NAMES]
        logger.info("Feature extraction complete — shape: %s", df.shape)
        return df