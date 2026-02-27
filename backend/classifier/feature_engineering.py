"""
Feature Engineering for Gradient Boosted Model
===============================================
Extracts features from an invoice & its participants (buyer/seller)
to feed into XGBoost.
"""

import pandas as pd
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self._feature_names = [
            "invoice_value",
            "tax_rate",
            "seller_historical_compliance",
            "buyer_historical_compliance",
            "filing_delay_days",
            "supplier_pagerank",
            "itc_ratio",
        ]

    def extract(self, invoice_data: dict) -> pd.DataFrame:
        """
        Extract numerical features from raw invoice metadata.
        Mock implementation for the hackathon baseline.
        """
        # In a real scenario, this merges SQL (invoice) + Neo4j (graph metrics)
        features = {
            "invoice_value": invoice_data.get("total_value", 0.0),
            "tax_rate": invoice_data.get("tax_amount", 0.0) / max(invoice_data.get("total_value", 1.0), 1.0),
            "seller_historical_compliance": invoice_data.get("seller_rating", 0.5),
            "buyer_historical_compliance": invoice_data.get("buyer_rating", 0.5),
            "filing_delay_days": invoice_data.get("delay_days", 0),
            "supplier_pagerank": invoice_data.get("supplier_pagerank", 0.01),
            "itc_ratio": invoice_data.get("itc_ratio", 0.0),
        }
        return pd.DataFrame([features])[self._feature_names]


def get_feature_names():
    return FeatureExtractor()._feature_names
