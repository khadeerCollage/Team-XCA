"""
Tests for the FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


VALID_PAYLOAD = {
    "invoice_number": "INV-2024-882",
    "buyer_name": "Ananya Textiles",
    "buyer_gstin": "27ABCDE1234F1Z5",
    "seller_name": "Vikram Fabrics",
    "seller_gstin": "29ABCDE5678G1Z9",
    "itc_amount": 60000,
    "gstr1_filed": True,
    "gstr3b_filed": True,
    "tax_paid": True,
    "irn_valid": True,
    "eway_bill_verified": True,
    "route": "BLR-MUM",
}


class TestHealthEndpoint:
    def test_health_check(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestAuditEndpoint:
    def test_valid_chain_returns_200(self):
        resp = client.post("/api/v1/audit", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chain_valid"] is True
        assert data["decision"] == "ITC APPROVED"
        assert data["risk_level"] == "LOW"

    def test_gstr3b_break_returns_rejected(self):
        payload = {**VALID_PAYLOAD, "gstr3b_filed": False}
        resp = client.post("/api/v1/audit", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chain_valid"] is False
        assert data["decision"] == "ITC REJECTED"
        assert data["risk_level"] == "HIGH"
        assert data["break_point"] == "GSTR3B"

    def test_missing_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "itc_amount"}
        resp = client.post("/api/v1/audit", json=payload)
        assert resp.status_code == 422

    def test_unknown_field_returns_422(self):
        payload = {**VALID_PAYLOAD, "unknown_field": "bad"}
        resp = client.post("/api/v1/audit", json=payload)
        assert resp.status_code == 422

    def test_invalid_gstin_returns_422(self):
        payload = {**VALID_PAYLOAD, "buyer_gstin": "SHORT"}
        resp = client.post("/api/v1/audit", json=payload)
        assert resp.status_code == 422

    def test_zero_amount_returns_422(self):
        payload = {**VALID_PAYLOAD, "itc_amount": 0}
        resp = client.post("/api/v1/audit", json=payload)
        assert resp.status_code == 422


class TestReportOnlyEndpoint:
    def test_report_only_valid(self):
        resp = client.post("/api/v1/audit/report-only", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert "decision" in data
        assert data["decision"] == "ITC APPROVED"

    def test_report_only_invalid(self):
        payload = {**VALID_PAYLOAD, "tax_paid": False}
        resp = client.post("/api/v1/audit/report-only", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "ITC REJECTED"
        assert data["risk_level"] == "HIGH"


class TestRootEndpoint:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "GST Audit Report Engine"
