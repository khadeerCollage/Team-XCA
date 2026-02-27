# GST Audit Report Engine

**Deterministic, structured audit report generator for GST compliance validation.**

Evaluates the 5-step GST chain → classifies risk → produces regulator-ready reports.

No LLM. No AI magic. Pure deterministic logic.

---

## Architecture

```
Validation Input (JSON)
        ↓
  Chain Validator        → 5 checkpoints evaluated in strict order
        ↓
  Risk Classifier        → LOW / MEDIUM / HIGH / CRITICAL
        ↓
  Report Renderer        → Structured audit report text
        ↓
  FastAPI REST API       → JSON response with full audit result
```

## Validation Chain (Strict Order)

| Step | Checkpoint   | Major? |
|------|-------------|--------|
| 1    | GSTR-1      | No     |
| 2    | GSTR-3B     | Yes    |
| 3    | Tax Payment | Yes    |
| 4    | IRN         | Yes    |
| 5    | e-Way Bill  | No     |

## Risk Classification

| Condition | Risk Level | Decision |
|-----------|-----------|----------|
| All pass | LOW | ITC APPROVED |
| Minor fail only (GSTR-1 / e-Way) | MEDIUM | ITC APPROVED |
| Any major fail (GSTR-3B / Tax / IRN) | HIGH | ITC REJECTED |
| All fail | CRITICAL | ITC REJECTED |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/audit` | Full audit report (JSON) |
| POST | `/api/v1/audit/report-only` | Report text + decision only |

### Interactive docs: `http://localhost:8000/docs`

---

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/audit \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_number": "INV-2024-882",
    "buyer_name": "Ananya Textiles",
    "buyer_gstin": "27ABCDE1234F1Z5",
    "seller_name": "Vikram Fabrics",
    "seller_gstin": "29ABCDE5678G1Z9",
    "itc_amount": 60000,
    "gstr1_filed": true,
    "gstr3b_filed": false,
    "tax_paid": false,
    "irn_valid": true,
    "eway_bill_verified": true,
    "route": "BLR-MUM"
  }'
```

## Example Report Output

**Valid chain:**
```
Invoice #INV-2024-882 claimed by Ananya Textiles (GSTIN: 27ABCDE...)
for ₹60,000 ITC from Vikram Fabrics is VALID.
All 5 verification checkpoints passed. IRN confirmed. e-Way Bill verified for route BLR-MUM.
Chain: GSTR1 ✓ → GSTR3B ✓ → TAX ✓ → IRN ✓ → EWB ✓.
Risk Level: LOW. Risk Score: 100/100.
Final Decision: ITC APPROVED.
```

**Broken chain (GSTR-3B):**
```
Invoice #INV-2024-882 claimed by Ananya Textiles (GSTIN: 27ABCDE...)
for ₹60,000 ITC from Vikram Fabrics is INVALID.
Seller has not filed GSTR-3B. Tax payment could not be verified.
Chain broken at GSTR3B checkpoint (Step 2).
Chain: GSTR1 ✓ → GSTR3B ✗ → TAX ✓ → IRN ✓ → EWB ✓.
Risk Level: HIGH. Risk Score: 75/100.
Final Decision: ITC REJECTED.
```

---

## Project Structure

```
gst-audit-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # REST API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py        # Main orchestrator
│   │   ├── models.py        # Pydantic models & enums
│   │   ├── renderer.py      # Report rendering
│   │   ├── risk.py          # Risk classification
│   │   └── validator.py     # Chain validation
│   └── templates/
│       ├── __init__.py
│       └── report_templates.py  # Centralized templates
├── tests/
│   ├── __init__.py
│   ├── test_engine.py       # Core engine tests (40+ cases)
│   └── test_api.py          # API endpoint tests
├── requirements.txt
└── README.md
```
