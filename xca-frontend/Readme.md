# ⬡ GST Reconciliation Engine
### Intelligent GST Reconciliation Using Knowledge Graphs
**HackWithAI · PS #76 · ACM KLH Bachupally**

---

## Problem Statement

India's GST system suffers from massive ITC (Input Tax Credit) fraud — estimated at **₹50,000+ crore annually**. The root cause is that GST data lives in 5 separate flat files (GSTR-1, GSTR-2B, GSTR-3B, e-Invoice, e-Way Bill) that are **never connected together**. Existing solutions do only 1-to-1 value matching and miss circular fraud rings, broken ITC chains, and vendor-level risk patterns entirely.

**This project treats GST reconciliation as a graph traversal problem — not a table matching problem.**

---

## Our Solution

We model the entire GST ecosystem as a **Knowledge Graph** — taxpayers, invoices, IRNs, e-Way Bills, returns, and payments become nodes; their relationships become edges. We then traverse this graph to validate every invoice through a **6-hop chain**, detect **circular fraud rings**, compute **vendor risk scores**, and generate **AI-powered audit trails**.

---

## Features

| Feature | Description |
|---|---|
| Knowledge Graph | D3 force-directed graph of taxpayers and transactions |
| 6-Hop Chain Validation | Invoice → IRN → e-Way Bill → GSTR-2B → GSTR-3B → Payment |
| Circular Fraud Detection | Graph cycle detection for fake invoice networks |
| Mismatch Classification | CRITICAL / HIGH / MEDIUM / LOW risk categorization |
| Vendor Risk Scoring | Weighted formula with 5 compliance factors |
| AI Audit Trail | Plain English explanation of every mismatch |
| Graph Schema View | Visual data model with 6 nodes and 8 relationships |
| Data Ingestion Pipeline | Multi-format ingestion (CSV, JSON, Excel, TSV) |

---

## Data Flow

```
Input Files (CSV / JSON / Excel / TSV)
         │
         ▼
  gst_ingest.py
  ┌─────────────────────────────────┐
  │  1. Read any file format        │
  │  2. Normalize column names      │
  │  3. Build Knowledge Graph       │
  │  4. Run 6-hop reconciliation    │
  │  5. Detect circular rings       │
  │  6. Compute vendor risk scores  │
  └─────────────────────────────────┘
         │
         ▼
  graph_output.json
  { nodes, edges, mismatches, vendor_scores }
         │
         ▼
  React Dashboard
  ┌────────────────────────────────────────┐
  │  Ingest → Graph → Mismatches →        │
  │  Vendors → Audit Trail → Schema       │
  └────────────────────────────────────────┘
```

---

## Knowledge Graph Schema

### Node Types

| Node | Key Properties |
|---|---|
| **Taxpayer** | gstin, pan, name, state, turnover, risk_score |
| **Invoice** | irn, invoice_no, date, taxable_value, igst, cgst, sgst |
| **IRN** | irn_no, ack_no, ack_date, portal_status |
| **GST Return** | return_type, period, filing_date, total_itc, tax_paid |
| **e-Way Bill** | ewb_no, validity_date, vehicle_no, distance_km |
| **Payment** | challan_no, amount, date, tax_head |

### Relationship Types

```
(Taxpayer) ──[:ISSUED]──────────► (Invoice)
(Taxpayer) ──[:RECEIVED]────────► (Invoice)
(Taxpayer) ──[:FILED]───────────► (Return)
(Taxpayer) ──[:TRANSACTS_WITH]──► (Taxpayer)   ← Circular fraud detection
(Invoice)  ──[:HAS_IRN]─────────► (IRN)
(Invoice)  ──[:COVERED_BY]──────► (e-Way Bill)
(Invoice)  ──[:REPORTED_IN]─────► (Return)
(Return)   ──[:SETTLED_VIA]─────► (Payment)
```

---

## The 6-Hop ITC Chain

Every invoice must pass all 6 checkpoints for ITC to be valid:

```
Hop 1 — Invoice      Does the invoice document exist?
    ↓
Hop 2 — IRN          Is it registered at the e-Invoice portal?
    ↓
Hop 3 — e-Way Bill   Did goods physically move (valid e-Way Bill)?
    ↓
Hop 4 — GSTR-2B      Did seller report it? Is it in buyer's auto-ITC?
    ↓
Hop 5 — GSTR-3B      Did buyer file their summary return?
    ↓
Hop 6 — Payment      Was tax actually paid via challan?
```

**If any hop fails → ITC is blocked from that point onwards.**

---

## Vendor Risk Score Formula

```
score = ( 0.35 × mismatch_rate
        + 0.25 × filing_delay_normalized
        + 0.20 × itc_overclaim_ratio
        + 0.15 × circular_transaction_flag
        + 0.05 × new_vendor_flag )

Risk Buckets:
  score < 0.3  →  LOW RISK    
  score 0.3–0.6 → MEDIUM RISK 
  score > 0.6  →  HIGH RISK   
```

---

## Fraud Types Detected

| Fraud Type | Detection Method | Hop |
|---|---|---|
| Ghost Invoice | Missing IRN + No e-Way Bill | Hop 2, 3 |
| Circular Transaction | Graph cycle detection on TRANSACTS_WITH edges | All |
| Value Inflation | GSTR-1 vs GSTR-2B delta > threshold | Hop 4 |
| Missing IRN | IRN node absent for mandatory invoice | Hop 2 |
| Expired e-Way Bill | validity_date < delivery_date | Hop 3 |
| ITC Overclaim | GSTR-3B claim > GSTR-2B eligible amount | Hop 5 |
| Non-filing | GSTR-3B status = Not Filed | Hop 5 |
| Fake Payment | Challan not in government records | Hop 6 |

---

## Supported Input Formats

The Python pipeline accepts any of these formats for each data source:

| Source | Accepted Formats |
|---|---|
| GSTR-1 | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| GSTR-2B | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| GSTR-3B | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| IRN Data | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| e-Way Bills | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| Invoice Records | `.csv` `.xlsx` `.xls` `.json` `.tsv` |
| Taxpayer Metadata | `.csv` `.xlsx` `.xls` `.json` `.tsv` |

Column names are auto-normalized — the pipeline recognizes 5–8 alias variations per field so your file does not need to follow a strict naming convention.

### Example — Use Your Own Files

```python
from gst_ingest import run_pipeline

result = run_pipeline({
    "taxpayer":  "your_data/taxpayer_metadata.csv",
    "gstr1":     "your_data/gstr1_oct2024.xlsx",
    "gstr2b":    "your_data/gstr2b_oct2024.json",
    "gstr3b":    "your_data/gstr3b_oct2024.csv",
    "irn":       "your_data/irn_registry.csv",
    "ewaybill":  "your_data/ewaybill_oct2024.xlsx",
    "invoice":   "your_data/purchase_register.csv",
}, output_path="graph_output.json")

print(result["summary"])
# {
#   "total_nodes": 21,
#   "total_edges": 30,
#   "total_mismatches": 4,
#   "critical": 3,
#   "total_itc_at_risk": 470000
# }
```

Any source can be omitted — the pipeline skips missing files gracefully.

---

## Dashboard Tabs

### Tab 1 — Ingest
Run the data ingestion pipeline. Watch 6 data sources load with live progress bars and a system log. All other tabs unlock after ingestion completes.

### Tab 2 — Graph (Fraud Network)
D3 force-directed graph of all taxpayers and their transaction relationships. Red glowing nodes = high risk. Dashed red edges = mismatch. Solid green = clean. The circular fraud ring is highlighted automatically. Drag nodes, click to inspect.

### Tab 3 — Mismatches (Risk Table)
Full sortable/filterable table of all detected mismatches. Shows GSTR-1, GSTR-2B, and GSTR-3B values side by side with delta and risk classification. Click "Audit →" on any row to open its audit trail.

### Tab 4 — Vendors (Risk Scores)
Every vendor ranked by their computed risk score. Shows the full formula with per-vendor breakdowns across all 5 scoring components. Sort by score, mismatch rate, ITC at risk, or filing delay.

### Tab 5 — Audit Trail
Select any mismatch to see its complete audit trail — invoice summary card, value comparison across all 3 returns, the 6-hop chain with the exact break point highlighted, and an AI-generated plain English explanation with recommended action.

### Tab 6 — Schema
Visual data model of the Knowledge Graph. Hover any node to see all its properties. Full list of all 8 relationship types with direction and color coding.

---

## Tech Stack

### Frontend
| Library | Version | Purpose |
|---|---|---|
| React | 18 | UI framework |
| D3.js | 7 | Force graph simulation |
| Vite | 5 | Build tool |

### Backend / Pipeline
| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Ingestion pipeline |
| pandas | latest | File reading (CSV/Excel/JSON) |
| openpyxl | latest | Excel support |
| NetworkX | (optional) | Graph traversal at scale |

### No paid APIs. No cloud dependencies. Runs fully local.

---

## Why Knowledge Graph Over Flat Matching?

| Capability | Flat Matching (Excel/GSTN) | Knowledge Graph (Our Approach) |
|---|---|---|
| Value mismatch detection | ✅ | ✅ |
| Missing IRN detection | ❌ | ✅ |
| e-Way Bill validation | ❌ | ✅ |
| Full 6-hop chain check | ❌ | ✅ |
| Circular fraud ring detection | ❌ | ✅ |
| Vendor risk prediction | ❌ | ✅ |
| Explainable audit trail | ❌ | ✅ |
| Cross-entity relationship analysis | ❌ | ✅ |

---

## Known Limitations

- **Mock data only** — Real GSTN API access requires GSP (GST Suvidha Provider) license
- **In-memory graph** — NetworkX works for demo scale; production needs Neo4j or AWS Neptune
- **Rule-based detection** — Fraud rules are hardcoded; production would use trained GNN models
- **Assumed score weights** — Formula weights are expert-estimated, not ML-trained on real fraud data
- **Batch processing** — Pipeline runs on full dataset; production needs Kafka streaming for real-time
- **Simplified GST rules** — Section 17(5) blocked credits, RCM, ISD rules not fully encoded

---

## Team

Built for **HackWithAI Hackathon** — PS #76
Organized by **ACM KLH Bachupally**
Duration: **24 hours**

---

## License

MIT License — free to use, modify, and distribute.