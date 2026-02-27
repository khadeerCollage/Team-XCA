# Intelligent GST Reconciliation & Fraud Risk Detection Using Knowledge Graphs

This repository contains the backend and machine learning components for GST (Goods and Services Tax) fraud detection. The system leverages Graph Neural Networks (GNN) and a Hybrid Classifier (Rule Engine + XGBoost) to identify tax evasion and syndicate frauds.

---

## Testing & Verification Guide

To ensure all models and rules are working perfectly, follow the data inputs and expected outputs described below.

### 1. Hybrid Classifier Testing (Rules + XGBoost)
The Hybrid Classifier handles individual invoice scoring by combining deterministic rules (for obvious GST mismatches) and an XGBoost machine learning model (for subtle anomalies like sudden filing spikes).

**A. Rule Engine Scenarios**
| Scenario | Invoice Details (Input) | Expected Rule | Expected Risk | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Transaction** | Present in GSTR-1, matches GSTR-3B exactly, valid IRN & e-Way Bill. | None | `LOW` | `< 0.3` |
| **Missing Filing** | Invoice in buyer\'s GSTR-2B, but NOT in supplier\'s GSTR-1. | `TYPE_A_MISSING_GSTR1` | `HIGH` | `> 0.7` |
| **Value Mismatch** | Supplier GSTR-1 shows ₹50,000, but GSTR-3B shows tax paid on ₹10,000. | `TYPE_B_VALUE_MISMATCH`| `MEDIUM` | `> 0.4` |
| **Fake E-Invoice** | Invoice > ₹5Cr, IRN hash doesn\'t match IRP portal signature. | `TYPE_C_FAKE_IRN` | `CRITICAL`| `> 0.9` |
| **Transit Fraud** | Goods > ₹50k transported without a valid e-Way Bill. | `TYPE_D_NO_EWAY` | `HIGH` | `> 0.8` |

**B. XGBoost ML Scenarios**
| Scenario | Features (Input) | Expected ML Label | Risk Probability |
| :--- | :--- | :--- | :--- |
| **Normal Behavior** | Files GSTR-1 by 10th, uses 10% cash/90% ITC, steady volume. | `0` (Safe) | `< 0.30` |
| **Shell Company** | Files GSTR-1 late, uses 100% ITC (zero cash tax paid), sudden 500% volume spike. | `1` (Risky) | `> 0.80` |

*Verification:* Call `POST /api/classifier/classify`. Check that `ml_probability` and `rule_score` align with the tables above to correctly calculate the final `risk_probability`.

---

### 2. Graph Neural Network (GNN) Testing
The GNN analyzes the entire Neo4j taxpayer network to find syndicates based on relationships (who buys from whom).

**A. Graph Features Extraction**
* **Isolated Taxpayer (Input)**: Buys from 2 suppliers, sells to 3 buyers.
* **Central Hub/Shell (Input)**: Buys from 50 suppliers and sells to 50 buyers within the exact same day.
* *Verification:* Run `backend/ml/graph_features.py`. The "Isolated" taxpayer must have low `in_degree` (2). The "Central Hub" must have an exceptionally high `pagerank` and massive degree metrics.

**B. Prediction Pipeline**
* **Fraud Syndicate (Input)**: In Neo4j, create nodes A->B->C->D->E->A trading high-value invoices circularly and exhausting ITC. Only label Node A as `is_fraud=True`.
* *Verification:* Run `/api/ml/train` then `/api/ml/predict`. 
  * Training should reach > 0.85 Accuracy. 
  * Predictions for Nodes C & D should yield a `fraud_probability > 0.80` despite not being explicitly labeled as fraud in the database, proving the GNN learned the fraudulent neighborhood structure.

---

### 3. AI Explainability Testing
The Explainer translates the GNN\'s raw logic into human-readable text for GST Officers.

* **Input**: Query `/api/ml/explain/{gstin}` using the GSTIN of the "Central Hub" Shell mentioned above.
* *Verification:* The API should return a JSON containing:
  1. `risk_level`: "CRITICAL"
  2. `top_risk_factors`: Explicit English explanations (e.g., *"100% of tax paid using ITC (0% cash)"*).
  3. `influential_neighbors`: A list of surrounding GSTINs that dragged the score down.
  4. `itc_impact`: The exact calculation of Input Tax Credit (₹) at risk for downstream buyers.
