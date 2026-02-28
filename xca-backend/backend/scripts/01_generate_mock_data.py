import pandas as pd
import json
import random
import os
from faker import Faker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, '..', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'converted')

fake = Faker('en_IN')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: generate GSTIN ──────────────────────────────────────
def random_gstin():
    state_codes = ['27', '29', '07', '09', '33', '19']
    return random.choice(state_codes) + fake.bothify('????#####?#?#').upper()

# ── 1. taxpayers.json ───────────────────────────────────────────
taxpayers = []
gstin_pool = [random_gstin() for _ in range(20)]

for i, gstin in enumerate(gstin_pool):
    taxpayers.append({
        "gstin": gstin,
        "name": fake.company(),
        "state": fake.state(),
        "turnover": round(random.uniform(10, 500) * 100000, 2),
        "registration_date": str(fake.date_between(start_date='-5y', end_date='-1y')),
        "business_type": random.choice(["Manufacturer", "Trader", "Service Provider"]),
        "risk_score": round(random.uniform(0.1, 0.9), 2)
    })

with open(os.path.join(INPUT_DIR, 'taxpayers.json'), 'w') as f:
    json.dump(taxpayers, f, indent=2)

print("taxpayers.json created")

# ── 2. gstr1.xlsx (seller's outward invoices) ───────────────────
gstr1_rows = []
invoice_pool = []

for i in range(50):
    seller = random.choice(taxpayers)
    buyer  = random.choice([t for t in taxpayers if t['gstin'] != seller['gstin']])
    taxable = round(random.uniform(10000, 500000), 2)
    gst_rate = random.choice([5, 12, 18, 28])
    gst_amount = round(taxable * gst_rate / 100, 2)
    irn = fake.bothify('IRN-????-####-????').upper()
    
    row = {
        "invoice_no":     f"INV-2024-{str(i+1).zfill(4)}",
        "irn":            irn,
        "seller_gstin":   seller['gstin'],
        "seller_name":    seller['name'],
        "buyer_gstin":    buyer['gstin'],
        "buyer_name":     buyer['name'],
        "invoice_date":   str(fake.date_between(start_date='-1y', end_date='today')),
        "taxable_value":  taxable,
        "gst_rate":       gst_rate,
        "igst":           gst_amount if seller['gstin'][:2] != buyer['gstin'][:2] else 0,
        "cgst":           round(gst_amount/2, 2) if seller['gstin'][:2] == buyer['gstin'][:2] else 0,
        "sgst":           round(gst_amount/2, 2) if seller['gstin'][:2] == buyer['gstin'][:2] else 0,
        "period":         random.choice(["Oct-2024", "Nov-2024", "Dec-2024"]),
        "status":         "Filed"
    }
    gstr1_rows.append(row)
    invoice_pool.append(row)  # save for cross-referencing

pd.DataFrame(gstr1_rows).to_excel(os.path.join(INPUT_DIR, 'gstr1.xlsx'), index=False)
print("gstr1.xlsx created")

# ── 3. gstr2b.csv (buyer's ITC statement — WITH intentional mismatches) ──
gstr2b_rows = []
for inv in invoice_pool:
    # Inject ~20% mismatches
    mismatch = random.random() < 0.20
    reported_value = inv['taxable_value']
    if mismatch:
        reported_value = round(inv['taxable_value'] * random.uniform(0.7, 0.95), 2)

    gstr2b_rows.append({
        "invoice_no":        inv['invoice_no'],
        "irn":               inv['irn'] if random.random() > 0.05 else "MISSING",
        "supplier_gstin":    inv['seller_gstin'],
        "recipient_gstin":   inv['buyer_gstin'],
        "taxable_value":     reported_value,
        "itc_eligible":      round(reported_value * inv['gst_rate'] / 100, 2),
        "period":            inv['period'],
        "has_mismatch":      mismatch,
        "mismatch_type":     "Value Delta" if mismatch else "None"
    })

pd.DataFrame(gstr2b_rows).to_csv(os.path.join(INPUT_DIR, 'gstr2b.csv'), index=False)
print("gstr2b.csv created")

# ── 4. ewaybill.csv ─────────────────────────────────────────────
ewb_rows = []
for inv in invoice_pool:
    has_ewb = random.random() > 0.10  # 10% missing (fraud signal)
    ewb_rows.append({
        "ewb_no":         fake.bothify('EWB-########') if has_ewb else "MISSING",
        "invoice_no":     inv['invoice_no'],
        "seller_gstin":   inv['seller_gstin'],
        "buyer_gstin":    inv['buyer_gstin'],
        "vehicle_no":     fake.bothify('??-##-??-####').upper() if has_ewb else "MISSING",
        "from_state":     inv['seller_gstin'][:2],
        "to_state":       inv['buyer_gstin'][:2],
        "goods_value":    inv['taxable_value'],
        "valid":          has_ewb
    })

pd.DataFrame(ewb_rows).to_csv(os.path.join(INPUT_DIR, 'ewayhill.csv'), index=False)
print("ewaybill.csv created")

print(f"\nAll mock data files generated in {INPUT_DIR}")
