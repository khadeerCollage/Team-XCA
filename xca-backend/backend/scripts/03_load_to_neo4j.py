import json
import os
import sys

# Make sure we can import from backend modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from database.neo4j_connection import get_driver, close_driver, run_write_query

INPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'converted', 'MASTER.json')

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run 02_convert_to_json.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Make connection to Neo4J
    driver = get_driver()
    if not driver:
         print("❌ Error: Could not connect to Neo4j. Check .env variables.")
         return
         
    print("🔌 Connected to Neo4j")

    # Support --append flag to keep existing data (used by pipeline API)
    append_mode = "--append" in sys.argv
    if not append_mode:
        print("🧹 Clearing database...")
        run_write_query("MATCH (n) DETACH DELETE n")
    else:
        print("📎 Append mode — keeping existing data")

    print("\n🛠️ Setting up Constraints and Indexes (Step 1 & 2)...")
    
    constraints = [
        "CREATE CONSTRAINT taxpayer_gstin IF NOT EXISTS FOR (t:Taxpayer) REQUIRE t.gstin IS UNIQUE;",
        "CREATE CONSTRAINT invoice_no IF NOT EXISTS FOR (i:Invoice) REQUIRE i.invoice_no IS UNIQUE;",
        "CREATE CONSTRAINT irn_code IF NOT EXISTS FOR (r:IRN) REQUIRE r.code IS UNIQUE;",
        "CREATE CONSTRAINT ewb_no IF NOT EXISTS FOR (e:EWayBill) REQUIRE e.ewb_no IS UNIQUE;",
        "CREATE CONSTRAINT payment_challan IF NOT EXISTS FOR (p:Payment) REQUIRE p.challan_no IS UNIQUE;",
        "CREATE INDEX invoice_period IF NOT EXISTS FOR (i:Invoice) ON (i.period);",
        "CREATE INDEX taxpayer_risk IF NOT EXISTS FOR (t:Taxpayer) ON (t.risk_score);",
        "CREATE INDEX return_type IF NOT EXISTS FOR (r:Return) ON (r.return_type);"
    ]
    
    for q in constraints:
        try:
             run_write_query(q)
        except Exception as e:
             # Just skip if constraint logic is unsupported by an old/free neo4j instance
             print(f"  ⚠️ Skipped constraint/index: {str(e)}")


    print(f"\n🚀 Loading Data (Steps 3+)...")

    # 1. Taxpayers
    taxpayers = data.get("taxpayers", [])
    print(f"Loading {len(taxpayers)} Taxpayers...")
    
    for t in taxpayers:
        run_write_query("""
            MERGE (v:Taxpayer {gstin: $gstin})
            SET v.name              = $name,
                v.state             = $state,
                v.turnover          = $turnover,
                v.business_type     = $business_type,
                v.registration_date = $registration_date,
                v.risk_score        = $risk_score,
                v.risk_level        = CASE 
                                        WHEN $risk_score > 0.6 THEN 'HIGH' 
                                        WHEN $risk_score > 0.3 THEN 'MEDIUM' 
                                        ELSE 'LOW' 
                                      END
        """, t)


    # 2. Add Ghost fraud company (for testing fraud rings)
    run_write_query("""
        MERGE (g:Taxpayer {gstin: "29FRAUD0000F0Z0"})
        SET g.name              = "Ghost Company LLC",
            g.state             = "Karnataka",
            g.turnover          = 0,
            g.business_type     = "Trader",
            g.registration_date = "2024-09-01",
            g.risk_score        = 0.98,
            g.risk_level        = "HIGH"
    """)


    # 3. GSTR-1 -> Generates Invoices, IRNs, and Return Nodes
    invoices = data.get("gstr1", [])
    print(f"Loading {len(invoices)} Invoices (GSTR-1)...")
    
    for inv in invoices:
        # Create Invoice
        run_write_query("""
            MERGE (i:Invoice {invoice_no: $invoice_no})
            SET i.irn           = $irn,
                i.seller_gstin  = $seller_gstin,
                i.seller_name   = $seller_name,
                i.buyer_gstin   = $buyer_gstin,
                i.buyer_name    = $buyer_name,
                i.invoice_date  = $invoice_date,
                i.taxable_value = $taxable_value,
                i.gst_rate      = $gst_rate,
                i.igst          = $igst,
                i.cgst          = $cgst,
                i.sgst          = $sgst,
                i.total_amount  = $taxable_value + $igst + $cgst + $sgst,
                i.period        = $period,
                i.status        = $status,
                i.missing_ewb   = false,
                i.supply_type   = "Interstate"
        """, inv)

        # Create IRN
        run_write_query("""
            MERGE (irn:IRN {code: $irn})
            SET irn.invoice_no   = $invoice_no,
                irn.generated_on = $invoice_date,
                irn.seller_gstin = $seller_gstin,
                irn.registered   = true,
                irn.nic_verified = true
                
            WITH irn
            MATCH (i:Invoice {invoice_no: $invoice_no})
            MERGE (i)-[:HAS_IRN {verified: true}]->(irn)
        """, inv)
        
        # Link Seller to Invoice (ISSUED)
        run_write_query("""
            MATCH (v:Taxpayer {gstin: $seller_gstin})
            MATCH (i:Invoice {invoice_no: $invoice_no})
            MERGE (v)-[:ISSUED {date: $invoice_date, period: $period}]->(i)
        """, inv)
        
        # Link Buyer to Invoice (RECEIVED)
        run_write_query("""
            MATCH (a:Taxpayer {gstin: $buyer_gstin})
            MATCH (i:Invoice {invoice_no: $invoice_no})
            MERGE (a)-[:RECEIVED {date: $invoice_date}]->(i)
        """, inv)
        
        # GSTR-1 Return Mapping
        run_write_query("""
            MERGE (r1:Return {
                return_type: "GSTR-1",
                gstin: $seller_gstin,
                period: $period
            })
            SET r1.total_taxable_value = coalesce(r1.total_taxable_value, 0) + $taxable_value,
                r1.total_tax = coalesce(r1.total_tax, 0) + $igst + $cgst + $sgst,
                r1.status = "Filed"
                
            WITH r1
            MATCH (i:Invoice {invoice_no: $invoice_no})
            MERGE (i)-[:REPORTED_IN {match_status: "Matched"}]->(r1)
        """, inv)


    # 4. E-Way Bills
    ewbs = data.get("ewaybill", [])
    print(f"Loading {len(ewbs)} E-Way Bills...")
    
    for e in ewbs:
        is_missing = str(e.get('ewb_no')).upper() == "MISSING"
        valid = bool(e.get('valid', not is_missing))
        
        if not is_missing:
            run_write_query("""
                MERGE (ewb:EWayBill {ewb_no: $ewb_no})
                SET ewb.invoice_no    = $invoice_no,
                    ewb.vehicle_no    = $vehicle_no,
                    ewb.from_state    = $from_state,
                    ewb.to_state      = $to_state,
                    ewb.goods_value   = $goods_value,
                    ewb.valid         = true
                    
                WITH ewb
                MATCH (i:Invoice {invoice_no: $invoice_no})
                MERGE (i)-[:COVERED_BY {goods_match: true}]->(ewb)
            """, e)
        else:
             run_write_query("""
                MATCH (i:Invoice {invoice_no: $invoice_no})
                SET i.missing_ewb = true
             """, e)


    # 5. GSTR-2B -> Inject discrepancies directly onto the Invoices & mapping nodes
    gstr2b = data.get("gstr2b", [])
    print(f"Loading {len(gstr2b)} GSTR-2B Records...")
    
    for g2b in gstr2b:
        mismatch_flag = bool(g2b.get('has_mismatch', False))
        mismatch_typ = g2b.get('mismatch_type') or None
        
        # Step 1: Create GSTR-2B Return node + REFLECTED_IN link (always)
        run_write_query("""
            MERGE (r2:Return {
                return_type: "GSTR-2B",
                gstin: $recipient_gstin,
                period: $period
            })
            SET r2.taxable_value = coalesce(r2.taxable_value, 0) + $taxable_value,
                r2.itc_eligible = coalesce(r2.itc_eligible, 0) + $itc_eligible,
                r2.status = "Auto-Generated"
                
            WITH r2
            MATCH (i:Invoice {invoice_no: $invoice_no})
            SET i.has_mismatch = $has_mismatch
            MERGE (i)-[:REFLECTED_IN {matched: not $has_mismatch}]->(r2)
        """, {**g2b, 'has_mismatch': mismatch_flag})

        # Step 2: Only create HAS_MISMATCH relationship when there IS a mismatch
        if mismatch_flag and mismatch_typ:
            run_write_query("""
                MATCH (i:Invoice {invoice_no: $invoice_no})
                MATCH (r2:Return {return_type: "GSTR-2B", gstin: $recipient_gstin, period: $period})
                MERGE (i)-[:HAS_MISMATCH {type: $mismatch}]->(r2)
            """, {**g2b, 'mismatch': mismatch_typ})

    
    # Optional: Mock a circular fraud ring explicitly based on logic from step 9
    print("🕸️ Planting a simulated fraud ring for validation...")
    if len(taxpayers) > 2:
         target_a = taxpayers[0]['gstin']
         target_b = taxpayers[1]['gstin']
         
         run_write_query("""
            MATCH (v:Taxpayer {gstin: $tA})
            MATCH (b:Taxpayer {gstin: $tB})
            MATCH (g:Taxpayer {gstin: "29FRAUD0000F0Z0"})
            
            MERGE (v)-[:TRANSACTS_WITH {flagged: false}]->(b)
            MERGE (b)-[:TRANSACTS_WITH {flagged: true}]->(g)
            MERGE (g)-[:TRANSACTS_WITH {flagged: true}]->(v)
         """, {'tA': target_a, 'tB': target_b})


    print("\n✅ Database Load Complete!")
    close_driver()

if __name__ == "__main__":
    load_data()
