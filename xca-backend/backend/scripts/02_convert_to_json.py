import pandas as pd
import json
import os
import math

# Use paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, '..', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'converted')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════
#  UNIVERSAL CONVERTER FUNCTION
# ════════════════════════════════════════════════════════

def convert_file_to_json(filename):
    """
    Reads ANY file (Excel, CSV, JSON) and returns
    a clean Python list of dictionaries.
    """
    filepath = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}, skipping.")
        return None
        
    ext = filename.split('.')[-1].lower()

    print(f"  📂 Reading: {filename} [{ext.upper()}]")

    data = None
    if ext in ['xlsx', 'xls']:
        # ── Excel → DataFrame → list of dicts
        df = pd.read_excel(filepath)
        data = df.to_dict(orient='records')

    elif ext == 'csv':
        # ── CSV → DataFrame → list of dicts
        df = pd.read_csv(filepath)
        data = df.to_dict(orient='records')

    elif ext == 'json':
        # ── JSON → already a list, just load it
        with open(filepath, 'r') as f:
            data = json.load(f)
        # If it's a dict (not list), wrap it
        if isinstance(data, dict):
            data = [data]

    else:
        print(f"  ⚠️  Unknown format: {ext}, skipping.")
        return None

    # ── Clean NaN values (Excel often has empty cells = NaN)
    if data:
        data = clean_nans(data)
        print(f"  ✅ Converted {len(data)} records")
        
    return data


def clean_nans(data):
    """Replace NaN / None / 'nan' strings with None cleanly."""
    cleaned = []
    for row in data:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None
            elif v in ['nan', 'NaN']:
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


def save_json(data, output_filename):
    """Save data as a JSON file in the converted folder."""
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  💾 Saved → {output_path}\n")


# ════════════════════════════════════════════════════════
#  CONVERT ALL FILES
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  GST DATA PIPELINE — File Converter")
    print("=" * 50)

    files_to_convert = {
        "taxpayers.json": "taxpayers_clean.json",
        "gstr1.xlsx":     "gstr1_clean.json",
        "gstr2b.csv":     "gstr2b_clean.json",
        "ewayhill.csv":    "ewayhill_clean.json",
    }

    all_converted = {}

    for input_file, output_file in files_to_convert.items():
        print(f"\n🔄 Processing: {input_file}")
        data = convert_file_to_json(input_file)
        if data is not None:
            save_json(data, output_file)
            all_converted[output_file] = data

    # ── Also save one MASTER JSON combining everything ──────
    master = {
        "taxpayers": all_converted.get("taxpayers_clean.json", []),
        "gstr1":     all_converted.get("gstr1_clean.json", []),
        "gstr2b":    all_converted.get("gstr2b_clean.json", []),
        "ewaybill":  all_converted.get("ewayhill_clean.json", [])
    }
    
    master_path = os.path.join(OUTPUT_DIR, "MASTER.json")
    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)

    print("=" * 50)
    print("✅ ALL FILES CONVERTED")
    print(f"📦 Master JSON saved: {master_path}")
    print(f"   - Taxpayers : {len(master['taxpayers'])}")
    print(f"   - GSTR-1    : {len(master['gstr1'])}")
    print(f"   - GSTR-2B   : {len(master['gstr2b'])}")
    print(f"   - e-Way Bill: {len(master['ewaybill'])}")
    print("=" * 50)
