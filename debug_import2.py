import traceback
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import backend.main
    with open('err.json', 'w') as f:
        json.dump({"traceback": "SUCCESS"}, f)
except Exception:
    with open('err.json', 'w') as f:
        json.dump({"traceback": traceback.format_exc()}, f)
