import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing import of backend.main")
try:
    import backend.main
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    traceback.print_exc()
