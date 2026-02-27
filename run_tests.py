import sys
import subprocess
import json

p = subprocess.run([sys.executable, 'backend/classifier/test_classifier.py'], capture_output=True, text=True, encoding='utf-8')
with open('test_result.json', 'w', encoding='utf-8') as f:
    json.dump({'stdout': p.stdout, 'stderr': p.stderr, 'code': p.returncode}, f)
