import subprocess
with open("experiments/tier2_audit/run_item3.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("export CUDA_VISIBLE_DEVICES=0\n")
    f.write("python3 -u experiments/tier2_audit/item3.py > experiments/tier2_audit/item3_log.txt 2>&1\n")

import os
os.chmod("experiments/tier2_audit/run_item3.sh", 0o755)

subprocess.Popen(
    ["./experiments/tier2_audit/run_item3.sh"],
    start_new_session=True
)
print("Item 3 detached execution started.")
