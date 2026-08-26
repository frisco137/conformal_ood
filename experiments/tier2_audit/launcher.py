import subprocess
import sys
import os

with open(f"experiments/tier2_audit/phase3_out_{sys.argv[1]}.txt", "w") as out:
    subprocess.Popen(
        ["python3", "-u", "experiments/tier2_audit/audit_phase3.py", sys.argv[1]],
        stdout=out,
        stderr=out,
        start_new_session=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    )
print("Launched successfully")
