#!/usr/bin/env python3
"""Launch KG ingestion as a subprocess."""
import subprocess, sys, os

cfg = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Harsha/AoLM/project/clinproof/config/default.yaml"
log_dir = "/mnt/d/Harsha/AoLM/project/clinproof/logs"
os.makedirs(log_dir, exist_ok=True)
src = "/mnt/d/Harsha/AoLM/project/clinproof/src/kg"

scripts = [
    # ("ingest_umls.py",   "ingest_umls.log"),
    # ("ingest_snomed.py", "ingest_snomed.log"),
    # ("ingest_rxnorm.py", "ingest_rxnorm.log"),
    ("build_graph.py",   "build_graph.log"),
]

print("Starting KG ingestion pipeline (sequential)...")
for script, logfile in scripts:
    log_path = os.path.join(log_dir, logfile)
    script_path = os.path.join(src, script)
    print(f"  Running {script} -> {log_path}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, script_path, cfg],
            stdout=lf, stderr=lf
        )
    print(f"  PID: {proc.pid} (check log: {log_path})")
    # Run sequentially - wait for each step
    ret = proc.wait()
    print(f"  Done: exit={ret}")

print("KG ingestion complete. Graph saved.")
