#!/usr/bin/env python3
"""Launch KG ingestion as a subprocess."""
import subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

cfg = sys.argv[1] if len(sys.argv) > 1 else project_path("config", "default.yaml")
log_dir = project_path("logs")
os.makedirs(log_dir, exist_ok=True)
src = project_path("src", "kg")

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
