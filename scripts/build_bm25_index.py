#!/usr/bin/env python3
"""
Pre-build BM25 index from downloaded textbooks corpus.
Run this while KG ingestion is happening - it's CPU-only and independent.
"""
import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

corpus_dir = project_path("data", "corpus")

for corpus_name in ["textbooks", "statpearls"]:
    chunk_dir = os.path.join(corpus_dir, corpus_name, "chunk")
    cache_path = os.path.join(corpus_dir, f"{corpus_name}_bm25.pkl")

    if not os.path.exists(chunk_dir):
        print(f"[{corpus_name}] No chunk dir found, skipping")
        continue
    n_files = len([f for f in os.listdir(chunk_dir) if f.endswith(".jsonl")])
    if n_files == 0:
        print(f"[{corpus_name}] No JSONL files, skipping")
        continue
    if os.path.exists(cache_path):
        print(f"[{corpus_name}] BM25 index already exists at {cache_path}")
        continue

    print(f"[{corpus_name}] Building BM25 index from {n_files} JSONL files...")
    t0 = time.time()
    from src.retrieval.bm25_retriever import BM25Retriever
    BM25Retriever(corpus_dir, corpus_name=corpus_name, cache=True)
    elapsed = time.time() - t0
    print(f"[{corpus_name}] BM25 index built in {elapsed:.1f}s -> {cache_path}")

print("Done. BM25 indexes ready.")
