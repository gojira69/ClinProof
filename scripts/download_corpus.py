#!/usr/bin/env python3
"""
Download MedRAG corpus chunks from HuggingFace.
Downloads: textbooks, statpearls (most useful for medical QA)
Saves as JSONL chunks ready for BM25Retriever and DenseRetriever.
"""
import os, sys, json, time
sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/clinproof")

CORPUS_DIR = "/mnt/d/Harsha/AoLM/project/clinproof/data/corpus"

# Each corpus is its own HF repo under MedRAG org
CORPUS_HF_MAP = {
    "textbooks":  "MedRAG/textbooks",
    "statpearls": "MedRAG/statpearls",
    "wikipedia":  "MedRAG/wikipedia",   # large ~8GB, optional
    "pubmed":     "MedRAG/pubmed",      # very large ~30GB, skip by default
}
CORPORA = ["textbooks", "statpearls"]  # default: textbooks + StatPearls

def download_corpus(corpus_name: str):
    out_dir = os.path.join(CORPUS_DIR, corpus_name, "chunk")
    os.makedirs(out_dir, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
    if existing:
        print(f"[{corpus_name}] Already downloaded: {len(existing)} chunk files")
        return

    hf_repo = CORPUS_HF_MAP.get(corpus_name, f"MedRAG/{corpus_name}")
    print(f"[{corpus_name}] Downloading from {hf_repo} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_repo, split="train")
        print(f"[{corpus_name}] Downloaded {len(ds)} documents. Saving chunks...")

        # Save in chunks of 1000 docs per JSONL file (matches BM25Retriever format)
        chunk_size = 1000
        chunk_idx = 0
        buf = []
        for doc in ds:
            # Normalize fields to: {title, content, PMID (if any)}
            buf.append({
                "title":   doc.get("title", ""),
                "content": doc.get("content", doc.get("text", "")),
                "PMID":    str(doc.get("PMID", doc.get("pmid", ""))),
                "source":  corpus_name,
            })
            if len(buf) >= chunk_size:
                fname = os.path.join(out_dir, f"chunk_{chunk_idx:05d}.jsonl")
                with open(fname, "w") as f:
                    for item in buf:
                        f.write(json.dumps(item) + "\n")
                chunk_idx += 1
                buf = []
                if chunk_idx % 10 == 0:
                    print(f"  [{corpus_name}] Saved {chunk_idx * chunk_size:,} docs...")

        if buf:
            fname = os.path.join(out_dir, f"chunk_{chunk_idx:05d}.jsonl")
            with open(fname, "w") as f:
                for item in buf:
                    f.write(json.dumps(item) + "\n")

        total = chunk_idx * chunk_size + len(buf)
        print(f"[{corpus_name}] Done: {total:,} docs in {chunk_idx+1} files → {out_dir}")

    except Exception as e:
        print(f"[{corpus_name}] ERROR: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    corpora = sys.argv[1:] if len(sys.argv) > 1 else CORPORA
    t0 = time.time()
    for c in corpora:
        download_corpus(c)
    elapsed = (time.time() - t0) / 60
    print(f"\nAll done in {elapsed:.1f} min. Corpus at: {CORPUS_DIR}")
    print("BM25 index will be built automatically on first retrieval.")
