import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.pubmed_dense_retriever import PubMedDenseRetriever
from src.utils.paths import project_path

def test_retrievers():
    query = "Have mutations in the Polycomb group been found in human diseases?"
    print(f"\n======== Testing Local Retrievers ========")
    print(f"Query: '{query}'\n")

    # 1. Test BM25
    corpus_dir = project_path("data", "corpus")
    print("--- 1. Testing BM25 Retriever ---")
    t0 = time.time()
    try:
        bm25 = BM25Retriever(corpus_dir=corpus_dir, corpus_name="textbooks", cache=True)
        if bm25.bm25:
            t1 = time.time()
            docs, scores = bm25.retrieve(query, k=10)
            t2 = time.time()
            print(f"[BM25] Init time: {t1-t0:.2f}s | Search time: {t2-t1:.4f}s")
            print(f"[BM25] Found {len(docs)} documents.")
            for i, (d, s) in enumerate(zip(docs, scores)):
                print(f"       [{i+1}] Score {s:.2f} | Title: {d.get('title', 'N/A')[:60]}... | PMID: {d.get('PMID', 'N/A')}")
                print(f"               Abstract: {d.get('content', 'N/A')[:150]}...")
        else:
            print("[BM25] Index could not be loaded or is empty.")
    except Exception as e:
        print(f"[BM25] Error: {e}")


    # 2. Test Local MedCPT FAISS
    print("\n--- 2. Testing Local MedCPT (FAISS) Retriever ---")
    config = {
        "pubmed": {
            "local_index_dir": project_path("data", "pubmed_index"),
            "cache_dir": project_path("data", "pubmed_cache")
        }
    }
    
    t0 = time.time()
    try:
        medcpt = PubMedDenseRetriever(config)
        t1 = time.time()
        
        # We need to trigger the lazy loading of encoders before we time the pure search speed
        medcpt._load_query_encoder()
        t2 = time.time()
        
        docs, scores = medcpt.retrieve(query, k=10)
        t3 = time.time()
        
        print(f"[MedCPT] Init time (FAISS): {t1-t0:.2f}s | Encoder load time: {t2-t1:.2f}s | Search time: {t3-t2:.4f}s")
        print(f"[MedCPT] Found {len(docs)} documents.")
        for i, (d, s) in enumerate(zip(docs, scores)):
            print(f"         [{i+1}] Score {s:.2f} | Title: {d.get('title', 'N/A')[:60]}... | PMID: {d.get('pmid', 'N/A')} | DB: {d.get('source', 'N/A')}")
            print(f"                 Abstract: {d.get('content', 'N/A')[:150]}...")
    except Exception as e:
        print(f"[MedCPT] Error: {e}")

if __name__ == "__main__":
    test_retrievers()
