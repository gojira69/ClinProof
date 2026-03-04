"""
ClinProof BM25 Retriever using rank-bm25
"""
import os, json, logging, pickle
import numpy as np

log = logging.getLogger("bm25_retriever")


class BM25Retriever:
    def __init__(self, corpus_dir, corpus_name="textbooks", cache=True):
        self.corpus_dir = corpus_dir
        self.corpus_name = corpus_name
        self.cache_path = os.path.join(corpus_dir, f"{corpus_name}_bm25.pkl")
        self.docs, self.bm25 = [], None
        self._load_or_build(cache)

    def _load_or_build(self, cache):
        if cache and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.docs, self.bm25 = data["docs"], data["bm25"]
            log.info(f"BM25 loaded: {len(self.docs)} docs")
            return
        chunk_dir = os.path.join(self.corpus_dir, self.corpus_name, "chunk")
        if not os.path.exists(chunk_dir):
            log.warning(f"Chunk dir not found: {chunk_dir}"); return
        from rank_bm25 import BM25Okapi
        tokenized = []
        for fname in sorted(os.listdir(chunk_dir)):
            if not fname.endswith(".jsonl"): continue
            with open(os.path.join(chunk_dir, fname)) as f:
                for line in f:
                    if not line.strip(): continue
                    doc = json.loads(line)
                    self.docs.append(doc)
                    tokenized.append((doc.get("title","")+" "+doc.get("content","")).lower().split())
        self.bm25 = BM25Okapi(tokenized)
        if cache:
            with open(self.cache_path, "wb") as f:
                pickle.dump({"docs": self.docs, "bm25": self.bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"BM25 built: {len(self.docs)} docs")

    def retrieve(self, query, k=32):
        if not self.bm25 or not self.docs: return [], []
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in idx], [float(scores[i]) for i in idx]
