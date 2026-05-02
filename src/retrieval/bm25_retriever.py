"""
ClinProof BM25 Retriever using rank-bm25
Supports optional recency-weighted scoring (for MedChangeQA temporal evaluation).

Reference: Robertson & Zaragoza (2009) BM25.
Recency weighting: score *= (1 + alpha * norm_age) where norm_age is
  (doc_year - min_year) / (max_year - min_year) ∈ [0,1].
  More recent docs get higher weight (alpha=0 = no weighting).
"""
import os
import json
import logging
import pickle
import re
import numpy as np

log = logging.getLogger("bm25_retriever")


def _extract_year(doc: dict) -> int | None:
    """
    Try to extract a publication year from a document dict.
    Checks: doc['year'], doc['metadata']['year'], doc['published'],
    and a 4-digit year pattern in doc['title'] or doc['content'][:200].
    """
    # Direct fields
    for field in ("year", "pub_year", "published_year"):
        v = doc.get(field) or doc.get("metadata", {}).get(field)
        if v:
            try:
                y = int(str(v)[:4])
                if 1900 <= y <= 2100:
                    return y
            except (ValueError, TypeError):
                pass

    # Published date string: "2023-05-12" or "2023"
    pub = doc.get("published") or doc.get(
        "date") or doc.get("metadata", {}).get("published")
    if pub:
        m = re.search(r"(19|20)\d{2}", str(pub))
        if m:
            return int(m.group())

    # Heuristic: look in title or first 200 chars of content
    text = (doc.get("title", "") + " " + doc.get("content", "")[:200])
    years = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", text)
    if years:
        return int(years[0])

    return None


class BM25Retriever:
    def __init__(self, corpus_dir, corpus_name="textbooks", cache=True):
        self.corpus_dir = corpus_dir
        self.corpus_name = corpus_name
        self.cache_path = os.path.join(corpus_dir, f"{corpus_name}_bm25.pkl")
        self.docs, self.bm25 = [], None
        self._load_or_build(cache)
        # Pre-extract years once for recency scoring
        self._years = [_extract_year(d) for d in self.docs]
        valid_years = [y for y in self._years if y is not None]
        self._min_year = min(valid_years) if valid_years else 1990
        self._max_year = max(valid_years) if valid_years else 2025
        log.info(f"BM25 year range: {self._min_year}–{self._max_year} "
                 f"({sum(1 for y in self._years if y)} / {len(self._years)} docs have year)")

    def _load_or_build(self, cache):
        if cache and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.docs, self.bm25 = data["docs"], data["bm25"]
            log.info(f"BM25 loaded: {len(self.docs)} docs")
            return
        chunk_dir = os.path.join(self.corpus_dir, self.corpus_name, "chunk")
        if not os.path.exists(chunk_dir):
            log.warning(f"Chunk dir not found: {chunk_dir}")
            return
        from rank_bm25 import BM25Okapi
        tokenized = []
        for fname in sorted(os.listdir(chunk_dir)):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(chunk_dir, fname)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    self.docs.append(doc)
                    tokenized.append(
                        (doc.get("title", "") + " " +
                         doc.get("content", "")).lower().split()
                    )
        self.bm25 = BM25Okapi(tokenized)
        if cache:
            with open(self.cache_path, "wb") as f:
                pickle.dump({"docs": self.docs, "bm25": self.bm25}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"BM25 built: {len(self.docs)} docs")

    def _recency_weights(self, alpha: float) -> np.ndarray:
        """
        Compute a per-doc recency multiplier in [1.0, 1+alpha].
        Docs with unknown year get multiplier = 1.0 (neutral).
        """
        span = max(self._max_year - self._min_year, 1)
        weights = np.ones(len(self.docs), dtype=np.float32)
        for i, yr in enumerate(self._years):
            if yr is not None:
                norm = (yr - self._min_year) / span        # 0=oldest, 1=newest
                weights[i] = 1.0 + alpha * norm
        return weights

    def retrieve(self, query: str, k: int = 32,
                 recency_alpha: float = 0.0) -> tuple[list, list]:
        """
        Retrieve top-k documents.

        Args:
            query:          Free-text query string.
            k:              Number of results to return.
            recency_alpha:  If > 0, apply recency reweighting.
                            alpha=0.3 → mild boost, alpha=0.7 → strong boost.
                            Recommended for MedChangeQA (temporal evaluation).

        Returns:
            (docs, scores) — both sorted descending by final score.
        """
        if not self.bm25 or not self.docs:
            return [], []

        scores = self.bm25.get_scores(query.lower().split())

        if recency_alpha > 0.0:
            weights = self._recency_weights(recency_alpha)
            scores = scores * weights
            log.debug(
                f"BM25 recency reweighting applied (alpha={recency_alpha})")

        idx = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in idx], [float(scores[i]) for i in idx]
