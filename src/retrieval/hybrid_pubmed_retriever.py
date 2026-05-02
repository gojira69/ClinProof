"""
ClinProof — Hybrid PubMed Retriever
====================================
Fuses BM25 (SQLite FTS5, keyword) + FAISS IVF-PQ (dense, semantic)
via Reciprocal Rank Fusion (RRF), then cross-encoder re-ranks.

Why hybrid beats either alone:
- BM25 wins on BioASQ-style factoid yes/no questions (keyword precision)
- Dense wins on SciFact/HealthFC-style semantic claim verification
- RRF fusion consistently +5-10% over either individually (BEIR benchmark)

No threading — must stay in main thread (GPU ops).
"""
from __future__ import annotations

import csv
import logging
import os
import re
import sqlite3

import numpy as np
import torch

log = logging.getLogger("hybrid_pubmed_retriever")

_DATA = "/mnt/d/Harsha/AoLM/project/data"
_CLINPROOF = "/mnt/d/Harsha/AoLM/project/clinproof"

EMBEDDING_FILE = f"{_DATA}/PubMedBERT_embeddings_float16_2024.npy"
METADATA_FILE  = f"{_DATA}/pubmed_landscape_data_2024_v2.csv"
ABSTRACTS_FILE = f"{_DATA}/pubmed_landscape_abstracts_2024.csv"
OFFSETS_FILE   = f"{_DATA}/pubmed_landscape_abstracts_2024_offsets.npy"
FAISS_INDEX    = f"{_DATA}/pubmed_faiss_ivfpq.index"
FTS_DB         = f"{_CLINPROOF}/data/pubmed_fts.db"

QUERY_MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
RERANKER_ID    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF constant — 60 is the standard value from the original RRF paper
RRF_K = 60


def _rrf_merge(
    list_a: list[dict], list_b: list[dict],
    weight_a: float = 1.0, weight_b: float = 1.0,
) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked lists.
    De-duplicates by PMID, returns merged list ordered by RRF score (desc).
    """
    scores: dict[str, float] = {}
    by_pmid: dict[str, dict] = {}

    for rank, doc in enumerate(list_a):
        pmid = str(doc.get("pmid", doc.get("title", rank)))
        scores[pmid] = scores.get(pmid, 0.0) + weight_a / (RRF_K + rank + 1)
        by_pmid.setdefault(pmid, doc)

    for rank, doc in enumerate(list_b):
        pmid = str(doc.get("pmid", doc.get("title", rank)))
        scores[pmid] = scores.get(pmid, 0.0) + weight_b / (RRF_K + rank + 1)
        by_pmid.setdefault(pmid, doc)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_pmid[pmid] for pmid, _ in ordered]


class HybridPubMedRetriever:
    """
    Two-source retriever (main thread only):

    Source A — BM25 (SQLite FTS5, 47GB DB, near-instant keyword search)
    Source B — FAISS IVF-PQ (dense semantic, 1.7GB index in RAM, ~100ms)

    Merged via RRF, then cross-encoder re-ranked.
    """

    def __init__(
        self,
        top_k_bm25: int = 50,
        top_k_dense: int = 50,
        top_k_final: int = 20,
        device: str | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
    ) -> None:
        self.top_k_bm25  = top_k_bm25
        self.top_k_dense = top_k_dense
        self.top_k_final = top_k_final
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bm25    = use_bm25 and os.path.exists(FTS_DB)
        self.use_dense   = use_dense and os.path.exists(FAISS_INDEX)

        # Lazy-loaded state
        self._faiss       = None
        self._metadata    = None
        self._offsets     = None
        self._abs_file    = None
        self._abs_col_idx: int | None = None
        self._query_tok   = None
        self._query_model = None
        self._reranker    = None

    # ── BM25 ────────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str) -> list[dict]:
        clean = re.sub(r"[^a-zA-Z0-9 ]", "", query)
        words = [w for w in clean.split() if len(w) > 3]
        if not words:
            return []
        match_str = " OR ".join(words)
        try:
            conn = sqlite3.connect(FTS_DB, timeout=15)
            rows = conn.execute(
                "SELECT pmid, abstract, rank FROM pubmed_fts "
                "WHERE pubmed_fts MATCH ? ORDER BY rank LIMIT ?",
                (match_str, self.top_k_bm25),
            ).fetchall()
            conn.close()
        except Exception as e:
            log.warning(f"BM25 search error: {e}")
            return []
        return [
            {"pmid": str(r[0]), "content": r[1] or "", "source": "bm25",
             "title": f"PubMed PMID:{r[0]}"}
            for r in rows
        ]

    # ── Dense / FAISS ────────────────────────────────────────────────────────

    def _load_faiss(self) -> bool:
        if self._faiss is not None:
            return True
        if not os.path.exists(FAISS_INDEX):
            return False
        try:
            import faiss
            log.info("Loading FAISS index …")
            self._faiss = faiss.read_index(FAISS_INDEX)
            return True
        except Exception as e:
            log.warning(f"FAISS load failed: {e}")
            return False

    def _load_metadata(self) -> None:
        if self._metadata is not None:
            return
        import pandas as pd
        log.info("Loading metadata (4 cols) …")
        self._metadata = pd.read_csv(
            METADATA_FILE,
            usecols=["PMID", "Title", "Journal", "Year"],
            dtype={"PMID": "Int32", "Year": "Int16"},
            low_memory=True,
        )
        self._metadata.columns = self._metadata.columns.str.strip()

    def _load_abstracts_index(self) -> None:
        if self._offsets is not None:
            return
        self._offsets  = np.load(OFFSETS_FILE)
        self._abs_file = open(ABSTRACTS_FILE, "rb")
        self._abs_file.seek(int(self._offsets[0]))
        hdr = self._abs_file.readline().decode("utf-8", errors="replace")
        cols = [c.strip().strip('"') for c in hdr.rstrip("\r\n").split(",")]
        self._abs_col_idx = next(
            (i for i, c in enumerate(cols)
             if c.lower() in ("abstract", "abstracttext", "text", "abstracts")),
            None,
        ) or next(i for i, c in enumerate(cols) if "pmid" not in c.lower())

    def _load_query_model(self) -> None:
        if self._query_model is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        log.info(f"Loading PubMedBERT → {self.device}")
        self._query_tok   = AutoTokenizer.from_pretrained(QUERY_MODEL_ID)
        self._query_model = (
            AutoModel.from_pretrained(QUERY_MODEL_ID).to(self.device).eval()
        )

    def _load_reranker(self) -> None:
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_ID, device=self.device)
        except Exception as e:
            log.warning(f"Re-ranker unavailable: {e}")

    def _embed_query(self, query: str) -> np.ndarray:
        inputs = self._query_tok(
            query, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self._query_model(**inputs)
        vec = out.last_hidden_state[0, -1, :].half()
        vec = vec / (vec.norm() + 1e-9)
        return vec.float().cpu().numpy()

    def _fetch_abstract(self, row_idx: int) -> str:
        if row_idx < 1 or row_idx >= len(self._offsets):
            return ""
        self._abs_file.seek(int(self._offsets[row_idx]))
        line = self._abs_file.readline().decode("utf-8", errors="replace").rstrip("\r\n")
        try:
            parts = next(csv.reader([line]))
            return parts[self._abs_col_idx] if self._abs_col_idx < len(parts) else ""
        except Exception:
            return line

    def _dense_search(self, query: str) -> list[dict]:
        self._load_query_model()
        if not self._load_faiss():
            return []
        self._load_metadata()
        self._load_abstracts_index()

        q_vec = self._embed_query(query).reshape(1, -1).astype("float32")
        scs, idxs = self._faiss.search(q_vec, self.top_k_dense)

        docs: list[dict] = []
        for row_idx, sc in zip(idxs[0], scs[0]):
            if row_idx < 0:
                continue
            try:
                meta     = self._metadata.iloc[int(row_idx)]
                abstract = self._fetch_abstract(int(row_idx) + 1)
            except Exception:
                continue
            docs.append({
                "pmid":    str(meta.get("PMID", row_idx)),
                "title":   f"{meta.get('Title','')} ({meta.get('Journal','')}, {meta.get('Year','')})",
                "content": abstract,
                "source":  "dense",
            })
        return docs

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def _rerank(self, query: str, docs: list[dict], k: int) -> list[dict]:
        if not docs:
            return []
        self._load_reranker()
        if self._reranker is None:
            return docs[:k]
        pairs  = [(query, d["content"][:512]) for d in docs]
        re_scs = self._reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(docs, re_scs.tolist()), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:k]]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int | None = None, **_) -> tuple[list[dict], list[float]]:
        final_k = k or self.top_k_final

        bm25_docs  = self._bm25_search(query) if self.use_bm25 else []
        dense_docs = self._dense_search(query) if self.use_dense else []

        if not bm25_docs and not dense_docs:
            return [], []

        # RRF fusion: BM25 gets slightly higher weight for medical factoid tasks
        merged = _rrf_merge(bm25_docs, dense_docs, weight_a=1.2, weight_b=1.0)

        # Cross-encoder re-ranking on the fused candidates
        reranked = self._rerank(query, merged, k=final_k)
        scores   = list(range(len(reranked), 0, -1))
        return reranked, [float(s) for s in scores]

    def multi_retrieve(
        self,
        queries: list[str],
        k: int | None = None,
    ) -> tuple[list[dict], list[float]]:
        """Multi-query retrieval with RRF fusion across all sub-queries.

        Instead of one big question, fire each atomic query (entity / proposition)
        independently through BM25+Dense, then RRF-merge all per-query results.
        This dramatically improves recall for factoid and claim-verification tasks.

        Args:
            queries : list of sub-queries [original_question, entity1, entity2, ...]
            k       : final number of docs to return after cross-encoder re-ranking
        """
        final_k = k or self.top_k_final
        if not queries:
            return [], []

        # Deduplicate and cap queries
        seen_q: set[str] = set()
        clean_queries: list[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                clean_queries.append(q)
        clean_queries = clean_queries[:8]  # cap at 8 sub-queries to avoid slowdown

        # Per-query retrieval (BM25 + Dense separately)
        all_bm25:  list[dict] = []
        all_dense: list[dict] = []
        for subq in clean_queries:
            if self.use_bm25:
                all_bm25.extend(self._bm25_search(subq))
            if self.use_dense:
                all_dense.extend(self._dense_search(subq))

        if not all_bm25 and not all_dense:
            return [], []

        # Global RRF merge across all per-query results
        # BM25 weight slightly higher for keyword precision on medical queries
        merged = _rrf_merge(all_bm25, all_dense, weight_a=1.2, weight_b=1.0)

        # Cross-encoder re-ranks using the ORIGINAL question (most informative)
        anchor_query = clean_queries[0]
        reranked = self._rerank(anchor_query, merged, k=final_k)
        scores   = list(range(len(reranked), 0, -1))
        return reranked, [float(s) for s in scores]



    def __del__(self) -> None:
        if self._abs_file:
            try:
                self._abs_file.close()
            except Exception:
                pass
