"""
ClinProof — PubMed Embedding Retriever (FAISS IVF-PQ + cross-encoder re-ranking)
==================================================================================
- Embeddings   : 23M float16 vectors on disk (mmap, never loaded to RAM)
- ANN search   : FAISS IVF-PQ index loaded into RAM (~1.5 GB), sub-100ms per query
- Abstracts    : Random row access via byte-offset index (0 RAM overhead)
- Metadata     : 4-col pandas DF (~2 GB)
- Re-ranker    : CrossEncoder on GPU
- Threading    : None — all GPU ops in calling thread

One-time setup (run in order):
    python scripts/build_abstracts_offset_index.py   # ~5-10 min
    python scripts/build_faiss_ivf_index.py          # ~20-30 min
"""
from __future__ import annotations

import csv
import logging
import os
import numpy as np
import torch
from pathlib import Path
import sys

log = logging.getLogger("pubmed_embedding_retriever")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA = project_path("data")
EMBEDDING_FILE = f"{_DATA}/PubMedBERT_embeddings_float16_2024.npy"
METADATA_FILE  = f"{_DATA}/pubmed_landscape_data_2024_v2.csv"
ABSTRACTS_FILE = f"{_DATA}/pubmed_landscape_abstracts_2024.csv"
OFFSETS_FILE   = f"{_DATA}/pubmed_landscape_abstracts_2024_offsets.npy"
FAISS_INDEX    = f"{_DATA}/pubmed_faiss_ivfpq.index"

QUERY_MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
RERANKER_ID    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# GPU ANN chunk (fallback if FAISS index not built yet)
ANN_CHUNK = 1_000_000


class PubMedEmbeddingRetriever:
    """
    Two-stage retriever (main thread only, no threading).

    Stage 1: FAISS IVF-PQ ANN  ~50–150 ms
    Stage 2: CrossEncoder re-ranking on GPU
    """

    def __init__(self, top_k_ann: int = 100, top_k_final: int = 20,
                 device: str | None = None) -> None:
        self.top_k_ann   = top_k_ann
        self.top_k_final = top_k_final
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._faiss_index = None
        self._embeddings  = None   # mmap fallback
        self._metadata    = None
        self._offsets     = None
        self._abs_file    = None
        self._abs_col_idx: int | None = None
        self._query_tok   = None
        self._query_model = None
        self._reranker    = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _load_faiss(self) -> bool:
        """Load FAISS index if available. Returns True on success."""
        if self._faiss_index is not None:
            return True
        if not os.path.exists(FAISS_INDEX):
            return False
        try:
            import faiss
            log.info(f"Loading FAISS index: {FAISS_INDEX}")
            self._faiss_index = faiss.read_index(FAISS_INDEX)
            log.info(f"  ntotal={self._faiss_index.ntotal:,}  nprobe={self._faiss_index.nprobe}")
            return True
        except Exception as e:
            log.warning(f"FAISS load failed ({e}) — falling back to GPU mmap scan")
            return False

    def _load_embeddings_mmap(self) -> None:
        """Fallback: mmap embeddings for GPU chunked scan."""
        if self._embeddings is not None:
            return
        log.warning("FAISS index not available — using slow GPU mmap scan. "
                    "Run scripts/build_faiss_ivf_index.py to fix this.")
        self._embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

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
        if not os.path.exists(OFFSETS_FILE):
            raise FileNotFoundError(
                f"Byte-offset index missing: {OFFSETS_FILE}\n"
                "Run: python scripts/build_abstracts_offset_index.py"
            )
        self._offsets  = np.load(OFFSETS_FILE)
        self._abs_file = open(ABSTRACTS_FILE, "rb")
        # Parse header
        self._abs_file.seek(int(self._offsets[0]))
        hdr_line = self._abs_file.readline().decode("utf-8", errors="replace")
        cols     = [c.strip().strip('"') for c in hdr_line.rstrip("\r\n").split(",")]
        self._abs_col_idx = next(
            (i for i, c in enumerate(cols)
             if c.lower() in ("abstract", "abstracttext", "text", "abstracts")),
            None,
        )
        if self._abs_col_idx is None:
            self._abs_col_idx = next(
                i for i, c in enumerate(cols) if "pmid" not in c.lower()
            )

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
            self._reranker = None

    # ── Core ops ──────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        inputs = self._query_tok(
            query, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self._query_model(**inputs)
        vec = out.last_hidden_state[0, -1, :].half()
        vec = vec / (vec.norm() + 1e-9)
        return vec.float().cpu().numpy()   # FAISS needs float32 on CPU

    def _faiss_search(self, query_vec: np.ndarray) -> tuple[list[int], list[float]]:
        q = query_vec.reshape(1, -1).astype("float32")
        scs, idxs = self._faiss_index.search(q, self.top_k_ann)
        # Filter out -1 (padding) returned by FAISS
        valid = [(int(i), float(s)) for i, s in zip(idxs[0], scs[0]) if i >= 0]
        return [v[0] for v in valid], [v[1] for v in valid]

    def _gpu_topk(self, query_vec: np.ndarray) -> tuple[list[int], list[float]]:
        """Slow fallback: chunked GPU dot-product over mmap embeddings."""
        q   = torch.from_numpy(query_vec.astype("float16")).to(self.device)
        n   = self._embeddings.shape[0]
        top_scores  = torch.full((self.top_k_ann,), float("-inf"), device=self.device)
        top_indices = torch.zeros(self.top_k_ann, dtype=torch.long, device=self.device)
        for start in range(0, n, ANN_CHUNK):
            end   = min(start + ANN_CHUNK, n)
            chunk = torch.from_numpy(np.array(self._embeddings[start:end])).half().to(self.device)
            scs   = chunk @ q
            all_scs = torch.cat([top_scores, scs])
            all_idx = torch.cat([top_indices, torch.arange(start, end, device=self.device)])
            vals, sel = torch.topk(all_scs, self.top_k_ann)
            top_scores, top_indices = vals, all_idx[sel]
            del chunk, scs
        return top_indices.cpu().tolist(), top_scores.cpu().float().tolist()

    def _fetch_abstract(self, row_idx: int) -> str:
        """Seek to data row `row_idx` (1-based) in abstracts CSV."""
        if row_idx < 1 or row_idx >= len(self._offsets):
            return ""
        self._abs_file.seek(int(self._offsets[row_idx]))
        line = self._abs_file.readline().decode("utf-8", errors="replace").rstrip("\r\n")
        try:
            parts = next(csv.reader([line]))
            return parts[self._abs_col_idx] if self._abs_col_idx < len(parts) else ""
        except Exception:
            return line

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int | None = None, **_) -> tuple[list[dict], list[float]]:
        final_k = k or self.top_k_final

        self._load_metadata()
        self._load_abstracts_index()
        self._load_query_model()

        q_vec = self._embed_query(query)

        # Stage 1 — ANN (FAISS preferred, GPU mmap fallback)
        if self._load_faiss():
            ann_idxs, ann_scs = self._faiss_search(q_vec)
        else:
            self._load_embeddings_mmap()
            ann_idxs, ann_scs = self._gpu_topk(q_vec)

        # Build candidate docs
        candidates: list[dict] = []
        for row_idx, sc in zip(ann_idxs, ann_scs):
            try:
                meta     = self._metadata.iloc[int(row_idx)]
                abstract = self._fetch_abstract(int(row_idx) + 1)
            except Exception:
                continue
            candidates.append({
                "_sc":   sc,
                "title":   f"{meta.get('Title','')} ({meta.get('Journal','')}, {meta.get('Year','')})",
                "content": abstract,
                "pmid":    str(meta.get("PMID", row_idx)),
                "source":  "pubmed_embedding",
            })

        if not candidates:
            return [], []

        # Stage 2 — Cross-encoder re-ranking
        self._load_reranker()
        if self._reranker is not None:
            pairs   = [(query, c["content"][:512]) for c in candidates]
            re_scs  = self._reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            ranked  = sorted(zip(candidates, re_scs.tolist()), key=lambda x: x[1], reverse=True)
            docs    = [d for d, _ in ranked[:final_k]]
            scs     = [float(s) for _, s in ranked[:final_k]]
        else:
            docs = candidates[:final_k]
            scs  = [c["_sc"] for c in docs]

        for d in docs:
            d.pop("_sc", None)
        return docs, scs

    def __del__(self) -> None:
        if self._abs_file:
            try:
                self._abs_file.close()
            except Exception:
                pass
