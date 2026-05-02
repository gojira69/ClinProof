"""
ClinProof BM25 Retriever using rank-bm25
Supports optional recency-weighted scoring (for MedChangeQA temporal evaluation).

Reference: Robertson & Zaragoza (2009) BM25.
Recency weighting: score *= (1 + alpha * norm_age) where norm_age is
  (doc_year - min_year) / (max_year - min_year) ∈ [0,1].
  More recent docs get higher weight (alpha=0 = no weighting).

NOTE on corpus limitations
--------------------------
The textbook corpus (InternalMed_Harrison, Surgery_Schwartz, etc.) stores only
{title, content, PMID, source} per chunk — there is NO per-document date field.
The previous _extract_year heuristic therefore returned None for 100% of docs,
making recency weighting a literal no-op (all weights = 1.0).

Fix: TEXTBOOK_EDITION_YEARS maps each known title prefix to its edition year.
This assigns a corpus-level (not chunk-level) year, which is the only
meaningful year available for these static textbook chunks.  Within-book
chunks all share the same year; across-book, newer edition textbooks are
preferred over older ones.

Implication for experiments G1a-G1f: re-running C2/C3 after this fix will
produce meaningfully different scores because recency weights are no longer
uniformly 1.0.
"""
import os
import json
import logging
import pickle
import re
import numpy as np

log = logging.getLogger("bm25_retriever")


# ---------------------------------------------------------------------------
# Textbook edition years — curated lookup for the static textbook corpus.
# Key: prefix of the doc['title'] field (case-insensitive match).
# Value: edition publication year used for recency weighting.
# Source: publisher pages / Wikipedia for each textbook edition in the corpus.
# ---------------------------------------------------------------------------
TEXTBOOK_EDITION_YEARS: dict[str, int] = {
    "InternalMed_Harrison":   2022,   # Harrison's Principles, 21st ed. 2022
    "Surgery_Schwartz":       2019,   # Schwartz's Principles of Surgery, 11th ed. 2019
    "Neurology_Adams":        2019,   # Adams & Victor's Principles of Neurology, 11th ed. 2019
    "Obstentrics_Williams":   2018,   # Williams Obstetrics, 25th ed. 2018
    "Gynecology_Novak":       2019,   # Berek & Novak's Gynecology, 16th ed. 2019 (est.)
    "Pharmacology_Katzung":   2018,   # Katzung Basic & Clinical Pharmacology, 14th ed. 2018
    "Cell_Biology_Alberts":   2015,   # Molecular Biology of the Cell, 6th ed. 2015
    "Pathology_Robbins":      2020,   # Robbins & Cotran Pathologic Basis of Disease, 10th ed. 2020
    "Immunology_Janeway":     2017,   # Janeway's Immunobiology, 9th ed. 2017
    "Histology_Ross":         2020,   # Ross Histology: A Text and Atlas, 8th ed. 2020
    "Physiology_Levy":        2017,   # Berne & Levy Physiology, 7th ed. 2017
    "Pediatrics_Nelson":      2020,   # Nelson Textbook of Pediatrics, 21st ed. 2020
    "Psichiatry_DSM-5":       2013,   # DSM-5, 2013
    "Anatomy_Gray":           2016,   # Gray's Anatomy, 41st ed. 2016
    "Biochemistry_Lippinco":  2017,   # Lippincott's Biochemistry, 7th ed. 2017
    "First_Aid_Step2":        2019,   # First Aid for the USMLE Step 2, 2019
    "First_Aid_Step1":        2019,   # First Aid for the USMLE Step 1, 2019
    "Pathoma_Husain":         2015,   # Pathoma: Fundamentals of Pathology, 2015
}


def _extract_year(doc: dict) -> int | None:
    """
    Extract a publication year for a corpus document.

    Strategy (priority order):
    1. Direct metadata fields (year, pub_year, published_year, published, date).
       These exist in PubMed / dense-retrieval docs but NOT in the textbook corpus.
    2. TEXTBOOK_EDITION_YEARS lookup: match doc['title'] prefix to the curated
       table of textbook edition years.  This is the only year signal available
       for the static textbook corpus used in ablations C1-C3, F2, G1.
    3. Heuristic: regex scan for a 4-digit year in title/content[:200].
    """
    # 1. Direct metadata
    for field in ("year", "pub_year", "published_year"):
        v = doc.get(field) or doc.get("metadata", {}).get(field)
        if v:
            try:
                y = int(str(v)[:4])
                if 1900 <= y <= 2100:
                    return y
            except (ValueError, TypeError):
                pass

    pub = doc.get("published") or doc.get(
        "date") or doc.get("metadata", {}).get("published")
    if pub:
        m = re.search(r"(19|20)\d{2}", str(pub))
        if m:
            return int(m.group())

    # 2. Textbook edition lookup (primary path for the static textbook corpus)
    title = doc.get("title", "")
    title_lower = title.lower()
    for prefix, year in TEXTBOOK_EDITION_YEARS.items():
        if title_lower.startswith(prefix.lower()):
            return year

    # 3. Heuristic regex in title + first 200 chars of content
    text = title + " " + doc.get("content", "")[:200]
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
