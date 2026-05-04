# """
# ClinProof PubMed/PMC Dense Retriever
# Conditionally fetches PMC full texts or PubMed abstracts via Entrez E-utilities,
# chunks them, encodes with MedCPT-Query-Encoder, and retrieves using FAISS/Cosine.
# """
# import os, json, logging, hashlib, time, re
# import numpy as np
# import torch
# import requests

# log = logging.getLogger("pubmed_dense_retriever")

# ENTREZ_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
# ENTREZ_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# QUERY_MODEL   = "ncbi/MedCPT-Query-Encoder"
# ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


# class PubMedDenseRetriever:
#     """
#     On-the-fly PMC/PubMed retriever using MedCPT:
#       1. Searches PMC (full text) or PubMed for top-N IDs relevant to the query
#       2. Fetches full texts/abstracts
#       3. Chunks texts into fixed sizes (e.g. 200 words)
#       4. Encodes chunks with MedCPT-Article-Encoder
#       5. Encodes query with MedCPT-Query-Encoder
#       6. Returns cosine-similarity ranked chunks
#     """

#     def __init__(self, config):
#         pm_cfg = config.get("pubmed", {})
#         self.email      = pm_cfg.get("email", "clinproof@example.com")
#         self.api_key    = pm_cfg.get("api_key", "")
#         self.max_results= pm_cfg.get("max_results", 10) # fewer results but full text
#         self.cache_dir  = pm_cfg.get("cache_dir", "/mnt/d/Harsha/AoLM/project/clinproof/data/pubmed_cache")
#         os.makedirs(self.cache_dir, exist_ok=True)

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         log.info(f"PubMedDenseRetriever: device={self.device}, cache={self.cache_dir}")

#         self._q_encoder  = None
#         self._a_encoder  = None

#     def _load_encoders(self):
#         if self._q_encoder is not None:
#             return
#         try:
#             from sentence_transformers import SentenceTransformer
#             log.info("Loading MedCPT-Query-Encoder...")
#             self._q_encoder = SentenceTransformer(QUERY_MODEL,   device=self.device)
#             log.info("Loading MedCPT-Article-Encoder...")
#             self._a_encoder = SentenceTransformer(ARTICLE_MODEL, device=self.device)
#         except Exception as e:
#             log.error(f"MedCPT encoder load failed: {e}")

#     # ── E-utilities ──────────────────────────────────────────────────────────

#     def _search_ids(self, query, db="pmc", max_results=5):
#         cache_key  = hashlib.md5(f"search:{db}:{query}:{max_results}".encode()).hexdigest()
#         cache_path = os.path.join(self.cache_dir, f"ids_{cache_key}.json")
#         if os.path.exists(cache_path):
#             with open(cache_path) as f:
#                 return json.load(f)

#         params = {
#             "db": db, "term": query, "retmax": max_results,
#             "retmode": "json", "email": self.email,
#         }
#         if self.api_key: params["api_key"] = self.api_key
#         try:
#             resp = requests.get(ENTREZ_SEARCH, params=params, timeout=10)
#             idlist = resp.json().get("esearchresult", {}).get("idlist", [])
#             with open(cache_path, "w") as f: json.dump(idlist, f)
#             return idlist
#         except Exception as e:
#             log.warning(f"{db} search failed: {e}")
#             return []

#     def _fetch_documents(self, doc_ids, db="pmc"):
#         docs = []
#         uncached, uncached_ids = [], []

#         for did in doc_ids:
#             cache_path = os.path.join(self.cache_dir, f"{db}_doc_{did}.json")
#             if os.path.exists(cache_path):
#                 with open(cache_path) as f:
#                     docs.append(json.load(f))
#             else:
#                 uncached_ids.append(did)

#         if uncached_ids:
#             # Batch fetch in chunks of 50 to prevent URI too long
#             for i in range(0, len(uncached_ids), 50):
#                 batch_ids = uncached_ids[i:i+50]
#                 params = {
#                     "db": db, "id": ",".join(batch_ids),
#                     "retmode": "xml", "email": self.email,
#                 }
#                 if self.api_key: params["api_key"] = self.api_key
#                 try:
#                     resp = requests.get(ENTREZ_FETCH, params=params, timeout=20)
#                     if db == "pmc":
#                         parsed = self._parse_pmc_xml(resp.text, batch_ids)
#                     else:
#                         parsed = self._parse_pubmed_xml(resp.text, batch_ids)

#                     for doc in parsed:
#                         did = doc.get("ID", "")
#                         cache_path = os.path.join(self.cache_dir, f"{db}_doc_{did}.json")
#                         with open(cache_path, "w") as f: json.dump(doc, f)
#                     docs.extend(parsed)
#                     time.sleep(0.3)
#                 except Exception as e:
#                     log.warning(f"{db} fetch failed: {e}")

#         return docs

#     def _parse_pmc_xml(self, xml_text, pmcids):
#         docs = []
#         articles = re.split(r'<article ', xml_text)[1:]
#         for block in articles:
#             pmcid_m = re.search(r'<article-id pub-id-type="pmc">(\d+)</article-id>', block)
#             title_m = re.search(r'<article-title[^>]*>(.*?)</article-title>', block, re.DOTALL)

#             if not pmcid_m: continue
#             pmcid = pmcid_m.group(1)
#             title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else "PMC Article"

#             # Extract paragraphs / abstract
#             paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
#             abstract_texts = re.findall(r'<abstract[^>]*>(.*?)</abstract>', block, re.DOTALL)

#             # clean abstract and body separately to make sure we don't duplicate
#             all_text_arr = []
#             if abstract_texts:
#                 for a in abstract_texts:
#                     all_text_arr.append(re.sub(r'<[^>]+>', '', a).strip())

#             for p in paragraphs:
#                 cl = re.sub(r'<[^>]+>', '', p).strip()
#                 if len(cl) > 30:  # Skip trivial texts
#                     all_text_arr.append(cl)

#             content = " ".join(all_text_arr)
#             docs.append({"ID": pmcid, "title": title, "content": content, "source": "pmc"})
#         return docs

#     def _parse_pubmed_xml(self, xml_text, pmids):
#         docs = []
#         articles = re.split(r'<PubmedArticle>', xml_text)[1:]
#         for block in articles:
#             pmid_m   = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
#             title_m  = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', block, re.DOTALL)
#             abs_matches = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', block, re.DOTALL)
#             if not pmid_m: continue
#             pmid = pmid_m.group(1)
#             title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""
#             if not abs_matches: continue

#             abstract = " ".join([re.sub(r'<[^>]+>', '', m).strip() for m in abs_matches]).strip()
#             if abstract:
#                 docs.append({"ID": pmid, "title": title, "content": abstract, "source": "pubmed"})
#         return docs

#     # ── Chunking ────────────────────────────────────────────────────────────

#     def _chunk_text(self, text, max_words=150, overlap=30):
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), max_words - overlap):
#             c = " ".join(words[i:i + max_words])
#             if len(c.split()) > 10:  # minimum chunk size
#                 chunks.append(c)
#         return chunks

#     # ── Main retrieve ────────────────────────────────────────────────────────

#     def retrieve(self, query, k=15, pubmed_mode="pmc"):
#         """Search PMC/PubMed, fetch full/abstracts, chunk them, return top-k semantic chunks."""
#         self._load_encoders()
#         if self._q_encoder is None:
#             log.warning("Encoders not loaded — returning empty results")
#             return [], []

#         if pubmed_mode == "pubmed":
#             doc_ids = self._search_ids(query, db="pubmed", max_results=self.max_results)
#             db_used = "pubmed"
#         else:
#             # Try PMC First (Full Texts)
#             doc_ids = self._search_ids(query, db="pmc", max_results=self.max_results)
#             db_used = "pmc"

#             # Fallback to PubMed (Abstracts) if PMC has no hits
#             if not doc_ids:
#                 doc_ids = self._search_ids(query, db="pubmed", max_results=self.max_results)
#                 db_used = "pubmed"

#         if not doc_ids:
#             return [], []

#         docs = self._fetch_documents(doc_ids, db=db_used)
#         if not docs:
#             return [], []

#         # Chunk all retrieved documents
#         all_chunks = []
#         for d in docs:
#             text_chunks = self._chunk_text(d.get("content", ""))
#             for i, tc in enumerate(text_chunks):
#                 all_chunks.append({
#                     "title": f"{d.get('title', '')} [Chunk {i+1}]",
#                     "content": tc,
#                     "source": db_used,
#                     "PMID": d.get("ID") # use ID field
#                 })

#         if not all_chunks:
#             return [], []

#         # Encode chunks
#         pairs = [[c["title"], c["content"]] for c in all_chunks]
#         with torch.no_grad():
#             a_embs = self._a_encoder.encode(pairs, batch_size=32,
#                                             normalize_embeddings=True,
#                                             show_progress_bar=False).astype("float32")

#         # Encode query
#         with torch.no_grad():
#             q_emb = self._q_encoder.encode([query], normalize_embeddings=True,
#                                            show_progress_bar=False).astype("float32")

#         # Cosine similarity
#         sims = (a_embs @ q_emb.T).flatten()
#         top_k = min(k, len(all_chunks))
#         idx   = np.argsort(sims)[::-1][:top_k]

#         return [all_chunks[i] for i in idx], [float(sims[i]) for i in idx]


"""
ClinProof PubMed Local Dense Retriever
Drop-in replacement for PubMedDenseRetriever that uses a pre-built local
FAISS index instead of live Entrez API calls.

Build the index first:
    python build_pubmed_index.py --xml_dir /path/to/xmls --out_dir /path/to/index

Then point your config to the index:
    pubmed:
      local_index_dir: /path/to/index
      # (all other pubmed keys still work as fallback config)
"""
import os
import json
import logging
import numpy as np
import torch

log = logging.getLogger("pubmed_local_retriever")

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


class PubMedDenseRetriever:
    """
    Local FAISS-backed retriever. Same interface as the API version:
        docs, scores = retriever.retrieve(query, k=15)

    Falls back to the API-based retriever if no local index is configured.
    """

    def __init__(self, config):
        pm_cfg = config.get("pubmed", {})

        self.local_index_dir = pm_cfg.get("local_index_dir", "")
        self.cache_dir = pm_cfg.get("cache_dir",
                                    "/mnt/d/Harsha/AoLM/project/clinproof/data/pubmed_cache")
        self.email = pm_cfg.get("email",   "clinproof@example.com")
        self.api_key = pm_cfg.get("api_key", "")
        self.max_results = pm_cfg.get("max_results", 10)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"PubMedLocalRetriever: device={self.device}")

        self._q_encoder = None
        self._faiss_index = None
        self._chunks = None        # list of chunk metadata dicts
        self._api_fallback = None       # lazy-loaded API retriever

        # Try to load local index immediately so we fail fast if misconfigured
        if self.local_index_dir:
            self._load_local_index()

    # ── Index loading ────────────────────────────────────────────────────────

    def _load_local_index(self):
        index_path = os.path.join(self.local_index_dir, "pubmed.faiss")
        metadata_path = os.path.join(self.local_index_dir, "pubmed_meta.db")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            log.warning(
                f"Local index not found at {self.local_index_dir}. "
                f"Run build_sqlite_meta.py first. Will fall back to API."
            )
            return

        try:
            import faiss
            log.info(f"Loading FAISS index from {index_path}...")
            self._faiss_index = faiss.read_index(index_path)
            log.info(f"FAISS index loaded: {self._faiss_index.ntotal} vectors")
            self._chunks = "sqlite"

        except Exception as e:
            log.error(f"Failed to load local index: {e}")
            self._faiss_index = None
            self._chunks = None

    def _load_query_encoder(self):
        if self._q_encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading MedCPT-Query-Encoder...")
            self._q_encoder = SentenceTransformer(
                QUERY_MODEL, device=self.device)
        except Exception as e:
            log.error(f"Query encoder load failed: {e}")

    # ── Local retrieval ──────────────────────────────────────────────────────

    def _retrieve_local(self, query, k=15):
        """Query the local FAISS index."""
        self._load_query_encoder()
        if self._q_encoder is None:
            return [], []

        with torch.no_grad():
            q_emb = self._q_encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

        # FAISS inner product search (embeddings are normalized → cosine sim)
        if hasattr(self._faiss_index, "nprobe"):
            self._faiss_index.nprobe = max(16, getattr(self._faiss_index, "nprobe", 16))
            
        scores, indices = self._faiss_index.search(q_emb, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        import sqlite3
        conn = sqlite3.connect(os.path.join(self.local_index_dir, "pubmed_meta.db"))
        c = conn.cursor()

        docs, final_scores = [], []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= self._faiss_index.ntotal:
                continue
            
            # FAISS index is 0-indexed, SQLite rowid is 1-indexed
            c.execute("SELECT title, content, pmid, source FROM chunks WHERE rowid = ?", (idx + 1,))
            row = c.fetchone()
            if row:
                docs.append({
                    "title": row[0],
                    "content": row[1],
                    "pmid": row[2],
                    "source": row[3]
                })
                final_scores.append(float(score))

        conn.close()
        return docs, final_scores

    # ── API fallback ─────────────────────────────────────────────────────────

    def _get_api_fallback(self):
        """Lazy-load the original API-based retriever as fallback."""
        if self._api_fallback is not None:
            return self._api_fallback
        try:
            from src.retrieval.pubmed_api_retriever import PubMedAPIRetriever
            cfg = {
                "pubmed": {
                    "email":       self.email,
                    "api_key":     self.api_key,
                    "max_results": self.max_results,
                    "cache_dir":   self.cache_dir,
                }
            }
            self._api_fallback = PubMedAPIRetriever(cfg)
            log.info("API fallback retriever loaded")
        except Exception as e:
            log.warning(f"Could not load API fallback: {e}")
        return self._api_fallback

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query, k=15, pubmed_mode="pmc"):
        """
        Retrieve top-k chunks for a query.

        Priority:
          1. Local FAISS index (fast, offline, full corpus)
          2. API fallback (if local index not available)
        """
        # Local index path
        if self._faiss_index is not None and self._chunks is not None:
            docs, scores = self._retrieve_local(query, k=k)
            if docs:
                log.debug(
                    f"Local retrieval: {len(docs)} chunks for query '{query[:50]}'")
                return docs, scores
            else:
                log.warning(
                    f"Local retrieval returned nothing for: '{query[:60]}'")

        # API fallback
        log.info("Falling back to API retrieval")
        fallback = self._get_api_fallback()
        if fallback:
            return fallback.retrieve(query, k=k, pubmed_mode=pubmed_mode)

        log.error("No retrieval method available")
        return [], []
