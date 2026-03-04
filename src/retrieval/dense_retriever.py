"""
ClinProof Dense Retriever: FAISS + MedCPT (or sentence-transformers)
"""
import os, json, logging, pickle
import numpy as np
import torch

log = logging.getLogger("dense_retriever")
DEFAULT_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


class DenseRetriever:
    def __init__(self, corpus_dir, corpus_name="textbooks", model_name=DEFAULT_MODEL, article_model_name=DEFAULT_ARTICLE_MODEL, device="cuda"):
        self.corpus_dir = corpus_dir
        self.corpus_name = corpus_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.docs, self.index = [], None
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading query encoder: {model_name}")
            self.query_encoder = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            log.warning(f"SentenceTransformer load failed: {e}")
            self.query_encoder = None
        self._load_or_build(article_model_name)

    def _load_or_build(self, article_model_name):
        try:
            import faiss
        except ImportError:
            log.warning("faiss not installed - dense retriever disabled"); return
        index_path = os.path.join(self.corpus_dir, self.corpus_name, "faiss_medcpt.index")
        docs_path = os.path.join(self.corpus_dir, self.corpus_name, "docs.pkl")
        if os.path.exists(index_path) and os.path.exists(docs_path):
            import faiss as faiss_lib
            self.index = faiss_lib.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.docs = pickle.load(f)
            log.info(f"Dense index loaded: {self.index.ntotal} vectors")
            return
        chunk_dir = os.path.join(self.corpus_dir, self.corpus_name, "chunk")
        if not os.path.exists(chunk_dir) or not self.query_encoder:
            log.warning("Cannot build dense index - missing corpus or encoder"); return
        try:
            from sentence_transformers import SentenceTransformer
            import faiss as faiss_lib
            article_encoder = SentenceTransformer(article_model_name, device=self.device)
            all_embs = []
            for fname in sorted(os.listdir(chunk_dir)):
                if not fname.endswith(".jsonl"): continue
                batch_texts, batch_docs = [], []
                with open(os.path.join(chunk_dir, fname)) as f:
                    for line in f:
                        if not line.strip(): continue
                        doc = json.loads(line)
                        batch_docs.append(doc)
                        batch_texts.append([doc.get("title",""), doc.get("content","")])
                if batch_texts:
                    with torch.no_grad():
                        embs = article_encoder.encode(batch_texts, batch_size=64, show_progress_bar=False)
                    all_embs.append(embs); self.docs.extend(batch_docs)
            if not all_embs: return
            arr = np.vstack(all_embs).astype("float32")
            faiss_lib.normalize_L2(arr)
            self.index = faiss_lib.IndexFlatIP(arr.shape[1])
            self.index.add(arr)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss_lib.write_index(self.index, index_path)
            with open(docs_path, "wb") as f:
                pickle.dump(self.docs, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Dense index built: {self.index.ntotal} vectors")
        except Exception as e:
            log.warning(f"Dense index build failed: {e}")

    def retrieve(self, query, k=32):
        if self.index is None or not self.docs or not self.query_encoder: return [], []
        try:
            import faiss
            with torch.no_grad():
                q_emb = self.query_encoder.encode([query], normalize_embeddings=True).astype("float32")
            scores, indices = self.index.search(q_emb, k)
            top = [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]
            return top, [float(s) for s in scores[0]]
        except Exception as e:
            log.warning(f"Dense retrieve failed: {e}"); return [], []
