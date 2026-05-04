"""
ClinProof Extractive Compressor: MMR-based sentence selection
"""
import re, logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("extractor")
LIVE_WEB_SOURCE = "duckduckgo_live"


def sent_tokenize(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 20]


def _display_title(doc):
    title = doc.get("title", "")
    if doc.get("source") == LIVE_WEB_SOURCE:
        return f"Live Web Evidence: {title}"
    return title


def _display_content(doc):
    if doc.get("source") == LIVE_WEB_SOURCE:
        parts = [
            f"URL: {doc.get('url', '')}",
            f"Domain: {doc.get('domain', '')}",
            f"Retrieved at: {doc.get('retrieved_at', '')}",
        ]
        if doc.get("published_date"):
            parts.append(f"Published: {doc.get('published_date')}")
        if doc.get("snippet"):
            parts.append(f"Snippet: {doc.get('snippet')}")
        return ". ".join(p for p in parts if p)
    return doc.get("content", "")


class ExtractiveCompressor:
    def __init__(self, config):
        comp = config.get("compression", {})
        self.enabled = comp.get("enabled", True)
        self.budget_ratio = comp.get("budget_ratio", 0.4)
        self.mmr_lambda = comp.get("mmr_lambda", 0.7)
        self.min_sentences = comp.get("min_sentences", 5)

    def compress(self, query, docs, context_length=32768, tokenizer=None):
        if not self.enabled or not docs:
            return self._raw_context(docs)
        # Flatten to sentences
        sents, doc_idx = [], []
        for i, doc in enumerate(docs):
            title = _display_title(doc)
            content = _display_content(doc)
            for s in sent_tokenize(f"{title}. {content}" if title else content):
                sents.append(s); doc_idx.append(i)
        if not sents:
            return self._raw_context(docs)
        # TF-IDF
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=10000)
            mat = vec.fit_transform([query] + sents)
        except Exception:
            return self._raw_context(docs)
        q_vec = mat[0]; s_vecs = mat[1:]
        relevance = cosine_similarity(q_vec, s_vecs).flatten()
        # Convert sparse rows to dense for MMR
        s_vecs_dense = [np.asarray(s_vecs[i].todense()).flatten() for i in range(s_vecs.shape[0])]
        # MMR
        budget_sents = max(self.min_sentences, int(context_length * self.budget_ratio / 200))
        selected, sel_vecs, remaining = [], [], list(range(len(sents)))
        while remaining and len(selected) < budget_sents:
            if not sel_vecs:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sm = np.vstack(sel_vecs)
                best = max(remaining, key=lambda i: self.mmr_lambda*relevance[i] - (1-self.mmr_lambda)*cosine_similarity(s_vecs_dense[i].reshape(1,-1), sm).max())
            selected.append(best); sel_vecs.append(s_vecs_dense[best]); remaining.remove(best)
        # Preserve doc order
        selected.sort(key=lambda i: (doc_idx[i], i))
        parts, cur_doc, doc_counter = [], None, 0
        for si in selected:
            di = doc_idx[si]
            if di != cur_doc:
                doc_counter += 1; cur_doc = di
                parts.append(f"\nDocument [{doc_counter}] (Title: {_display_title(docs[di]) or 'Unknown'}) ")
            parts.append(sents[si])
        result = " ".join(parts)
        log.debug(f"Compressor: {len(sents)} sents -> {len(selected)} ({len(result)} chars)")
        return result

    def _raw_context(self, docs):
        return "\n".join(
            f"Document [{i+1}] (Title: {_display_title(d)}) {_display_content(d)}"
            for i, d in enumerate(docs)
        )
