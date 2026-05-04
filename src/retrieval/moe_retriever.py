"""
ClinProof MoE Retriever
Domain classification -> domain-aware GraphRAG traversal (no BM25 / dense corpus)
"""
import logging, re
from ollama import Client as OllamaClient

log = logging.getLogger("moe_retriever")

DOMAIN_KEYWORDS = {
    "pharmacology": ["drug","medication","dose","dosage","mg","pill","tablet","capsule",
                     "antibiotic","receptor","toxicity","overdose","contraindicated",
                     "side effect","adverse","brand","generic","ingredient","inhibitor",
                     "agonist","antagonist","bioavailability","half-life","pharmacokinetic"],
    "anatomy":      ["nerve","artery","vein","muscle","bone","organ","tissue","cell",
                     "anatomical","structure","location","region","innervation","blood supply",
                     "lymph","ligament","tendon","histology","morphology","cortex","nucleus"],
    "clinical":     ["patient","treatment","therapy","diagnosis","symptom","sign","prognosis",
                     "prevalence","incidence","risk factor","comorbidity","management",
                     "guideline","surgery","outcome","mortality","trial","study","cohort"],
}

# Per-domain: which edge types to prioritise in the KG traversal
DOMAIN_EDGE_PRIORITY = {
    "pharmacology": [
        "may_treat", "may_prevent", "mechanism_of_action", "has_mechanism_of_action",
        "has_physiologic_effect", "has_active_ingredient", "has_ingredient",
        "has_target", "contraindicated_with", "has_tradename", "tradename_of",
    ],
    "anatomy": [
        "finding_site", "part_of", "has_part", "component_of",
        "associated_with", "occurs_in", "classified_as",
    ],
    "clinical": [
        "causes", "associated_with", "disease_may_have_finding",
        "disease_may_have_associated_disease", "has_manifestation", "due_to",
        "may_be_treated_by", "may_treat", "causative_agent",
    ],
    "general": [],   # empty = use all USEFUL_RELS
}

# Per-domain: max hops to use
DOMAIN_HOPS = {
    "pharmacology": 4,
    "anatomy":      3,
    "clinical":     4,
    "general":      3,
}


class MoERetriever:
    """Domain-aware GraphRAG router.  No BM25/textbook corpus used."""

    def __init__(
        self,
        graph_retriever,
        bm25_retriever,
        dense_retriever,
        config,
        live_web_retriever=None,
        ollama_client=None,
    ):
        self.graph = graph_retriever
        # bm25 / dense kept as params for backwards-compatibility but intentionally unused
        self.config = config
        self.live_web = live_web_retriever
        self.model_name = config.get("model", {}).get("name", "mistral:7b")
        self.ollama = ollama_client or OllamaClient()
        self.last_live_search_meta = {
            "enabled": False,
            "attempted": False,
            "queries": [],
            "region": None,
            "k": 0,
            "results": 0,
            "error": None,
        }

    def classify_domain(self, query):
        q = query.lower()
        scores = {d: sum(1 for kw in kws if kw in q) for d, kws in DOMAIN_KEYWORDS.items()}
        best = max(scores, key=lambda d: scores[d])
        if scores[best] == 0:
            best = self._llm_classify(query)
        log.debug(f"MoE domain: {best}")
        return best

    def _llm_classify(self, query):
        try:
            resp = self.ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content":
                    f"Classify into ONE domain: pharmacology / anatomy / clinical / general.\n"
                    f"Output only the single word.\nQuery: {query}"}],
                options={"temperature": 0, "num_predict": 5})
            ans = resp["message"]["content"].strip().lower()
            for d in ["pharmacology", "anatomy", "clinical"]:
                if d in ans: return d
        except Exception:
            pass
        return "general"

    def retrieve(
        self,
        query,
        k1=32,
        k2=5,
        options=None,
        enable_pubmed=True,
        pubmed_mode="pmc",
        enable_live_search=False,
        live_search_k=5,
        live_search_region="in-en",
        live_search_queries=None,
    ):
        """
        Route query to:
          - GraphRAG (domain-aware edge priorities + hop depth)
          - PubMed dense retriever (if set via self.pubmed and enable_pubmed is True)
        Fuse results with Reciprocal Rank Fusion.
        """
        domain         = self.classify_domain(query)
        priority_edges = DOMAIN_EDGE_PRIORITY.get(domain, [])
        max_hops       = DOMAIN_HOPS.get(domain, 3)

        result_pools = []  # list of (docs, scores, weight)
        self.last_live_search_meta = {
            "enabled": bool(enable_live_search and self.live_web),
            "attempted": False,
            "queries": [q for q in (live_search_queries or []) if q],
            "region": live_search_region,
            "k": live_search_k,
            "results": 0,
            "error": None,
        }

        # ── GraphRAG (always used) ───────────────────────────────────────────
        if self.graph:
            original_hops      = getattr(self.graph, "max_hops", 2)
            self.graph.max_hops = max_hops
            try:
                res = self.graph.retrieve(
                    query, k=k1, options=options,
                    domain_edge_priority=priority_edges
                )
            except TypeError:
                res = self.graph.retrieve(query, k=k1, options=options)
            finally:
                self.graph.max_hops = original_hops
                
            if len(res) >= 2:
                docs, scores = res[0], res[1]
            else:
                docs, scores = [], []
            # Graph weight: higher for pharmacology/anatomy (structured data advantage),
            # slightly lower for clinical where PubMed abstracts shine
            graph_w = 0.6 if domain in ("pharmacology", "anatomy") else 0.45
            result_pools.append((docs, scores, graph_w))

        # ── PubMed dense retriever (optional) ───────────────────────────────
        pubmed = getattr(self, "pubmed", None)
        if pubmed and enable_pubmed:
            try:
                pm_docs, pm_scores = pubmed.retrieve(query, k=min(k1, 25), pubmed_mode=pubmed_mode)
                pubmed_w = 1.0 - graph_w if result_pools else 1.0
                result_pools.append((pm_docs, pm_scores, pubmed_w))
            except Exception as e:
                log.warning(f"PubMed dense retriever failed: {e}")

        # ── Live DDGS web retriever (optional, weak secondary evidence) ─────
        if self.live_web and enable_live_search:
            live_queries = [q for q in (live_search_queries or [query]) if q and str(q).strip()]
            self.last_live_search_meta.update({
                "attempted": True,
                "queries": live_queries,
            })
            try:
                live_docs, live_scores = self.live_web.multi_retrieve(
                    live_queries,
                    k=live_search_k,
                    region=live_search_region,
                )
                self.last_live_search_meta["results"] = len(live_docs)
                live_w = 0.2 if result_pools else 1.0
                if live_docs:
                    result_pools.append((live_docs, live_scores, live_w))
            except Exception as e:
                self.last_live_search_meta["error"] = str(e)
                log.warning(f"Live DDGS retriever failed: {e}")

        if not result_pools:
            return [], []
        if len(result_pools) == 1:
            return result_pools[0][0], result_pools[0][1]

        return self._rrf_merge(result_pools, k=k1)

    def _rrf_merge(self, result_pools, k=32, rrf_k=60):
        """Reciprocal Rank Fusion over multiple (docs, scores, weight) pools."""
        rrf_scores, doc_registry = {}, {}
        for docs, scores, weight in result_pools:
            for rank, doc in enumerate(docs):
                doc_id = (doc.get("url") or doc.get("PMID") or doc.get("id") or
                          doc.get("title", "")[:40])
                rrf_scores[doc_id] = (rrf_scores.get(doc_id, 0.0) +
                                      weight / (rrf_k + rank + 1))
                if doc_id not in doc_registry:
                    doc_registry[doc_id] = doc
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
        return ([doc_registry[i] for i in sorted_ids],
                [rrf_scores[i]    for i in sorted_ids])
