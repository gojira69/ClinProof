"""
ClinProof live web retriever using the current `ddgs` package.

This module keeps retrieval tool-based: the verifier model never browses directly.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse

log = logging.getLogger("live_web_search")

SOURCE_NAME = "duckduckgo_live"

GLOBAL_PRIORITY_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "who.int",
    "cdc.gov",
]

INDIA_PRIORITY_DOMAINS = [
    "mohfw.gov.in",
    "icmr.gov.in",
    "nhp.gov.in",
    "pib.gov.in",
    "ncdc.mohfw.gov.in",
    "nhm.gov.in",
    "ayush.gov.in",
]

PRIORITY_WEIGHTS = {
    "pubmed.ncbi.nlm.nih.gov": 3.0,
    "ncbi.nlm.nih.gov": 2.8,
    "nih.gov": 2.3,
    "who.int": 2.2,
    "cdc.gov": 2.2,
    "mohfw.gov.in": 2.5,
    "icmr.gov.in": 2.5,
    "nhp.gov.in": 2.0,
    "pib.gov.in": 1.8,
    "ncdc.mohfw.gov.in": 2.4,
    "nhm.gov.in": 1.8,
    "ayush.gov.in": 1.2,
}

DEPRIORITIZED_DOMAIN_PATTERNS = (
    "reddit.com",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "tiktok.com",
    "blogspot.",
    "wordpress.",
    "medium.com",
    "pinterest.com",
)

DEPRIORITIZED_URL_PATTERNS = (
    "/shop",
    "/product",
    "/products",
    "/buy",
    "/forum",
    "/forums",
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower()))


def _clean_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domain_matches(domain: str, target: str) -> bool:
    return domain == target or domain.endswith(f".{target}")


class LiveWebSearchRetriever:
    """
    DDGS-backed live web search retriever.

    Returned documents include the standard ClinProof fields plus the required
    live-web metadata: title, url, snippet, source, domain, published_date,
    retrieved_at, and score.
    """

    def __init__(self, timeout: int = 8, backend: str = "duckduckgo"):
        self.timeout = timeout
        self.backend = backend
        self._memory_cache: dict[tuple, tuple[list[dict], list[float]]] = {}

    def retrieve(self, query: str, k: int = 5, region: str = "in-en") -> tuple[list[dict], list[float]]:
        return self.multi_retrieve([query], k=k, region=region)

    def multi_retrieve(
        self,
        queries: Iterable[str],
        k: int = 5,
        region: str = "in-en",
    ) -> tuple[list[dict], list[float]]:
        cleaned_queries = []
        seen_queries = set()
        for query in queries:
            if not query or not str(query).strip():
                continue
            cleaned = str(query).strip()
            if cleaned in seen_queries:
                continue
            seen_queries.add(cleaned)
            cleaned_queries.append(cleaned)
            if len(cleaned_queries) >= 3:
                break
        if not cleaned_queries or k <= 0:
            return [], []

        cache_key = (tuple(cleaned_queries), int(k), region, self.backend)
        if cache_key in self._memory_cache:
            docs, scores = self._memory_cache[cache_key]
            return list(docs), list(scores)

        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "Live web search requires the `ddgs` package. Install it with `pip install -U ddgs`."
            ) from e

        candidates: dict[str, dict] = {}
        now_iso = datetime.now(timezone.utc).isoformat()

        ddgs = DDGS(timeout=self.timeout)
        for query_idx, query in enumerate(cleaned_queries):
            try:
                results = ddgs.text(
                    query,
                    region=region,
                    safesearch="off",
                    max_results=max(k * 3, 10),
                    backend=self.backend,
                ) or []
            except Exception as e:
                log.warning(f"DDGS live search failed for '{query[:80]}': {e}")
                continue

            claim_terms = _tokenize(query)
            for rank, result in enumerate(results):
                doc = self._normalise_result(result, retrieved_at=now_iso)
                if not doc:
                    continue
                doc["score"] = self._score_result(
                    doc=doc,
                    rank=rank,
                    query_idx=query_idx,
                    search_query_idx=0,
                    claim_terms=claim_terms,
                )
                # Delay scraping until we have selected the top-k docs
                doc["content"] = self._format_content(doc, scrape=False)
                existing = candidates.get(doc["url"])
                if existing is None or doc["score"] > existing["score"]:
                    candidates[doc["url"]] = doc

        docs = sorted(candidates.values(), key=lambda d: d["score"], reverse=True)
        docs = self._diversify_by_domain(docs, k)
        
        # Scrape full text for the final selected documents
        for doc in docs:
            doc["content"] = self._format_content(doc, scrape=True)
        scores = [float(d["score"]) for d in docs]
        self._memory_cache[cache_key] = (list(docs), list(scores))
        return docs, scores

    def _priority_domains(self, region: str) -> list[str]:
        region = (region or "").lower()
        if region.startswith("in-"):
            return INDIA_PRIORITY_DOMAINS + GLOBAL_PRIORITY_DOMAINS
        return GLOBAL_PRIORITY_DOMAINS + INDIA_PRIORITY_DOMAINS[:3]

    def _normalise_result(self, result: dict, retrieved_at: str) -> dict | None:
        title = str(result.get("title") or "").strip()
        url = str(result.get("href") or result.get("url") or result.get("link") or "").strip()
        snippet = str(result.get("body") or result.get("snippet") or result.get("description") or "").strip()
        if not title or not url:
            return None

        domain = _clean_domain(url)
        if not domain:
            return None

        published_date = None
        for field in ("published_date", "published", "date"):
            value = result.get(field)
            if value:
                published_date = str(value).strip()
                break

        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": SOURCE_NAME,
            "domain": domain,
            "published_date": published_date,
            "retrieved_at": retrieved_at,
            "score": 0.0,
        }

    def _score_result(
        self,
        doc: dict,
        rank: int,
        query_idx: int,
        search_query_idx: int,
        claim_terms: set[str],
    ) -> float:
        domain = doc.get("domain", "")
        title_snippet_terms = _tokenize(f"{doc.get('title', '')} {doc.get('snippet', '')}")
        overlap = len(claim_terms & title_snippet_terms) / max(len(claim_terms), 1)

        score = 1.0 + overlap
        score += self._domain_priority_score(domain)
        score -= rank * 0.05
        score -= query_idx * 0.02
        if search_query_idx > 0:
            score += 0.15
        if len(doc.get("snippet", "")) < 40:
            score -= 0.1
        if any(pat in doc.get("url", "").lower() for pat in DEPRIORITIZED_URL_PATTERNS):
            score -= 0.6
        return float(score)

    def _domain_priority_score(self, domain: str) -> float:
        for target, weight in PRIORITY_WEIGHTS.items():
            if _domain_matches(domain, target):
                return weight

        if any(pat in domain for pat in DEPRIORITIZED_DOMAIN_PATTERNS):
            return -1.2

        lower_url_bits = domain.lower()
        if "blog" in lower_url_bits or "forum" in lower_url_bits:
            return -0.8
        if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".int"):
            return 0.8
        if domain.endswith(".edu") or ".ac.in" in domain:
            return 0.5
        return 0.0

    def _format_content(self, doc: dict, scrape: bool = False) -> str:
        parts = [
            f"URL: {doc.get('url', '')}",
            f"Domain: {doc.get('domain', '')}",
            f"Retrieved at: {doc.get('retrieved_at', '')}",
        ]
        if doc.get("published_date"):
            parts.append(f"Published: {doc['published_date']}")
            
        full_text = ""
        if scrape and doc.get("url"):
            try:
                import requests
                from bs4 import BeautifulSoup
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                resp = requests.get(doc["url"], timeout=4, headers=headers)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "html.parser")
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.extract()
                    text = soup.get_text(separator=' ')
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    # Limit to ~3000 chars to avoid blowing up context window
                    full_text = text[:3500]
            except Exception as e:
                log.warning(f"Failed to scrape {doc.get('url', '')}: {e}")

        if full_text and len(full_text) > 100:
            parts.append(f"Content:\n{full_text}")
        elif doc.get("snippet"):
            parts.append(f"Snippet: {doc['snippet']}")
            
        parts.append(
            "Runtime note: This live web snippet/article is weak secondary evidence and requires human verification."
        )
        return "\n".join(parts)

    def _diversify_by_domain(self, docs: list[dict], k: int) -> list[dict]:
        selected: list[dict] = []
        seen_domains: set[str] = set()

        for doc in docs:
            domain = doc.get("domain", "")
            if domain not in seen_domains:
                selected.append(doc)
                seen_domains.add(domain)
            if len(selected) >= k:
                return selected

        for doc in docs:
            if doc not in selected:
                selected.append(doc)
            if len(selected) >= k:
                break

        return selected[:k]
