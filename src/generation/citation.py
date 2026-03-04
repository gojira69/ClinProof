"""
ClinProof Citation Module — Simplified (PubMed-only)
Attaches PubMed PMIDs to the answer based on retrieved PubMed corpus docs.
No LLM reranking, no two-pass system — just track which PubMed docs
were retrieved and tag the answer with their PMIDs.
"""
import re
import logging
from typing import Optional

log = logging.getLogger("citation")


class CitationAttacher:
    """
    Simplified PubMed-only citation.
    - Scans retrieved docs for PMID fields
    - Numbers them and appends a References block to the answer
    - Citation markers [1][2] etc. are optional inline markers
    """

    def __init__(self, config: dict):
        self.enabled = config.get("citation", {}).get("mode", "pubmed") != "none"

    def process(
        self,
        answer_text: str,
        answer_choice: Optional[str],
        snippets: list[dict],
    ) -> dict:
        """
        Attach PubMed citations to the answer.
        Returns: {answer, answer_choice, cited_docs}
        """
        if not self.enabled:
            return {
                "answer": answer_text,
                "answer_choice": answer_choice,
                "cited_docs": {}
            }

        # Collect unique PubMed docs (must have PMID)
        cited_docs = {}
        doc_counter = 1
        for doc in snippets:
            pmid = doc.get("PMID") or doc.get("pmid") or ""
            # Only cite PubMed-sourced docs (have numeric PMID)
            if not pmid or not str(pmid).strip().isdigit():
                continue
            pmid_str = str(pmid).strip()
            if pmid_str in [v.get("pmid") for v in cited_docs.values()]:
                continue  # deduplicate
            cited_docs[str(doc_counter)] = {
                "title": doc.get("title", ""),
                "content": doc.get("content", "")[:300],
                "pmid": pmid_str,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"
            }
            doc_counter += 1

        # Append a clean References block
        final_answer = answer_text.strip()
        if cited_docs:
            refs = "\n\nReferences:\n"
            for num, doc in cited_docs.items():
                title = doc["title"] or "Untitled"
                refs += f"[{num}] {title}. PMID: {doc['pmid']} — {doc['pubmed_url']}\n"
            final_answer += refs

        return {
            "answer": final_answer,
            "answer_choice": answer_choice,
            "cited_docs": cited_docs
        }
