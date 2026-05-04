"""
ClinProof Main Pipeline
Orchestrates: MoE retrieval → extractive compression → LLM generation → PubMed citation
"""
import os
import sys
import json
import logging
import yaml
from typing import Optional
from pathlib import Path

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.live_web_search import LiveWebSearchRetriever
from src.retrieval.moe_retriever import MoERetriever
from src.compression.extractor import ExtractiveCompressor
from src.generation.ollama_llm import OllamaLLM
from src.generation.citation import CitationAttacher
from src.utils.paths import load_yaml_config, project_path

log = logging.getLogger("pipeline")


class ClinProof:
    """
    ClinProof: Medical fact verification pipeline.

    vs. MedRAG / MedCite:
      + GraphRAG over UMLS/SNOMED/RxNorm knowledge graph
      + Mixture-of-Experts domain routing
      + Extractive compression (MMR) before LLM call
      + Fast Ollama inference (no HuggingFace pipeline overhead)
      + PubMed PMID citations (clean, no hallucinated references)
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = project_path("config", "default.yaml")

        self.config = load_yaml_config(config_path)

        log.info("Initializing ClinProof pipeline...")

        # LLM
        self.llm = OllamaLLM(self.config)

        # Compression
        self.compressor = ExtractiveCompressor(self.config)

        # Citation (simplified: PubMed PMID only)
        self.citation = CitationAttacher(self.config)

        # Retrievers
        retrieval_mode = self.config.get("retrieval", {}).get("mode", "moe_graph")
        corpus_dir = self.config.get("corpus", {}).get("dir", "data/corpus")

        self.bm25 = None
        self.dense = None
        self.graph = None
        self.moe = None
        self.live_web = None

        if retrieval_mode in ("bm25", "moe_flat", "moe_graph"):
            try:
                self.bm25 = BM25Retriever(corpus_dir)
                log.info("BM25 retriever ready")
            except Exception as e:
                log.warning(f"BM25 init failed: {e}")

        if retrieval_mode in ("medcpt", "moe_flat", "moe_graph"):
            try:
                self.dense = DenseRetriever(corpus_dir)
                log.info("Dense retriever ready")
            except Exception as e:
                log.warning(f"Dense retriever init failed: {e}")

        graph_path = self.config.get("kg", {}).get("graph_path", "")
        if retrieval_mode in ("graph", "moe_graph") and graph_path and os.path.exists(graph_path):
            try:
                self.graph = GraphRetriever(graph_path, self.config)
                log.info("GraphRAG retriever ready")
            except Exception as e:
                log.warning(f"GraphRAG init failed: {e}")

        live_cfg = self.config.get("live_search", {})
        if live_cfg.get("enabled", False):
            self.live_web = LiveWebSearchRetriever(
                timeout=live_cfg.get("timeout", 8)
            )
            log.info("Live DDGS retriever ready")

        if retrieval_mode in ("moe_flat", "moe_graph"):
            self.moe = MoERetriever(
                graph_retriever=self.graph,
                bm25_retriever=self.bm25,
                dense_retriever=self.dense,
                config=self.config,
                live_web_retriever=self.live_web,
            )
            log.info("MoE retriever ready")

        log.info(f"ClinProof ready — model={self.llm.model_name}, mode={retrieval_mode}")

    def verify(
        self,
        question: str,
        options: Optional[dict] = None,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Main entry point for QA / fact verification.
        Returns: {answer, answer_choice, cited_docs, retrieval_scores}
        """
        k1 = self.config.get("retrieval", {}).get("k1", 32)

        # 1. Retrieve
        docs, scores = self._retrieve(question, k1)

        # 2. Compress
        compressed_context = self.compressor.compress(
            query=question,
            docs=docs,
            context_length=self.llm.context_length
        )

        # 3. Generate
        if options:
            gen_result = self.llm.answer_mcq(question, options, compressed_context)
            answer_text = gen_result.get("step_by_step_thinking", "")
            answer_choice = gen_result.get("answer_choice", "")
        else:
            answer_text = self.llm.answer_open(question, compressed_context)
            answer_choice = None

        # 4. Attach PubMed citations (PMID from retrieved corpus docs)
        citation_result = self.citation.process(
            answer_text=answer_text,
            answer_choice=answer_choice,
            snippets=docs,
        )

        result = {
            **citation_result,
            "retrieval_scores": scores
        }

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    def batch_verify(
        self,
        questions: list,
        save_dir: str,
        start_idx: int = 0,
        n_workers: int = 1
    ) -> list:
        """Process a list of questions, saving test_N.json per result."""
        os.makedirs(save_dir, exist_ok=True)
        results = []

        from tqdm import tqdm
        for i, q_data in tqdm(
            enumerate(questions, start=start_idx),
            total=len(questions),
            desc="ClinProof"
        ):
            save_path = os.path.join(save_dir, f"test_{i}.json")
            # Resume
            if os.path.exists(save_path):
                with open(save_path) as f:
                    result = json.load(f)
            else:
                result = self.verify(
                    question=q_data["question"],
                    options=q_data.get("options"),
                    save_path=save_path
                )
            if "answer" in q_data:
                result["ground_truth"] = q_data["answer"]
            results.append(result)

        return results

    def _retrieve(self, question: str, k: int) -> tuple:
        mode = self.config.get("retrieval", {}).get("mode", "moe_graph")
        if mode == "bm25" and self.bm25:
            return self.bm25.retrieve(question, k)
        elif mode == "medcpt" and self.dense:
            return self.dense.retrieve(question, k)
        elif mode == "graph" and self.graph:
            return self.graph.retrieve(question, k)
        elif self.moe:
            live_cfg = self.config.get("live_search", {})
            return self.moe.retrieve(
                question,
                k1=k,
                enable_live_search=live_cfg.get("enabled", False),
                live_search_k=live_cfg.get("k", 5),
                live_search_region=live_cfg.get("region", "in-en"),
            )
        elif self.bm25:
            return self.bm25.retrieve(question, k)
        return [], []
