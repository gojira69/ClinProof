# ClinProof: Decomposition-Driven Retrieval-Augmented Medical Fact Verification

ClinProof is a high-precision medical fact-checking pipeline that combines **Atomic Proposition Decomposition**, **Hierarchical Multi-Source Retrieval**, and **Ensemble LLM Reasoning** to verify complex clinical claims against a unified knowledge base.

---

## 🚀 Key Features

### 1. Atomic Proposition Decomposition
Unlike standard RAG, ClinProof utilizes a specialized `medllama2:7b` module to break down complex, multi-faceted health claims into discrete, verifiable atomic units. This ensures that every sub-claim is independently grounded in evidence, mitigating the "partial truth" problem in medical verification.

### 2. Hierarchical Retrieval Engine (MoE Retrieval)
ClinProof merges evidence from four distinct layers to maximize factual recall:
*   **Knowledge Graph (GraphRAG):** Extracts structured clinical relationships (May_Treat, Contraindicated_In, Isa) from UMLS, SNOMED CT, and RxNorm.
*   **Dense Semantic Search (MedCPT):** Utilizes `ncbi/MedCPT` article/query encoders for deep conceptual matching across PubMed and PMC.
*   **Static Corpus (BM25):** High-speed keyword indexing of core medical textbooks with **Recency-Weighted Scoring** to handle temporal consensus shifts.
*   **Live Web Search:** Integrated DuckDuckGo scraping for real-time verification of recent claims or emerging clinical trials.

### 3. MMR-Based Context Compression
To mitigate "Lost in the Middle" syndrome in large LLM contexts, ClinProof employs a **Maximal Marginal Relevance (MMR)** extractive compressor. It reduces raw retrieval context (often 20k+ tokens) into a dense, high-signal set of sentences, improving accuracy by **+19.6%** on complex datasets like HealthFC.

### 4. Ensemble Reasoning & Self-Consistency
Final verdicts are reached through a **Majority-Vote Ensemble** mechanism. The pipeline supports:
*   **Homogeneous Ensembles:** Multiple self-consistency passes (3x) of high-parameter models (e.g., Qwen2.5-14B).
*   **Heterogeneous Ensembles:** Diverse voting across specialized medical models (Meditron-7B, MedLlama2, Llama-3.1-8B) to capture varied reasoning heuristics.

---

## 📊 Evaluation Benchmarks

ClinProof is validated across diverse medical reasoning tasks:

| Dataset | Type | Classes | Split (Tr/Vl/Te) | Focus |
| :--- | :--- | :---: | :---: | :--- |
| **BioASQ-7b** | Fact Verification | 2 | 579 / -- / 166 | Biomedical Yes/No |
| **HealthFC** | Claim Checking | 3 | 675 / -- / 75 | Consumer Health (T/F/M) |
| **MedChangeQA**| Temporal RAG | 3 | 307 / 102 / 103| Outdated vs. Current Consensus |

---

## 🛠️ System Flow

1.  **Decomposition:** Raw Query $\rightarrow$ Atomic Propositions $\{P_1, P_2, ... P_n\}$.
2.  **Linking:** Extraction of medical entities and mapping to canonical Concept IDs (CUI/RXCUI).
3.  **Retrieval:** Parallel fetching from Graph, Dense, and Static indexes.
4.  **Compression:** MMR filtering of evidence to fit LLM budget.
5.  **Reasoning:** Multi-pass verification of each $\{P_n\}$ against context.
6.  **Ensemble:** Consensus voting to produce the final `SUPPORTED`, `REFUTED`, or `MIXTURE` verdict.

---

## 📖 Usage

### Core Evaluation
```bash
python eval_all.py --dataset healthfc --model qwen2.5:14b --votes 3 --use-pubmed --use-graph
```

### Ablation Orchestrator
```bash
python scripts/run_ablations.py --group S --datasets healthfc --parallel 5
```

---

## 📂 Repository Structure

*   `src/retrieval/`: Implementations of GraphRAG, BM25, and MedCPT retrievers.
*   `src/compression/`: MMR sentence-selection logic.
*   `src/generation/`: Ollama-based LLM interfaces and ensemble logic.
*   `scripts/`: Analysis and ablation orchestrators.
*   `results/`: Detailed JSON traces for all experimental runs.
