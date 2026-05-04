# Section 3: Architecture & System Design (Deep Spec)

This section formally specifies the architecture of the ClinProof system, breaking down each component, its purpose, explicit implementation details, and input-output data flows.

## 3.1 Pipeline Overview

**Definition:** A hierarchical, multi-stage retrieval-augmented generation (RAG) pipeline designed for medical fact verification.
**Purpose:** To systematically break down complex medical claims, gather multi-modal evidence (sparse, dense, graph, live), compress it to eliminate noise, and reason over it using an ensemble of LLMs with calibration mechanisms.
**Implementation (THIS System):**
The execution flows through five ordered stages:
1. **Decomposition:** `AtomicDecomposer` splits the raw query into discrete propositions.
2. **Hierarchical Retrieval:**
   - **Stage 1 (Recall):** `BM25Retriever` fetches a broad pool of 200 candidates.
   - **Stage 2 (Precision):** `MoERetriever` fetches dense FAISS embeddings (MedCPT), KG paths (`GraphRetriever`), and live web search (`DDGS`).
   - **Merge:** Reciprocal Rank Fusion (RRF) combines Stage 1 and Stage 2 pools into a final top-k (`k=50`).
3. **Compression:** `ExtractiveCompressor` uses Maximal Marginal Relevance (MMR) to distill the top-50 documents into a dense, non-redundant context.
4. **Reasoning:** Local LLMs (`OllamaLLM`) process the compressed context using zero-shot role-playing prompts to output JSON reasoning traces.
5. **Ensemble & Voting:** Self-consistency voting across multiple passes/models aggregates the final verdict and computes confidence.

**Data Flow:**
Raw Query → [Atomic Decomposer] → Entities/Propositions → [Retrievers] → Raw Document Pool → [RRF Merge] → Ranked Docs → [MMR Compressor] → Compressed Context → [LLMs] → JSON Traces → [Voter] → Final Verdict.

---

## 3.2 Atomic Decomposition

**Definition:**
The process of breaking a complex, potentially ambiguous medical query into minimal, independent, and verifiable factual claims ("atoms") and isolating key medical entities.
*Constraints:*
- **Faithfulness:** Must not introduce facts absent from the query.
- **Minimality:** Each atom should test exactly one clinical mechanism or relationship.
- **Independence:** Atoms should be verifiable in isolation.

**Purpose:** To prevent the LLM from becoming confused by multi-faceted claims and to provide high-precision semantic seeds for Knowledge Graph traversal and live web searches.

**Implementation:**
- **Model:** `medllama2:7b` (via Ollama).
- **Method:** Prompting strategy that enforces a strict JSON schema output. If the LLM fails to output valid JSON, it falls back to a regex-based capitalization extractor.
- **Rules Enforced:** Exact drug names only, specific disease names only, simple factual claims.
- **# Atoms per query:** Typically 1 to 3 propositions and 2 to 4 entities.

**Failure Cases:**
- *Hedge introduction:* Adding "may" or "can" to a definitive claim, making it trivially true.
- *Parametric bleeding:* The decomposition LLM hallucinates known facts into the atoms instead of just parsing the query syntax.

**Examples (Query → Atoms):**
- *Query:* "A patient with heart failure is prescribed carvedilol. What is its mechanism?"
  *Atoms:* 
  `entities`: `["carvedilol", "heart failure"]`
  `propositions`: `["carvedilol is used to treat heart failure", "carvedilol is a beta-blocker", "carvedilol blocks adrenergic receptors"]`
- *Query:* "Does a selective sweep increase genetic variation?"
  *Atoms:*
  `entities`: `["selective sweep", "genetic variation"]`
  `propositions`: `["a selective sweep increases genetic variation"]`

---

## 3.3 Retrieval

**Definition:** The fetching of textual evidence from static and dynamic corpora.
**Purpose:** Ground the LLM in verified medical literature rather than relying solely on parametric memory.

**Implementation:**
- **Algorithm:** BM25Okapi (via `rank-bm25`).
- **Formula & Parameters:** TF-IDF based exact keyword matching. Stage-1 retrieves `k=200` candidates.
- **Corpus & Indexing:** 
  - *Textbooks:* 18 major medical textbooks (e.g., Harrison's, Schwartz) indexed at the chunk level. Years are statically mapped via title prefixes (e.g., "InternalMed_Harrison" → 2022).
  - *PubMed:* Abstract chunks indexed with explicit `published_year` metadata.
- **Query Formation:** Uses the raw text query. If live web search is triggered without an LLM decomposer, a fallback query of long words (length > 3) is generated.
- **Reranking:** Reciprocal Rank Fusion (RRF) with `rrf_k=60`. Weights: MoE Semantic/Graph (0.6), BM25 (0.4).

**Inputs → Outputs:**
Query String → Tokenized Array → BM25 Scorer → Top 200 Unstructured Text Dictionaries.

---

## 3.4 Knowledge Graph

**Definition:** A multi-hop networkx graph representing structured relationships between biomedical entities.
**Purpose:** Provide structured, multi-hop relational context (e.g., drug-target-disease pathways) that unstructured text chunks often omit or scatter.

**Implementation:**
- **Node Types:** UMLS CUIs, RxNorm drugs, SNOMED concepts.
- **Edge Types:** Strictly filtered to `USEFUL_RELS` (e.g., `may_treat`, `mechanism_of_action`, `has_side_effect`). Junk structural edges (`has_class`, `property_of`) are blocked to prevent tangential walks.
- **Construction Method:** Pickled `networkx` graph loaded into memory.
- **Usage:**
  1. *Entity Linking:* Exact n-gram (up to 5-gram) matching with stopword filtering on the raw query + Fuzzy substring matching for LLM-extracted atomic entities.
  2. *Traversal:* 2-hop traversal over high-value edges.
  3. *Relevance Gating:* TF-IDF cosine similarity against the query. Any KG path scoring `< 0.15` is discarded to prevent "GAGA-factor" noise.

**Inputs → Outputs:**
Extracted Entities → Node IDs → 2-Hop Edge Traversal → Synthesized Text Paragraph.

**Example Graph (Text Form):**
Node: `[Carvedilol]` (Type: Pharmacologic Substance, Def: A nonselective beta-adrenergic blocker...)
Outgoing Edges:
- May Treat: `[Heart Failure]`, `[Hypertension]`
- Mechanism Of Action: `[Adrenergic Receptor Blockade]`
Hop 2 Bridges:
- `[Carvedilol]` --may_treat--> `[Hypertension]`

---

## 3.5 Context + Compression

**Definition:** The refinement of retrieved documents into a token-efficient string.
**Purpose:** Fit massive multi-source evidence within the LLM context window while maximizing information density and eliminating redundant boilerplate.

**Implementation:**
- **Method:** `ExtractiveCompressor` using Maximal Marginal Relevance (MMR).
- **Compression Type:** Extractive (sentence selection), not abstractive.
- **Selection Criteria:** TF-IDF vectorization of individual sentences. Sentences are scored iteratively. The formula balances `lambda * (cosine similarity to query)` against `(1 - lambda) * (maximum cosine similarity to already selected sentences)`.
- **Parameters:** `mmr_lambda = 0.7`, `budget_ratio = 0.4` (target size is 40% of the LLM context window, roughly 10,000 chars for a 25,000 char window). Minimum 5 sentences.

**Inputs → Outputs:**
List of 50 Raw Dictionaries (BM25 + KG + Dense) → Sentence Tokenization → MMR Selection → Single concatenated Markdown string.

**BEFORE vs AFTER Example:**
- *BEFORE:* 50 full textbook chunks and 5 dense PubMed abstracts (approx. 45,000 tokens of dense medical text containing generic chapter introductions, formatting artifacts, and overlapping facts).
- *AFTER:* "Document [1] (Title: Harrison's Internal Med) Carvedilol reduces mortality in HFrEF. Document [4] (Title: KG: Carvedilol) Mechanism: nonselective beta blockade." (approx. 2,000 tokens of highly concentrated, non-redundant factual assertions).

---

## 3.6 LLM Reasoning

**Definition:** The generative inference engine that synthesizes context to produce a verdict.
**Purpose:** Apply clinical judgment and instruction-following to evaluate the claim against the compressed evidence.

**Implementation:**
- **Models:** `qwen2.5:14b` (Primary), `meditron:7b`, `llama3.1:8b`, `mistral:7b`.
- **Prompt Format:** System prompts use strict persona framing ("You are a critical biomedical expert..."). They enforce a Chain-of-Thought protocol ("Reason step-by-step: 1. What does the claim assert? 2. Does evidence explicitly support...").
- **Input/Output Schema:**
  - *Input:* System Prompt + User Query + Formatted Answer Options + Compressed Context String + Atomic Propositions.
  - *Output:* Strict JSON enforcing `{"step_by_step_thinking": "string", "answer_choice": "string"}`.

---

## 3.7 Ensemble & Voting

**Definition:** Aggregation of multiple LLM forward passes to form a final prediction.
**Purpose:** Smooth out model-specific hallucinations, mitigate temperature variance, and provide a confidence calibration metric.

**Implementation:**
- **Vote Space:** The defined options for the dataset (e.g., `["A", "B", "C"]` mapping to `SUPPORTED`, `REFUTED`, `NEI`).
- **Strategy:** Majority voting via self-consistency. `votes=3`.
  - *Homogeneous:* Same model runs 3 times (temperature = 0.35).
  - *Heterogeneous:* Round-robin scheduling across different models (e.g., Pass 1: Qwen, Pass 2: Meditron, Pass 3: Llama).
- **Tie Handling:** On a 3-way unanimous split (1 vote A, 1 vote B, 1 vote C), the system defaults to a conservative fallback (Option C: "NOT ENOUGH INFORMATION" / "MIXTURE").
- **Confidence:** Calculated as the vote fraction of the majority winner (e.g., `2/3 = 0.67`). Used in MedChangeQA for threshold-based abstention (e.g., if max fraction < 0.67, predict NEI).

**Example Disagreement:**
- Pass 1 (Qwen): "SUPPORTED"
- Pass 2 (Meditron): "SUPPORTED"
- Pass 3 (Llama): "REFUTED"
- *Output:* Majority winner = "SUPPORTED", Confidence = 0.67.

---

## 3.8 Recency Handling

**Definition:** Mathematical up-weighting of documents published more recently.
**Purpose:** Prevent temporal mismatch failures on datasets like MedChangeQA where medical consensus has evolved over time.

**Implementation:**
- **Formula:** Post-retrieval BM25 scores are multiplied by a normalized recency weight:
  `score = score * (1.0 + alpha * norm_age)`
  where `norm_age = (doc_year - min_year) / (max_year - min_year)`.
- **Parameters:** `alpha` controls the strength (0.0 = off, 0.3 = mild, 0.7 = strong).
- **Extraction:** Years are extracted via `pub_year` metadata (PubMed) or statically mapped from textbook title prefixes (e.g., `Schwartz's Principles of Surgery` → 2019).

---

## 3.9 External Tools

**Definition:** Third-party APIs and libraries integrated into the retrieval subsystem.
**Purpose:** Expand the evidence horizon beyond static offline files.

**Implementation:**
- **BM25:** `rank-bm25` (BM25Okapi) python library for offline sparse indexing.
- **Dense:** `FAISS` vector database utilizing the MedCPT dense embedding model.
- **Live Search:** `duckduckgo_search` (DDGS) python package used as a `LiveWebSearchRetriever`. Triggered on-the-fly with an 8-second timeout, restricted to "in-en" regions, fetching the top 5 snippets directly into the MMR compressor.
