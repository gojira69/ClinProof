# Section 4: Best Configuration per Dataset

This section outlines the optimal system configuration for each evaluation dataset, explicitly detailing the parameters, the rationale behind the performance, and a comparison with the second-best configuration.

## 4.1 BioASQ (Yes/No Biomedical Questions)

**Best Configuration: B3 (Dense + BM25 Hybrid)**
- **Models:** `qwen2.5:14b` (Reasoning), `medllama2:7b` (Decomposition)
- **Retrieval:** Hybrid (MedCPT Dense + BM25Okapi)
- **Knowledge Graph (GraphRAG):** Disabled
- **Atomic Decomposition:** Enabled
- **Self-Consistency Votes:** 3
- **Recency Weighting:** Disabled (α = 0.0)

**Why it works:**
BioASQ claims are highly specific but typically narrow in scope (strict Yes/No factual verification). The combination of dense semantic retrieval (MedCPT) and exact-match keyword retrieval (BM25) provides maximum recall for these specific entities without introducing multi-hop structural noise. The Knowledge Graph is actively harmful here because BioASQ questions rarely require complex multi-step reasoning over adverse effects or indirect mechanisms; the KG pulls in tangential associations that distract the LLM. 

**Comparison with Second-Best:**
The second-best completed configuration is **B1 (Dense Only)**.
- **B3 (Hybrid)** achieved **82.8% Macro-F1**.
- **B1 (Dense Only)** achieved **77.1% Macro-F1**.
Adding BM25 to the MedCPT dense index provided a massive **+5.7% F1** boost by recovering documents with exact acronym matches that the dense vector embeddings missed. Both configurations significantly outperform the 1-vote baselines (D1) and medical-only ensembles (D4).

---

## 4.2 HealthFC (Complex Health Fact-Checking)

**Best Configuration: B2 (Dense + Knowledge Graph)**
- **Models:** `qwen2.5:14b` (Reasoning), `medllama2:7b` (Decomposition)
- **Retrieval:** Hybrid (MedCPT Dense + GraphRAG)
- **BM25:** Disabled
- **Atomic Decomposition:** Enabled
- **Self-Consistency Votes:** 3
- **Recency Weighting:** Disabled (α = 0.0)

**Why it works:**
HealthFC contains complex, multi-faceted claims that often fall into the nuanced "Mixture" class (partially true/false). Standard unstructured text often lacks the explicitly connected pathways to resolve these nuances. The Knowledge Graph (KG) succeeds here by providing explicit 2-hop structural relationships (e.g., specific side effects or contraindicated mechanisms). Furthermore, Atomic Decomposition is critical for HealthFC: breaking a convoluted claim into multiple distinct propositions allows the LLM to independently verify the true and false components, enabling it to correctly predict the "Mixture" class. BM25 exact-matching actually pulls in irrelevant documents and hurts performance.

**Comparison with Second-Best:**
The second-best configuration is **E1 (BM25 + Decomposition)**.
- **B2 (Dense + KG)** achieved **46.2% Macro-F1**.
- **E1 (BM25 + Decomp)** achieved **45.4% Macro-F1**.
B2 slightly edges out E1 because GraphRAG provides higher-precision context than BM25 for complex claims. More importantly, comparing against **E2 (No Decomp)** (25.8% F1), the presence of Atomic Decomposition in both B2 and E1 is the single largest ablation win in the entire study (**~+20% F1**).

---

## 4.3 MedChangeQA (Temporal Medical Consensus)

**Best Configuration: G1e (PubMed Dense + Strong Recency)**
- **Models:** `qwen2.5:14b` (Reasoning), `medllama2:7b` (Decomposition)
- **Retrieval:** PubMed FAISS Dense Only
- **Knowledge Graph (GraphRAG):** Disabled
- **BM25:** Disabled
- **Atomic Decomposition:** Enabled
- **Self-Consistency Votes:** 3
- **Recency Weighting:** Enabled (α = 0.7)

**Why it works:**
MedChangeQA explicitly tests whether the system understands when medical consensus has flipped (e.g., an old treatment is now contraindicated). Standard retrievers fail completely because they retrieve highly-ranked *older* documents that support the *old* consensus, causing the model to get the question wrong. G1e succeeds by completely abandoning static textbook corpora (which lack document-level dates) and exclusively querying the PubMed index, where exact publication years are known. Applying a strong recency multiplier (`α = 0.7`) artificially boosts the semantic score of the newest clinical trials, allowing the LLM to read the latest guidance and correctly classify claims that were recently REFUTED.

**Comparison with Second-Best:**
The second-best baseline is **G1f (PubMed Flat / No Recency)** or the old baseline **C1 (BM25 Flat)**.
- **G1e (Recency α=0.7)** achieved **~29.6% Macro-F1** (and the highest overall accuracy).
- **G1f (Flat)** achieved **~26.5% Macro-F1**.
- **C1 (BM25 Textbook)** achieved **20.8% Macro-F1** with a catastrophic bias toward the "SUPPORTED" class.
The strong recency weighting explicitly fixes the "Stale Knowledge" failure mode, recovering heavily penalized "REFUTED" predictions that flat retrievers fail on.
