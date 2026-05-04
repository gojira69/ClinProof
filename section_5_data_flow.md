# Section 5: Data Flow Through Pipeline

This section traces a single clinical query through the entire system from input to final output, illustrating the exact transformations that occur at each stage.

**Example Task:** A BioASQ claim verification question.

---

### 1. Raw Query
The user submits a factual medical question to the pipeline.
**Input:** 
> "Have mutations in the Polycomb group been found in human diseases?"

---

### 2. Atomic Decomposition
The `AtomicDecomposer` intercepts the raw query and passes it to the `medllama2:7b` model to extract entities and break the query into minimal verifiable claims.
**Output (JSON):**
```json
{
  "entities": ["Polycomb group", "mutations", "human diseases"],
  "propositions": [
    "mutations in the Polycomb group have been found in human diseases"
  ]
}
```

---

### 3. Retrieved Documents (Raw Candidates)
The system uses the raw query and extracted entities to query both the BM25 sparse index (textbooks) and the MedCPT dense index (PubMed FAISS).
**Output (Top-K Raw Chunks - *Pre-RRF*):**
- **BM25 Doc 1:** `InternalMed_Harrison` (Excerpt about nonconservative substitutions in genes segregating with inherited human diseases).
- **Dense Doc 2:** `Cell_Biology_Alberts` (Excerpt defining Polycomb group proteins forming stable complexes to maintain repressed chromatin states).
- **Dense Doc 3:** `PubMed Abstract` (Discussing Polycomb repressive complex 2 (PRC2) mutations in malignancies like lymphoma).
*(Returns up to 200 raw chunks)*

---

### 4. Knowledge Graph Traversal (GraphRAG)
The `GraphRetriever` performs entity linking on the extracted entities ("Polycomb group") using exact and fuzzy n-gram matching against the KG.
**KG Path Extracted:**
```
Entity: Polycomb Group Proteins
Definition: A family of highly conserved proteins that silence gene expression.
- Associated With: Hematologic Neoplasms, EZH2 mutations
- [Inverse] Causative Agent: Malignant Lymphoma
```
This multi-hop path is converted into a synthetic "document" and injected into the candidate pool.

---

### 5. Pre-Compression Context
All retrieved evidence (BM25 + Dense + KG) is merged via Reciprocal Rank Fusion (RRF) into a single ordered list of `k=50` documents. The atomic propositions are pinned to the top as Document [1].
**State:** 
> ~45,000 tokens of raw text. Documents include long textbook chapters containing formatting artifacts, tangential genetic pathways, and redundant definitions of human disease.

---

### 6. Post-Compression Context
The `ExtractiveCompressor` tokenizes all 50 documents into individual sentences, computes TF-IDF cosine similarity against the query, and uses Maximal Marginal Relevance (MMR, `λ=0.7`) to select the most relevant, non-redundant sentences up to a strict 10,000 character budget.
**Output (Compressed String):**
```text
Document [1] (Title: Atomic Propositions) 
Key medical claims to verify:
- mutations in the Polycomb group have been found in human diseases 

Document [2] (Title: PubMed 21816273) 
Recurrent mutations in the Polycomb group (PcG) gene EZH2 have recently been discovered in human B-cell lymphomas.

Document [3] (Title: KG: Polycomb Group Proteins)
Entity: Polycomb Group Proteins
- Associated With: Hematologic Neoplasms
- [Inverse] Causative Agent: Malignant Lymphoma

Document [4] (Title: Cell_Biology_Alberts) 
Polycomb group proteins form stable complexes that bind to chromatin to maintain the repressed state.
```
*(Context reduced from 45,000 tokens down to ~350 tokens of highly concentrated facts).*

---

### 7. LLM Input
The formatted prompt is constructed and sent to the reasoning ensemble (`qwen2.5:14b`, `meditron:7b`, `llama3.1:8b`).
**Input to LLM (Pass 1 - Qwen2.5):**
```text
You are a critical biomedical expert evaluating a yes/no biomedical question.
Before answering, explicitly reason through BOTH sides:
1. Evidence explicitly supporting YES
2. Evidence explicitly supporting NO or directly contradicting YES

Evidence:
[... Post-Compression Context inserted here ...]

Question: Have mutations in the Polycomb group been found in human diseases?
Options:
A. Yes
B. No

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B"}
```

---

### 8. Final Output
The three models independently reason over the context and output their JSON traces.
**Traces:**
- **Vote 1 (Qwen2.5):** `{"step_by_step_thinking": "Document [2] explicitly states that recurrent mutations in the Polycomb group gene EZH2 have been discovered in human B-cell lymphomas, which is a human disease. Document [3] also associates the Polycomb Group Proteins with hematologic neoplasms. Therefore, mutations in this group are found in human diseases.", "answer_choice": "A"}`
- **Vote 2 (Meditron):** `{"step_by_step_thinking": "Evidence in PubMed 21816273 links Polycomb group (PcG) gene mutations to B-cell lymphomas. This confirms the claim.", "answer_choice": "A"}`
- **Vote 3 (Llama3.1):** `{"step_by_step_thinking": "The provided context shows EZH2, a Polycomb group gene, is mutated in human lymphomas.", "answer_choice": "A"}`

**Aggregated System Output:**
```json
{
  "id": "bioasq_1",
  "pred_answer": "A",
  "pred_label": "Yes",
  "vote_distribution": {
    "A": 3
  },
  "confidence": 1.0,
  "correct": true
}
```
*The pipeline successfully outputs the correct label (Yes) with 100% ensemble confidence, explicitly citing the compressed medical evidence.*
