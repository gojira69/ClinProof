# Section 7: Context Analysis

This section analyzes the impact of the `ExtractiveCompressor` module, which bridges the gap between high-recall broad retrieval and precision-constrained LLM reasoning.

## 7.1 Pre vs. Post Compression

The standard ClinProof retrieval pipeline (e.g., Configuration B4) utilizes a top-$k$ of 50 documents, merging results from BM25, MedCPT Dense, and GraphRAG.

**Pre-Compression State (Raw Retrieval):**
- **Size:** Typically 20,000 to 45,000 tokens per query.
- **Format:** Full unstructured textbook chunks (often 500-1000 words each), complete PubMed abstracts, and raw Knowledge Graph paths.
- **Signal-to-Noise Ratio:** Extremely low. Because BM25 relies on keyword matching, a textbook chapter on "Cardiovascular Disease" might be retrieved due to a keyword match for a specific drug, bringing along 5 pages of irrelevant adjacent conditions.

**Post-Compression State (MMR Extraction):**
- **Size:** Typically 300 to 1,500 tokens per query (enforced by a target budget ratio of 40% of the LLM context window, but typically much smaller due to sentence thresholding).
- **Format:** An ordered list of discrete sentences, retaining the `[Title]` metadata to provide source grounding.
- **Signal-to-Noise Ratio:** Extremely high. 

---

## 7.2 What is Removed vs. Preserved

The compressor operates using **Maximal Marginal Relevance (MMR)** on TF-IDF vectors of individual sentences.

**What is Removed (Redundancy & Noise Penalty):**
1. **Formatting Artifacts:** Unstructured OCR artifacts, bibliography references, or textbook figure captions ("*See Figure 4.2...*").
2. **Tangential Content:** Sentences that are part of a retrieved chunk but mathematically distant from the query vector (e.g., surgical techniques in a chapter retrieved to answer a pharmacological question).
3. **Redundant Definitions:** If MedCPT retrieves 5 distinct PubMed abstracts that all begin with the same background sentence ("*Heart failure is a leading cause of mortality globally...*"), MMR heavily penalizes sentences 2 through 5 because their cosine similarity to the already-selected first sentence is near 1.0. 

**What is Preserved (Relevance & Diversity Bonus):**
1. **High-Relevance Facts:** Sentences possessing high cosine similarity to the core user query or the extracted atomic entities.
2. **Diverse Claims:** MMR ensures that if a query is multi-faceted, sentences addressing *different* facets are selected. If the query asks about both "efficacy" and "safety", MMR prevents the context from being saturated solely by efficacy statistics.
3. **Atomic Propositions:** The output from the `AtomicDecomposer` is hard-pinned to the top of the context window (Document `[1]`) bypassing the compressor entirely, ensuring the LLM never loses sight of the core claims it must verify.

---

## 7.3 Effect on LLM Reasoning

The transition from a raw retrieval context to an MMR-compressed context exerts profound effects on downstream reasoning:

1. **Mitigation of "Lost in the Middle" Syndrome:** 
   Current LLMs (including `qwen2.5:14b` and `llama3.1:8b`) suffer severe attention degradation when critical facts are buried in the middle of a massive 32k token prompt. By stripping the context down to a dense list of facts, the LLM correctly attends to contradictory evidence that it would otherwise gloss over.
2. **Reduction of Parametric Hallucination:** 
   When overwhelmed with noisy, tangential text, LLMs tend to ignore the retrieved context and revert to their pre-trained parametric memory (which may be outdated or incorrect). A concise, highly relevant context forces the model to ground its reasoning exclusively in the provided text.
3. **Latency and Compute Efficiency:** 
   Compressing the context reduces the number of input tokens by up to 95%. In a local environment utilizing self-consistency (where 3 separate forward passes are executed per query), this compression reduces inference time from minutes to mere seconds per claim, making real-time verification feasible.
4. **Improved Confidence Calibration:** 
   Because the context is stripped of contradictory noise from unrelated textbook sections, the ensemble models are more likely to reach unanimous agreement (3/3 votes) when the evidence is definitive. Conversely, if the compressed context legitimately lacks evidence for a specific atom, the LLM is significantly more likely to explicitly output `NOT ENOUGH INFORMATION` rather than guessing based on tangential noise.
