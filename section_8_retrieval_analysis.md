# Section 8: Retrieval Output Analysis

This section analyzes the raw performance of the retrieval subsystem, contrasting successful ranking with common failure modes such as vocabulary mismatch and absent evidence. It primarily evaluates the Stage 1 sparse retrieval (BM25) and its interplay with the dataset characteristics.

## 8.1 Raw BM25 Outputs and Ranking

The BM25Okapi retriever computes scores based on exact TF-IDF token overlaps between the user query and the corpus chunks (medical textbooks).

**Example Case: High Precision Ranking**
*Query:* "Does the use of aminaftone relieve symptoms such as itching or restless legs in chronic venous insufficiency?"
*Top 3 BM25 Raw Outputs:*
1. **Rank 1 (Score: 34.2) — Relevant:** `Surgery_Schwartz` (Chapter on Venous Disease). Mentions aminaftone and chronic venous insufficiency treatment.
2. **Rank 2 (Score: 28.5) — Relevant:** `InternalMed_Harrison` (Vascular section). Discusses management of restless legs and venous stasis using phlebotonics like aminaftone.
3. **Rank 3 (Score: 12.1) — Irrelevant:** `Neurology_Adams` (Restless Leg Syndrome). Mentions "itching" and "restless legs" but in the context of neurological dopamine deficiencies, completely unrelated to venous insufficiency or aminaftone.

*Analysis:* BM25 performs exceptionally well when the query contains rare, highly specific lexical tokens (e.g., "aminaftone"). The high IDF (Inverse Document Frequency) of the drug name forces the relevant documents to the top of the ranking, pushing noisy symptom-matching documents down.

---

## 8.2 Relevant vs. Irrelevant Retrieval (The "Over-Matching" Problem)

When queries consist of common medical terms without a rare anchor noun, BM25 frequently suffers from "over-matching" or "keyword salad" failures.

**Example Case: Low Precision Ranking**
*Query:* "Is there a relationship between dietary Vitamin D supplementation and the prevention of heart failure?"
*Top 3 BM25 Raw Outputs:*
1. **Rank 1 (Score: 41.2) — Irrelevant:** `Biochemistry_Lippinco` (Vitamin D Metabolism). Heavily matches "dietary", "Vitamin D", and "supplementation". Contains zero information about heart failure.
2. **Rank 2 (Score: 39.8) — Irrelevant:** `InternalMed_Harrison` (Heart Failure Management). Heavily matches "prevention" and "heart failure". Contains zero information about Vitamin D.
3. **Rank 3 (Score: 35.1) — Relevant (Partial):** `Cardiology_Atlas` (Nutrition in Cardiovascular Disease). Actually discusses the intersection of the two concepts.

*Analysis:* Because BM25 scores are additive across tokens, a document that densely repeats one half of the query (e.g., a chapter entirely about Vitamin D) will often outscore a document that briefly but accurately connects both concepts. This is why the `B2` configuration (Dense + KG without BM25) outperformed `B3` (Dense + BM25) on HealthFC—dense embeddings capture the *intersection* of the concepts in vector space, avoiding this exact-match trap.

---

## 8.3 Highlighting Retrieval Misses

A "Retrieval Miss" occurs when the evidence exists in the corpus but is not fetched in the top-$k$ candidates.

**Failure Mode: Vocabulary Mismatch (The Synonym Problem)**
BM25 fails when the user query uses clinical synonyms not present in the textbook text.
- *Query:* "Does *renal impairment* alter the dosing of rivaroxaban?"
- *Corpus Text:* "Patients with severe *kidney dysfunction* require dose adjustments when taking rivaroxaban."
- *Result:* BM25 assigns a score of 0 to "renal" and "impairment", causing the correct chunk to fall to Rank 150+, potentially dropping out of the context window.
*Fix in Pipeline:* This is why Stage 2 incorporates MedCPT dense retrieval, which maps "renal impairment" and "kidney dysfunction" to the same region in the embedding space.

---

## 8.4 Highlighting Missing Evidence

"Missing Evidence" occurs when the pipeline functions perfectly, but the required factual grounding simply does not exist in the indexed corpora. This was a catastrophic failure mode in early ablations on the **MedChangeQA** dataset.

**Failure Mode: Stale Corpora (The Temporal Shift Problem)**
- *Query (from MedChangeQA):* "Are beta-blockers recommended as first-line therapy for uncomplicated hypertension?" (Note: This changed from SUPPORTED to REFUTED in recent clinical guidelines).
- *Textbook Corpus:* The indexed `InternalMed_Harrison` (21st Ed, 2022) and `Pharmacology_Katzung` (14th Ed, 2018) still contain text stating beta-blockers are an acceptable first-line treatment.
- *Retrieval Output:* The retriever successfully fetches high-relevance paragraphs explicitly stating: "Beta-blockers are a standard first-line treatment for hypertension."
- *LLM Verdict:* The LLM faithfully reads the context and incorrectly outputs `SUPPORTED` (Accuracy drops).

*Fix in Pipeline:* This exact missing evidence gap necessitated the creation of the `G1e` configuration. By disabling the static textbook corpus entirely, dynamically querying the PubMed index, and applying a mathematical Recency Multiplier (`α = 0.7`), the system artificially boosts the latest clinical trials and updated meta-analyses to Rank 1, explicitly overwriting the stale textbook knowledge.
