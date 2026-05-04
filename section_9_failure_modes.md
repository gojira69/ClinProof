# Section 9: Failure Mode Analysis (Qualitative)

This section provides a rigorous qualitative analysis of the errors produced by the ClinProof pipeline. By extending the baseline taxonomy (F1–F6), we identify the root causes of failure, estimate their relative frequencies, and provide concrete examples drawn from the evaluation logs.

---

## F1 — Atomic Decomposition Error (Faithfulness Failure)

**Description:** The decomposition LLM (`medllama2:7b`) produces atoms that subtly change the claim's semantic meaning—adding hedges ("may", "can"), flipping polarity, or paraphrasing incorrectly. The downstream verifier then correctly reasons over the *wrong* atoms.
**Frequency:** ~5–10% of BioASQ errors.
**Concrete Example:**

- *Query:* "Does a selective sweep increase genetic variation?" (GT: No)
- *Generated Atom:* "A selective sweep *can* increase genetic variation."
- *Outcome:* The pipeline finds evidence that in rare outlier cases, a sweep might not completely eliminate variation, and answers YES.
**Root Cause Analysis:** Systematic bias in the decomposition prompt/model toward positive, hedged framing, destroying the strictness of the original claim.

---

## F2 — Retrieved Context Does Not Contain the Answer (Retrieval Miss)

**Description:** The retrievers (BM25/Dense) fetch topically related documents that lack the specific clinical fact needed to make a ruling. The LLM is forced to guess or default to NEI.
**Frequency:** ~30% of all errors (dominant on MedQA and SciFact).
**Concrete Example:**

- *Query:* "Has rituximab been considered as a treatment for chronic fatigue syndrome?" (GT: Yes)
- *Outcome:* BM25 retrieves general documents about rituximab and fatigue, but completely misses the specific clinical trial linking them for this syndrome. The LLM guesses based on the lack of explicit evidence and incorrectly outputs NO.
**Root Cause Analysis:** Corpus limitation or embedding space failure. The specific granular intersection of concepts does not exist in the indexed textbook or abstract chunks.

---

## F3 — Stale Knowledge / Temporal Mismatch (MedChangeQA specific)

**Description:** The claim tests a *recently changed* medical consensus, but the retriever fetches an older document reflecting the *old* consensus. The LLM faithfully follows the stale evidence and gets the question wrong.
**Frequency:** ~60% of errors on MedChangeQA (when recency weighting is disabled).
**Concrete Example:**

- *Query:* "Is cranberry juice effective for treating UTIs?" (GT: SUPPORTED under new guidelines, previously REFUTED).
- *Outcome:* The model retrieves a 2018 textbook stating the evidence is limited (REFUTED).
**Root Cause Analysis:** The BM25 static textbook index lacks document-level publication dates, making it impossible to surface newer guidelines over older ones without explicit PubMed dense retrieval (Configuration G1e).

---

## F4 — LLM Refuses / Hedges Despite Correct Context (Epistemic Conservatism)

**Description:** The retrieved context explicitly supports the correct answer, but the LLM over-hedges ("not explicitly stated," "insufficient evidence") and predicts NOT ENOUGH INFORMATION.
**Frequency:** ~15% of MedChangeQA and HealthFC errors.
**Concrete Example:**

- *Query:* "Is there evidence that tomato juice lowers cholesterol levels?" (GT: Yes)
- *Context/Reasoning:* Document [1] explicitly mentions "tomato juice may lower cholesterol levels."
- *Outcome:* The LLM states that this phrase is "speculative and does not provide concrete evidence," thus confidently predicting NO.
**Root Cause Analysis:** Instruct-tuned medical LLMs are RLHF-trained to be extremely conservative and avoid making definitive medical claims unless the lexical match implies absolute certainty.

---

## F5 — Parametric Knowledge Override (Context Ignored)

**Description:** The LLM actively ignores the retrieved context and answers based on its internal training data—usually because the parametric memory is strong but factually incorrect or outdated.
**Frequency:** ~8% of BioASQ errors.
**Concrete Example:**

- *Query:* "Do statins cause diabetes?" (GT: Yes).
- *Outcome:* The retrieved context contains nuanced evidence about statin-induced diabetes being a known but overestimated risk. However, the model aggressively prioritizes an internal parametric prior, stating 'statins do not cause diabetes' and ignores the nuanced retrieved evidence, incorrectly predicting NO.
**Root Cause Analysis:** When context contains nuance or conflicts with strong pre-training priors, the LLM falls back to parametric "safe" answers, which are prone to overriding the provided evidence.

---

## F6 — Ensemble Disagreement on Ambiguous Claims (Instability)

**Description:** Different models in the heterogeneous ensemble (or different temperature passes of the same model) disagree, leading to a split vote. The majority vote is ultimately incorrect.
**Frequency:** ~25–35% of errors in 3-vote experiments.
**Concrete Example:**

- *Query:* "Can regular exercise prevent or relieve migraine symptoms?" (GT: Mixture).
- *Outcome:* Qwen2.5 votes TRUE, Meditron votes FALSE, Llama3.1 votes TRUE. Final output: TRUE.
**Root Cause Analysis:** Ambiguous claims sit on the decision boundary. Different foundation models have different internal thresholds for what constitutes "sufficient evidence," causing voting instability.

---

## F7 — "Mixture" Class Miscalibration (HealthFC specific) *[Extended]*

**Description:** The pipeline fails to recognize nuanced claims and forces them into binary TRUE/FALSE categories, or conversely, predicts NEI instead of MIXTURE.
**Frequency:** ~45% of HealthFC errors.
**Concrete Example:**

- *Query:* "Can probiotics lower blood sugar in type 2 diabetes?" (GT: TRUE).
- *Outcome:* The retrieved systematic review states that probiotic supplementation significantly decreased fasting blood glucose. However, the model reasons that because probiotics "do not replace standard medication," the claim must be a MIXTURE of true and false, rather than simply evaluating the core claim as TRUE.
**Root Cause Analysis:** The prompt lacks strict enough logic gates for the MIXTURE class. The LLM exhibits premature convergence on caveats rather than directly answering the primary claim.

---

## F8 — Context Over-saturation (KG Distraction) *[Extended]*

**Description:** GraphRAG pulls in multi-hop paths that are factually correct but clinically irrelevant to the specific query, distracting the LLM.
**Frequency:** Responsible for the performance drop when applying KG to BioASQ.
**Concrete Example:**

- *Query:* "Is bradycardia a common side effect of carvedilol?" (GT: Yes)
- *Outcome:* The KG pulls in the entire pharmacological treatment pathway for carvedilol, including hypertension, angina, and heart failure protocols. The LLM gets confused by the sheer volume of mechanistic data and misinterprets the side-effect as a symptom of the underlying diseases rather than the drug.
**Root Cause Analysis:** The 2-hop KG traversal occasionally bridges through generic nodes (e.g., "Inflammation", "Heart Failure"), pulling in massive, clinically-adjacent subgraphs that pollute the MMR compressor.
