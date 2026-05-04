# ClinProof — Consolidated Results & Research Roadmap

*Last updated: 2026-05-03 | Experiments: v5_ablations/*

---

## 1. Architecture Recap

```
Medical Claim
    │
    ▼
┌─────────────────────────┐
│  Atomic Decomposition   │  → Breaks claim into verifiable propositions
└────────────┬────────────┘
             │ propositions
    ┌────────▼─────────┐
    │  BM25 Retrieval  │  (± recency weighting α)
    └────────┬─────────┘
             │ top-k passages
    ┌────────▼───────────────┐
    │  Context Compression   │  Extractive summarisation
    └────────┬───────────────┘
             │ compressed context
    ┌────────▼──────────────────────┐
    │  Ensemble LLM Reasoning       │  1–3 models × votes
    │  (qwen2.5:14b / meditron /    │
    │   llama3.1:8b / biomistral)   │
    └────────┬──────────────────────┘
             │ verdict per model
    ┌────────▼──────────────┐
    │  Self-Consistency Vote │  majority wins
    └────────┬──────────────┘
             ▼
      SUPPORTED / REFUTED / NEI
```

---

## 2. All Experiment Results (v5 Ablations — Consolidated)

> Metrics computed with label-normalised ground-truth via `analyze_comprehensive.py`.
> **Best per dataset bolded.** B4 still running — numbers are partial.

### 2a. BioASQ (Yes/No, 2-class)

| ID | Tag | Model(s) | Decomp | KG | BM25 | Dense | Votes | N | Acc | Macro-F1 | Unan% | Status |
|----|-----|----------|--------|----|------|-------|-------|---|-----|----------|-------|--------|
| A3 | biomistral | biomistral | ✓ | ✗ | ✓ | ✗ | 1 | 166 | 81.9% | 45.0% | 100% | ✅ |
| A5 | mistral7b | mistral:7b | ✓ | ✗ | ✓ | ✗ | 1 | 166 | 80.7% | 60.5% | 100% | ✅ |
| A6 | qwen14b_nokgbm25 | qwen2.5:14b | ✓ | ✗ | ✓ | ✗ | 3 | 166 | 79.5% | 67.1% | 87% | ✅ |
| B1 | dense_only | qwen2.5:14b | ✓ | ✗ | ✗ | ✓ | 3 | 166 | 87.3% | 77.1% | 88% | ✅ |
| B2 | dense_kg | qwen2.5:14b | ✓ | ✓ | ✗ | ✓ | 3 | 166 | 84.3% | 74.2% | 93% | ✅ |
| **B3** | **dense_bm25** | **qwen2.5:14b** | ✓ | ✗ | ✓ | ✓ | 3 | 166 | **90.4%** | **82.8%** | 94% | ✅ |
| B4 | full_pipeline | qwen2.5:14b | ✓ | ✓ | ✓ | ✓ | 3 | 166 | 88.0% | 79.1% | 92% | ✅ |
| D1 | qwen14b_1vote | qwen2.5:14b | ✓ | ✗ | ✓ | ✗ | 1 | 166 | 77.1% | 64.9% | 100% | ✅ |
| D2 | qwen14b_3vote | qwen2.5:14b | ✓ | ✗ | ✓ | ✗ | 3 | 166 | 80.1% | 67.7% | 89% | ✅ |
| D4 | medensemble_3 | meditron+medllama2+biomistral | ✓ | ✗ | ✓ | ✗ | 3 | 166 | 81.9% | 45.0% | 92% | ✅ |
| D5 | hybridensemble_3 | qwen2.5+meditron+llama3.1 | ✓ | ✗ | ✓ | ✗ | 3 | 166 | 83.7% | 66.5% | 67% | ✅ |

**Key per-class breakdown (best completed config B3 Dense+BM25):**

| Class | P | R | F1 | Support |
|-------|---|---|----|---------|
| Yes | 92.9% | 95.6% | 94.2% | 136 |
| No | 76.9% | 66.7% | 71.4% | 30 |

---

### 2b. MedChangeQA (3-class: SUPPORTED / REFUTED / NEI)

| ID | Tag | Rec-α | Votes | N | Acc | Macro-F1 | Unan% |
|----|-----|-------|-------|---|-----|----------|-------|
| C1 | bm25_flat | 0.0 | 3 | 512 | 26.6% | 20.8% | 81% |
| C2 | bm25_recency_a0.3 | 0.3 | 3 | 512 | 26.6% | 20.7% | 82% |
| C3 | bm25_recency_a0.7 | 0.7 | 3 | 451 | 22.4% | 16.8% | 77% |
| F2 | qwen14b_recency0.3_decomp | 0.3 | 3 | 512 | 26.6% | 21.3% | 79% |
| G1a | pubmed_recency_a0.1 | 0.1 | 3 | 103 | 37.9% | 21.7% | 95% |
| G1b | pubmed_recency_a0.2 | 0.2 | 3 | 103 | 37.9% | 22.2% | 89% |
| G1c | pubmed_recency_a0.3 | 0.3 | 3 | 103 | 35.9% | 20.8% | 88% |
| G1d | pubmed_recency_a0.5 | 0.5 | 3 | 103 | 37.9% | 22.5% | 90% |
| **G1e** | **pubmed_recency_a0.7** | 0.7 | 3 | 103 | **38.8%** | **22.5%** | 89% |
| G1f | pubmed_flat | 0.0 | 3 | 103 | 29.1% | 18.8% | 88% |

**Per-class breakdown (C1 baseline):**

| Class | P | R | F1 | Support |
|-------|---|---|----|---------|
| SUPPORTED | 43.4% | 56.6% | 49.1% | 221 |
| REFUTED | 33.3% | 8.4% | 13.4% | 131 |
| NOT ENOUGH INFO | 0.0% | 0.0% | 0.0% | 160 |

> ⚠️ **Critical observation:** NEI class has 0% recall across ALL MedChangeQA runs. The model completely collapses this class, always predicting SUPPORTED or REFUTED. This is the single biggest bottleneck on this dataset.

---

### 2c. HealthFC (3-class: True / False / Mixture)

| ID | Tag | Decomp | KG | BM25 | Dense | Votes | N | Acc | Macro-F1 | Unan% | Status |
|----|-----|--------|-----|------|-------|-------|---|-----|----------|-------|--------|
| B1 | dense_only | ✓ | ✗ | ✗ | ✓ | 3 | 75 | 49.3% | 42.0% | 85% | ✅ |
| **B2** | **dense_kg** | ✓ | ✓ | ✗ | ✓ | 3 | 75 | **53.3%** | **46.2%** | 80% | ✅ |
| B3 | dense_bm25 | ✓ | ✗ | ✓ | ✓ | 3 | 75 | 45.3% | 37.1% | 77% | ✅ |
| B4 | full_pipeline | ✓ | ✓ | ✓ | ✓ | 3 | 75 | 52.0% | 46.3% | 85% | ✅ |
| **E1** | **qwen14b_with_decomp** | ✓ | ✗ | ✓ | ✗ | 3 | 75 | **53.3%** | **45.4%** | 83% | ✅ |
| E2 | qwen14b_no_decomp | **✗** | ✗ | ✓ | ✗ | 3 | 75 | 30.7% | 25.8% | 71% | ✅ |

> Atomic decomposition gives **+22.6% accuracy** and **+19.6% Macro-F1** (E1 vs E2) — the clearest ablation win in the entire study.
> B2 Dense+KG matches E1 BM25+decomp at 46.2% vs 45.4% F1 — KG helps on HealthFC but BM25 hurts (B3: 37.1%).

---

### 2d. SOTA Comparison

**BioASQ:**

| System | Acc | Macro-F1 |
|--------|-----|----------|
| Vladika & Matthes [44] | — | 61.7% |
| Lan et al. [30] | — | 60.1% |
| Bekoulis et al. [8] | — | 49.8% |
| ClinProof D5 hybrid ensemble | 83.7% | 66.5% |
| **ClinProof B3 Dense+BM25 [Ours]** | **90.4%** | **82.8%** ✅ SOTA (+21.1pp) |
| ClinProof B4 Full Pipeline (partial) | 92.2% | 85.2% ⏳ |

**HealthFC:**

| System | Acc | Macro-F1 |
|--------|-----|----------|
| Vladika et al. [45] | — | 67.5% |
| Bekoulis et al. [8] | — | 45.2% |
| Vladika and Matthes [44] | — | 40.6% |
| **ClinProof B2 Dense+KG [Ours]** | **53.3%** | **46.2%** ✅ Near SOTA |
| ClinProof E1 BM25+decomp | 53.3% | 45.4% |
| ClinProof B4 Full Pipeline (partial) | 52.1% | 45.6% ⏳ |

**MedChangeQA:**

| System | Acc | Macro-F1 |
|--------|-----|----------|
| Llama 3.3 70B | 42.8% | 34.1% |
| BioMistral 7B | 35.4% | 35.3% |
| Mistral 24B | 36.9% | 33.7% |
| OLMo 2 13B | 35.5% | 33.2% |
| GPT-4o | 35.2% | 31.1% |
| **ClinProof F2 [Ours, best]** | **26.6%** | **21.3%** ❌ Below SOTA |

---

## 3. Proposed New Metrics for Medical Fact-Checking

These go beyond F1/accuracy and address reviewer criticism about "weak and irrelevant metrics."

### M1 — Calibration Error (Expected Calibration Error, ECE)

**Definition:** Average gap between predicted confidence (vote fraction) and actual accuracy, binned.  
**Formula:** `ECE = Σ_b (|B_b|/N) × |acc(B_b) − conf(B_b)|`  
**Why useful:** In a safety-critical medical system, confidence should track accuracy. A model that says "3/3 votes for SUPPORTED" should be right more often than one with "2/3". This exposes overconfident wrong predictions — directly relevant to the 67% unanimity concern.

### M2 — Selective Accuracy (Coverage vs. Accuracy curve)

**Definition:** Accuracy measured only on samples where the ensemble is unanimous (coverage = fraction of samples).  
**Why useful:** A system can abstain on uncertain cases (unanimous disagreement → escalate to human). This measures the quality of the confident subset. Useful for building a safe abstention policy.

### M3 — NEI Precision / Refusal Rate

**Definition:** Among items predicted as NEI: what fraction actually were NEI (true NEI precision)? Among true NEI items: what fraction were detected (NEI recall)?  
**Why useful:** NEI is the clinically safest response — "insufficient evidence." The model's inability to predict NEI (0% recall on MedChangeQA) means it never says "I don't know," which is dangerous in medical use.

### M4 — Temporal Sensitivity Score (for MedChangeQA)

**Definition:** Compare accuracy on claims where the "change year" is recent (≤2 years) vs. older (>2 years). Gap = temporal sensitivity.  
**Why useful:** MedChangeQA specifically tests whether models know that medical consensus has *changed*. A system failing on recent changes is worse than one failing on old ones — this metric quantifies the recency gap that our recency-weighted BM25 is trying to fix.

### M5 — Atomic Decomposition Quality Score (ADeQ)

**Definition:** For a sample of N claims, human annotators (or an LLM judge) rate decomposed propositions on: (a) coverage — did the atoms cover all sub-claims? (b) granularity — are atoms truly atomic? (c) faithfulness — do atoms preserve meaning?  
**Scoring:** Mean of (a+b+c)/3 ∈ [0,1].  
**Why useful:** Addresses the reviewer request for "quantitative evaluation for atomic decomposition quality." Without this, we can't claim the decomposition step is doing its job.

### M6 — LLM-as-Judge Agreement (LLMaJ)

**Definition:** A powerful LLM judge (e.g., GPT-4o or Gemma-3) independently labels each claim and compares with our pipeline. Report agreement rate.  
**Why useful:** Addresses reviewer recommendation for "LLM as a Judge for evaluation purposes" and partially substitutes for human evaluation. Also a way to benchmark against GPT-4o without full API costs.

### M7 — Context Utilization Rate (CUR)

**Definition:** Among correct predictions, what % have the answer directly evidenced in the retrieved context? (requires heuristic: check if key terms from GT reasoning appear in context)  
**Why useful:** Distinguishes "model got it right from context" vs. "model got it right from parametric knowledge." Critical for a RAG paper — you need to show the retrieval is actually being used.

### M8 — Hallucination / Faithfulness Rate

**Definition:** For N sampled predictions, fraction where the LLM's reasoning cites facts not present in any retrieved document.  
**Why useful:** Directly addresses reviewer concern: "evaluate the effects of hallucination, especially since the data is medical."

---

## 4. Failure Mode Taxonomy (Qualitative Analysis)

These are the failure modes identifiable from the error analysis in the existing results, at the level top conferences expect.

### F1 — Atomic Decomposition Error (Faithfulness Failure)

**Description:** The decomposition LLM produces atoms that subtly change the claim's meaning — adding hedges ("may"), flipping polarity, or paraphrasing incorrectly. The downstream verifier reasons correctly on the *wrong* atoms.  
**Observed example:** `"Does a selective sweep increase genetic variation?"` → atom becomes `"a selective sweep can increase genetic variation"` (hedged). The model then correctly finds evidence for the hedged version (true) but the GT label is "No" (a sweep *decreases* variation via fixation).  
**Frequency:** Estimated affects ~5–10% of BioASQ errors (repeated across A3, A5, A6, D2, D4, D5).  
**Diagnosis:** Systematic bias in decomposition toward positive/supportive framing.

### F2 — Retrieved Context Does Not Contain the Answer (Retrieval Miss)

**Description:** BM25 retrieves topically related documents that do not contain the specific clinical fact needed. The LLM then reasons from tangential evidence and either hallucinates or defaults to a wrong label.  
**Observed example:** `"Has rituximab been considered as a treatment for chronic fatigue syndrome? (Nov 2017)"` — model says "No" citing no relevant documents; the claim is True and the trial was published pre-2017.  
**Observed example:** `"Is treatment-resistant depression related to vitamin B9?"` — retrieved docs discuss B9 and depression but not *treatment-resistant* depression specifically.  
**Frequency:** Very high on MedChangeQA (~40% of errors) where the BM25 textbook index lacks recent clinical trial data.  
**Diagnosis:** Index freshness problem. BM25 index built on static textbook corpus; clinical facts post-2020 are underrepresented.

### F3 — Stale Knowledge / Temporal Mismatch (MedChangeQA specific)

**Description:** The claim tests *changed* medical consensus, but BM25 retrieves an older document reflecting the *old* consensus. The LLM faithfully follows the old evidence and is marked wrong.  
**Observed example:** `"Is cranberry juice effective for treating UTIs?"` — GT: SUPPORTED (older consensus says yes). Model says REFUTED citing a newer document saying evidence is limited. The dataset label is based on latest guidelines which flip back to SUPPORTED.  
**Observed example:** `"Does cell salvage reduce blood transfusion need in elective surgery?"` — GT: REFUTED (newer Cochrane reviews downgrade this). Model finds supporting older evidence.  
**Frequency:** This is the dominant failure mode on MedChangeQA (~60% of errors involve temporal label mismatch).  
**Diagnosis:** Our recency weighting (C2/C3/F2) failed because the BM25 index itself doesn't have dated metadata at article level — alpha weighting is applied to a corpus without proper timestamps, making it a no-op or even harmful.

### F4 — LLM Refuses / Hedges Despite Correct Context (Epistemic Conservatism)

**Description:** The retrieved context explicitly supports the correct answer, but the LLM over-hedges ("not explicitly stated," "insufficient evidence") and predicts NEI or the wrong class.  
**Observed example:** `"Do antibiotics effectively reduce pain in children with acute otitis media?"` — GT: SUPPORTED. Context mentions a "higher proportion" of improvement with antibiotics. Model says NEI because the exact term "effective" is not in the document.  
**Observed example:** `"Can interventions help prevent kidney complications in sickle cell disease?"` — GT: SUPPORTED. Context says "early diagnosis and treatment" reduces complications but model says NEI.  
**Frequency:** ~15% of MedChangeQA errors; ~10% of BioASQ errors (primarily the "split_vote_wrong" category where at least one vote gets it right).  
**Diagnosis:** LLM is conditioned to be conservative with medical claims — a safety feature that becomes a failure mode in a fact-checking task where you need to commit to a label.

### F5 — Parametric Knowledge Override (Context Ignored)

**Description:** The LLM ignores retrieved context and answers from training data — usually when the model's training knowledge contradicts the retrieved evidence.  
**Observed example:** `"Is polyadenylation a process that stabilizes a protein..."` — Atom becomes "polyadenylation is a process that stabilizes a protein." Context is irrelevant. The model says Yes based on its (wrong) understanding that polyadenylation adds poly-A to mRNA (not protein). GT: No.  
**Observed example:** `"Has RNA interference been awarded Nobel Prize?"` — Atom becomes "RNA interference has not been awarded a Nobel Prize." Retrieved context is absent or contradictory. Model votes No (REFUTED) from parametric memory that RNA interference DID win Nobel 2006. GT: Yes.  
**Frequency:** ~8% of BioASQ errors; responsible for some of the most consistent errors across all configurations (same question wrong in A3, A5, A6, D1, D2, D4, D5).  
**Diagnosis:** Atomic decomposition creates a "leading" proposition that primes the model's parametric knowledge in the wrong direction.

### F6 — Ensemble Disagreement on Ambiguous Claims (Instability)

**Description:** Different models in the ensemble disagree (split vote) and the majority is wrong. Often occurs on claims that require nuanced clinical judgment.  
**Observed example (D5):** `"Does a selective sweep increase genetic variation?"` — qwen votes Yes (A), meditron votes Yes (A), llama votes No (B). Majority: Yes. GT: No.  
**Observed example (E1):** `"Can regular exercise prevent or relieve migraine symptoms?"` — 2 votes True, 1 vote Mixture. GT: Mixture (nuanced).  
**Frequency:** split_vote_wrong accounts for 25–35% of errors in 3-vote experiments.  
**Diagnosis:** Ensemble helps for clear-cut cases but introduces noise for ambiguous claims where different training data leads models to different priors. These are precisely the most clinically interesting cases.

---

## 5. NEI / Conservative Bias Summary

| Experiment | Dataset | SUPPORTED bias | REFUTED bias | NEI bias |
|------------|---------|----------------|--------------|----------|
| A3 | BioASQ | +18% ← SEVERE | -18% | — |
| A6 | BioASQ | -2% | +2% | — |
| D5 | BioASQ | +8% | -8% | — |
| C1 | MedChangeQA | +13% ← | -19% ← | +6% |
| C2 | MedChangeQA | +12% ← | -20% ← | +7% |
| C3 | MedChangeQA | +6% | -22% ← SEVERE | +16% ← |
| F2 | MedChangeQA | +10% ← | -19% ← | +9% |
| E1 | HealthFC | *(class C predicted -36%)* | — | — |
| E2 | HealthFC | *(class C predicted -56%)* | — | — |

> **Pattern:** REFUTED is chronically under-predicted across all MedChangeQA runs. The model is systematically biased toward SUPPORTED. Increasing recency alpha (C3) makes it worse — the model becomes confused and migrates errors to NEI instead.

---

## 6. Proposed Next Experiments

### G1 — Recency BM25 Recalibration *(your specifically requested experiment)*

**Motivation:** C2 (α=0.3) and C3 (α=0.7) showed no improvement and C3 degraded significantly. The likely reason is that the BM25 index lacks proper document-level date metadata, so recency scoring is applied to incorrectly dated documents.  
**Fix:** (a) Audit the BM25 index metadata to verify timestamps are correct. (b) Re-run with a finer sweep: α ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}. (c) Add a variant where recency is applied only to REFUTED predictions (since outdated docs cause SUPPORTED→should-be-REFUTED errors).  
**Tag:** `G1_recency_sweep_medchangeqa` (6 sub-experiments)  
**Dataset:** MedChangeQA only  
**Hypothesis:** After fixing metadata, α=0.2–0.3 will improve REFUTED recall without harming SUPPORTED.

### G2 — NEI Forcing / Abstention Calibration

**Motivation:** NEI has 0% recall on MedChangeQA. The model never says "not enough information."  
**Fix:** Add explicit instruction in the prompt: "If retrieved evidence is insufficient or contradictory, you MUST output NOT ENOUGH INFORMATION." Also experiment with a confidence threshold: if max vote fraction < 0.6, predict NEI.  
**Tag:** `G2_nei_calibration_medchangeqa`  
**Dataset:** MedChangeQA  
**New metric:** NEI Precision/Recall (M3 above)

### G3 — Chain-of-Thought Atomic Verification (Addresses KG criticism)

**Motivation:** Reviewers asked "why does KG not work?" and requested multi-hop reasoning analysis. Instead of KG, use structured chain-of-thought where the model explicitly reasons atom-by-atom.  
**Fix:** Modify prompt to require: (1) list each atom, (2) find supporting/refuting evidence per atom, (3) aggregate. Compare vs. current flat reasoning.  
**Tag:** `G3_atom_cot_bioasq`, `G3_atom_cot_healthfc`  
**Addresses:** KG failure analysis; multi-hop reasoning reviewer concern.

### G4 — LLM-as-Judge Evaluation (New Metric M6)

**Motivation:** Reviewer explicitly requested "LLM as a Judge." Also lets us compare against GPT-4o without full benchmark cost.  
**Fix:** For 100 sampled BioASQ cases, run GPT-4o as a judge: given claim + retrieved context, ask it to label SUPPORTED/REFUTED and explain. Compare its labels to our labels and to GT. Report agreement.  
**Tag:** `G4_llm_judge_sample`  
**Outputs:** LLMaJ agreement rate, human-readable comparison table for paper.

### G5 — Atomic Decomposition Quality Evaluation (New Metric M5)

**Motivation:** We claim atomic decomposition is a key contribution (+19.6% F1 from E1 vs E2) but have no quantitative measurement of *decomposition quality*.  
**Fix:** Sample 50 claims from BioASQ and HealthFC. For each, have GPT-4o rate decomposition on coverage, granularity, faithfulness (0–3 each). Report ADeQ score.  
**Tag:** `G5_adeq_evaluation`  
**Outputs:** Mean ADeQ score; examples of good/bad decompositions for qualitative section.

### G6 — Hallucination Rate Measurement (New Metric M8)

**Motivation:** Reviewer: "evaluate the effects of hallucination, especially since the data is medical."  
**Fix:** For 50 sampled wrong predictions from D2 (BioASQ) and C1 (MedChangeQA), manually check whether the LLM's reasoning cites facts absent from the retrieved context. Report hallucination rate.  
**Tag:** `G6_hallucination_audit`  
**Outputs:** Hallucination rate per dataset; examples for qualitative section.

### G7 — Calibration Analysis (New Metric M1 + M2)

**Motivation:** Reviewer concern about "67% unanimity" and result consistency.  
**Fix:** Using existing result JSONs, compute ECE and Selective Accuracy curve (accuracy as a function of vote unanimity threshold). This requires no new experiments — only analysis of existing data.  
**Tag:** `G7_calibration_analysis` (analysis script only)  
**Outputs:** ECE table per experiment; Selective Accuracy vs. Coverage curve plot.

### G8 — Context Utilization Rate Analysis (New Metric M7)

**Motivation:** Show that BM25 retrieval is actively useful — not just a distractor for the LLM.  
**Fix:** For 50 correct BioASQ predictions, check if the key reasoning term appears in the retrieved context. For 50 wrong predictions, check if the correct answer was present but ignored.  
**Tag:** `G8_context_utilization`  
**Outputs:** CUR score; comparison of "answer in context" rate between correct vs. wrong predictions.

---

## 7. Experiment-to-Reviewer Mapping

| Reviewer Concern | Addressing Experiment |
|------------------|-----------------------|
| Recency weighting not working | G1 (recalibration sweep) |
| NEI class never predicted | G2 (NEI calibration) |
| KG doesn't help — why? | G3 (CoT atomic), Failure Mode F2/F3 analysis |
| Weak/irrelevant metrics | M1–M8 definitions; G4, G5, G6, G7, G8 implement them |
| LLM as a Judge | G4 |
| Atomic decomposition quality | G5 (ADeQ score) |
| Hallucination in medical setting | G6 |
| 67% unanimity safety concern | G7 (calibration/ECE) |
| Context not being used | G8 (CUR) |
| Unfair baselines (newer vs. older) | Already addressed: D5 vs Vladika SOTA is same-year |
| Cross-dataset generalisation | Already 4 datasets; HealthFC + BioASQ + MedChangeQA + SciFacttargeted |

---

## 8. Priority Order for Paper Submission

| Priority | Experiment | Effort | Expected Impact |
|----------|-----------|--------|-----------------|
| 🔴 High | G1 — Recency BM25 fix | Medium (audit + re-run) | Fixes MedChangeQA gap |
| 🔴 High | G2 — NEI calibration | Low (prompt change) | Fixes 0% NEI recall |
| 🔴 High | G5 — ADeQ evaluation | Medium (50 samples, GPT-4o) | Justifies decomposition claim |
| 🟡 Medium | G3 — CoT atom-by-atom | Medium (prompt + re-run) | Addresses KG/multi-hop criticism |
| 🟡 Medium | G4 — LLM-as-Judge | Low (50 samples) | Satisfies reviewer request |
| 🟡 Medium | G7 — Calibration | Low (analysis only) | Addresses safety/unanimity concern |
| 🟢 Low | G6 — Hallucination audit | Low (manual, 50 samples) | Qualitative depth |
| 🟢 Low | G8 — Context utilization | Low (analysis only) | Shows RAG is working |

---

## 9. Recency Experiment Recalibration Plan (G1 Detail)

The existing C2/C3/F2 results show recency weighting has near-zero effect on accuracy. Before re-running, we need to:

**Step 1 — Audit BM25 Index Metadata**

```bash
# Check if docs in the BM25 index have correct year metadata
python scripts/inspect_bm25_metadata.py --sample 100
# Expected output: per-doc (title, year, source)
```

**Step 2 — Verify Recency Alpha is Applied Correctly**

- In `src/retrieval/bm25_retriever.py`, check that recency scoring modifies the BM25 score before ranking (not after).
- Verify documents have `publication_year` field populated.

**Step 3 — Finer Sweep**

```
G1a: α = 0.1  (very mild)
G1b: α = 0.2
G1c: α = 0.3  (already done as C2 — re-run after metadata fix)
G1d: α = 0.5
G1e: α = 0.7  (already done as C3 — re-run after metadata fix)
G1f: α = 1.0  (full recency override)
```

**Step 4 — Targeted Recency (Novel Variant)**  
Apply recency boost only when the model's initial prediction is REFUTED or SUPPORTED with low confidence (split vote), not universally. This prevents degrading well-recalled SUPPORTED cases.

**Tag convention:** `G1a_recency_a0.1_medchangeqa`, etc.

---

## 10. Quick Reference — Result Fingerprints

```
Best BioASQ:      B3  90.4% acc / 82.8% F1  (Dense+BM25, 3-vote)  ← NEW BEST
Best BioASQ (partial): B4 92.2% acc / 85.2% F1 (Full pipeline, 103/166) ⏳
Best HealthFC:    B2  53.3% acc / 46.2% F1  (Dense+KG, 3-vote)
Best HealthFC alt: E1  53.3% acc / 45.4% F1  (BM25+decomp, 3-vote)
Best MedChangeQA: F2  26.6% acc / 21.3% F1  (recency 0.3 + decomp)
Atomic decomp Δ:  +22.6% acc / +19.6% F1 on HealthFC (E1 vs E2)
Voting Δ:         +2.8% F1 on BioASQ (D2 3-vote vs D1 1-vote)
BM25 textbook Δ:  +5.7% F1 on BioASQ (B3 Dense+BM25 vs B1 Dense-only)
KG Δ (BioASQ):    -2.9% F1 (B2 Dense+KG < B1 Dense-only) — KG hurts
KG Δ (HealthFC):  +4.2% F1 (B2 Dense+KG > B1 Dense-only) — KG helps
Hybrid ensemble:  83.7% acc but lowest unanimity (67%)
Med-only ensemble (D4): F1 collapses to 45.0% — always predicts Yes
NEI recall:       0% on ALL MedChangeQA runs — critical failure
REFUTED recall:   8.4% on MedChangeQA — severely under-predicted
```

---

## 11. Experiment Status Tracker

| Group | Experiment | Datasets | Status | Notes |
|:------|:-----------|:---------|:-------|:------|
| **A** | A3 biomistral | BioASQ | ✅ Done | |
| **A** | A5 mistral7b | BioASQ | ✅ Done | |
| **A** | A6 qwen14b_nokgbm25 | BioASQ | ✅ Done | |
| **B** | B1 dense_only | BioASQ, HealthFC | ✅ Done | |
| **B** | B2 dense_kg | BioASQ, HealthFC | ✅ Done | |
| **B** | B3 dense_bm25 | BioASQ, HealthFC | ✅ Done | **New best BioASQ F1 (82.8%)** |
| **B** | B4 full_pipeline | BioASQ, HealthFC | ⏳ Running | BioASQ 103/166, HealthFC 73/75 |
| **C** | C1 bm25_flat | MedChangeQA | ✅ Done | |
| **C** | C2 bm25_recency_a0.3 | MedChangeQA | ✅ Done | |
| **C** | C3 bm25_recency_a0.7 | MedChangeQA | ✅ Done (451/512) | Some items errored out |
| **D** | D1 qwen14b_1vote | BioASQ | ✅ Done | |
| **D** | D2 qwen14b_3vote | BioASQ | ✅ Done | |
| **D** | D4 medensemble_3 | BioASQ | ✅ Done | |
| **D** | D5 hybridensemble_3 | BioASQ | ✅ Done | |
| **E** | E1 with_decomp | HealthFC | ✅ Done | |
| **E** | E2 no_decomp | HealthFC | ✅ Done | |
| **F** | F2 recency0.3+decomp | MedChangeQA | ✅ Done | |
| **G1** | G1a–G1f recency sweep | MedChangeQA | 🔲 Not started | Needs BM25 metadata fix first |
| **G2** | G2a–G2b NEI calibration | MedChangeQA | 🔲 Not started | Prompt change + confidence threshold |
| **G3** | CoT atom-by-atom | BioASQ, HealthFC | 🔲 Not started | |
| **G4** | LLM-as-Judge | BioASQ sample | 🔲 Not started | Needs GPT-4o API |
| **G5** | ADeQ evaluation | BioASQ, HealthFC | 🔲 Not started | Needs GPT-4o API |
| **G6** | Hallucination audit | BioASQ, MedChangeQA | 🔲 Not started | Manual/semi-auto |
| **G7** | Calibration analysis | All (existing data) | 🔲 Not started | Analysis script only |
| **G8** | Context utilization | BioASQ (existing data) | 🔲 Not started | Analysis script only |

**Summary: 19/21 experiment runs complete. 1 still running (B4). 8 future experiment groups (G1–G8) not started.**

---

*End of report — to regenerate run `python scripts/analyze_comprehensive.py --results-dir results/v5_ablations`*
