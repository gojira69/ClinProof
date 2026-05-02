# ClinProof Evaluation Report: Current Benchmarks

## 1. Summary Table

| System Configuration | LLM | BioASQ-Y/N | PubMedQA |
|:---|:---|:---:|:---:|
| **ClinProof (Full Dataset)** | mixed | **79.33%** (833/1050) | **53.00%** (530/1000) |
| **ClinProof (100q Subset)** | mixed | **86.00%** (86/100) | **46.00%** (46/100) |
| ─── | ─── | ─── | ─── |
| MedRAG | Llama-3-8B | 82.85% | 70.80% |
| PRG | Llama-3-8B | 84.95% | 69.40% |
| **MedCite** | Llama-3-8B | **84.95%** | **69.40%** |

---

## 2. Qualitative Findings

### A. The "Zero-No" Problem (PubMedQA)
On clinical questions involving "Maybe" (Option C), the system exhibited a significant **Yes-Bias**.
- In the PubMedQA run, the model struggled to confidently assert a definitive "No" even when presented with null results from clinical trials.
- Over-attending to supporting literature fragments while underweighting negative or inconclusive evidence.

### B. High Binary Confidence (BioASQ)
BioASQ performance remains high.
- Structured KG retrieval is good at answering factual "Is/Are" questions where medical entities have clear hierarchical relationships in UMLS/SNOMED/RxNorm.

---

## 3. System Architecture
1. **Knowledge Graph:** Unified UMLS, SNOMED CT, and RxNorm graph.
2. **MoE Retrieval:** Domain-aware routing (Pharmacology, Anatomy, Clinical).
3. **Reasoning:** Double-pass verification with independent thought traces.
4. **Ensemble:** 3-way self-consistency voting in Llama 3.1 8B.

---
