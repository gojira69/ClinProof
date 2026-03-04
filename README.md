# ClinProof: Structured Medical GraphRAG

ClinProof operates over structured knowledge graphs (UMLS, SNOMED CT, RxNorm and PubMed abstracts) currently.

## System Flow & Step-by-Step Example

The following trace demonstrates the internal reasoning flow for the query: *"Is Carvedilol used to treat heart failure?"*

### 1. Atomic Decomposition
The system first breaks down the complex query into fundamental medical propositions and extracts core entities.
- **Entities Extracted:** `carvedilol`, `heart failure`
- **Generated Propositions:**
    - "carvedilol is used to treat heart failure"
    - "carvedilol is a beta-blocker"
    - "carvedilol blocks specific beta-1 receptors"

### 2. Medical Entity Linking
Links raw strings to canonical identifiers in medical databases (e.g., RxNorm, UMLS).
- **'carvedilol'** → Exact Link: `RX:20352` (CARVEDILOL)
- **'heart failure'** → Fuzzy Link: `C0018802` (Heart Failure, Congestive)

### 3. Multi-Hop Graph Retrieval
Fetches structured clinical relationships (Isa, May_Treat, Inverse_Isa) from the Knowledge Graph.
- **Node [Carvedilol phosphate]:**
    - `May_Treat` → `FAILURE HEART`
    - `Isa` → `Adrenergic beta Antagonists`
- **Node [Heart Failure, Congestive]:**
    - `Definition` → "complication of heart diseases; defective cardiac filling and/or impaired contraction..."
    - `Inverse_May_Be_Treated_By` → `Carvedilol phosphate`

### 4. Structured Context Injection
The retrieved graph nodes are formatted into a human-readable context and injected into the LLM prompt.
```markdown
Document [3] (Title: KG: Carvedilol phosphate)
- Par: Adrenergic beta Antagonists
- Inverse Isa: Beta blocking agents, non-selective
- May Treat: FAILURE HEART, HYPERTENSIVE DISEASE, Myocardial Infarction
```

### 5. Verified Generation
The LLM performs a double-pass verification against the injected KG facts to produce the final answer, ensuring every claim is backed by a structured graph edge.
- **Final Pred:** `A (Yes)`
- **Reasoning:** Knowledge Graph explicitly confirms Carvedilol (beta antagonist) treats congestive heart failure.
