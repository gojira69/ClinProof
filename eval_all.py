"""
ClinProof Comprehensive Evaluation Script
==========================================
Evaluates five medical datasets with a hierarchical two-stage retrieval pipeline:
  Stage 1: BM25 keyword matching (Robertson & Zaragoza, 2009)
  Stage 2: MedCPT semantic retrieval (Jin et al., 2023)

Supported datasets:
  - medqa       : MedQA-US 4-option MCQ (test split)
  - medchangeqa : MedChangeQA full dataset (SUPPORTED / REFUTED / NEI)
  - bioasq      : BioASQ-7b yes/no questions
  - scifact     : SciFact claim verification (SUPPORT / CONTRADICT / NEI)
  - healthfc    : HealthFC health claim verification (True / False / Mixture)

Run:
  conda activate aolm_project
  # Single dataset
  python eval_all.py --dataset medqa --count 50 --tag run1
  # All datasets
  python eval_all.py --dataset all --tag run1
  # Custom model / retrieval params
  python eval_all.py --dataset bioasq --model qwen2.5:14b --k 30 --votes 3
"""

import sys, os, json, yaml, time, argparse, logging
from collections import Counter
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ── WSL project path ──────────────────────────────────────────────────────────
PROJECT_ROOT = "/mnt/d/Harsha/AoLM/ClinProof"
DATA_ROOT    = "/mnt/d/Harsha/AoLM/ClinProof/data/processed"
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.bm25_retriever      import BM25Retriever
from src.retrieval.pubmed_dense_retriever import PubMedDenseRetriever
from src.retrieval.graph_retriever     import GraphRetriever, AtomicDecomposer
from src.retrieval.moe_retriever       import MoERetriever
from src.compression.extractor         import ExtractiveCompressor
from src.generation.ollama_llm         import OllamaLLM

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("eval_all")

# ─── Default Config ──────────────────────────────────────────────────────────
CONFIG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "model":            "qwen2.5:14b",   # Best available local model
    "decomp_model":     "medllama2:7b", # Medical model specifically for atomic decomp
    "votes":            3,               # Self-consistency: 3-vote majority
    "ensemble_mode":    True,            # Each vote from the same model (True = round-robin)
    # ── Retrieval ────────────────────────────────────────────────────────────
    "k":                50,              # Stage-2 final top-k docs to compressor
    "bm25_candidates":  200,             # Stage-1 BM25 candidate pool (pre-filter)
    "use_bm25":         True,            # Stage-1: BM25 keyword matching
    "use_pubmed":       False,           # Stage-2: MedCPT FAISS semantic retriever (DISABLED for now)
    "use_graph":        False,           # Stage-2+: KG GraphRAG (requires kg_graph.pkl; enable with --use-graph)
    # ── Recency weighting (for MedChangeQA temporal evaluation) ─────────────
    "recency_alpha":    0.0,             # 0 = disabled; 0.3 = mild; 0.7 = strong
    # ── Context & Compression ────────────────────────────────────────────────
    "context_len":      25000,           # Larger context window for 14b model
    # ── Run Control ──────────────────────────────────────────────────────────
    "count":            None,            # None = evaluate full dataset
    "resume":           True,            # Resume from checkpoint on interruption
    "checkpoint_every": 1,              # Checkpoint frequency (questions)
    "tag":              "fullpower_v1",  # Output file tag
    "results_dir":      f"{PROJECT_ROOT}/results",
    "experiment_id":    "",             # Optional experiment identifier for cross-run analysis
}

# ─── Dataset Paths ────────────────────────────────────────────────────────────
DATASET_PATHS = {
    "medqa":         f"{DATA_ROOT}/medqa-dataset/data_clean/questions/US/test.jsonl",
    "medchangeqa":   f"{DATA_ROOT}/MedChange-main/Datasets/MedChangeQA.csv",
    "bioasq":        f"{DATA_ROOT}/BioASQ-training7b/test.json",
    "scifact_test":  f"{DATA_ROOT}/scifact/claims_test.csv",
    "scifact_train": f"{DATA_ROOT}/scifact/claims_train.csv",
    "healthfc_test":  f"{DATA_ROOT}/healthfc_test.csv",
    "healthfc_train": f"{DATA_ROOT}/healthfc_train.csv",
}

# ─── Prompts ─────────────────────────────────────────────────────────────────
MEDQA_PROMPT = """You are a board-certified physician answering a USMLE-style question.

Use the retrieved medical evidence and your clinical expertise to select the single best answer.

Think step-by-step:
1. Identify the key clinical scenario.
2. Reason through each option.
3. Select the most accurate answer.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A"}"""

MEDCHANGEQA_PROMPT = """You are a critical biomedical expert evaluating whether a medical claim is current and accurate based on the provided evidence.

Your task: evaluate whether the core medical claim is SUPPORTED, REFUTED, or if there is NOT ENOUGH INFORMATION.

Reason step-by-step:
1. What is the core medical claim?
2. Does the provided evidence clearly confirm this claim? If yes, SUPPORTED.
3. Does the provided evidence contradict the claim, or state that the intervention is not recommended, ineffective, or harmful? If yes, REFUTED.
4. Only if the evidence is completely tangential and unrelated to the topic, choose NOT ENOUGH INFORMATION.

CRITICAL RULES:
- REFUTED → If the evidence states a treatment/practice is "not recommended", "ineffective", or "advises against" it, the claim is REFUTED.
- NOT ENOUGH INFORMATION → Use ONLY as a last resort when the documents do not discuss the core components of the claim at all. DO NOT use this just because the evidence lacks detail; make the best clinical judgment possible.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B or C"}
Where A=SUPPORTED, B=REFUTED, C=NOT ENOUGH INFORMATION."""

BIOASQ_PROMPT = """You are a critical biomedical expert evaluating a yes/no biomedical question.

Before answering, explicitly reason through BOTH sides:
1. Evidence explicitly supporting YES
2. Evidence explicitly supporting NO or directly contradicting YES

RULES:
- Base your answer primarily on the provided evidence.
- A "Yes" answer means the evidence confirms the statement.
- A "No" answer means the evidence either contradicts the statement or shows it to be false. 
- Do not default to "No" simply because the evidence is complex, but do not guess "Yes" without foundation.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B"}
Where A=Yes, B=No."""

SCIFACT_PROMPT = """You are a biomedical fact-verification expert. You are given a scientific claim.

Your task: determine whether the claim is SUPPORTED, CONTRADICTED by evidence, or has NOT ENOUGH INFORMATION.

Reason step-by-step:
1. What does the claim assert?
2. Does the retrieved evidence explicitly support or contradict it?
3. Is the evidence sufficient to make a determination?

RULES:
- SUPPORTED     → the evidence clearly confirms the claim.
- CONTRADICTED  → the evidence clearly refutes the claim.
- NOT ENOUGH INFORMATION → evidence is absent, ambiguous, or tangential.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B or C"}
Where A=SUPPORTED, B=CONTRADICTED, C=NOT ENOUGH INFORMATION."""

HEALTHFC_PROMPT = """You are a health claim fact-checker with expertise in evidence-based medicine.

Your task: evaluate whether the health claim is TRUE, FALSE, or a MIXTURE of true and false elements.

Reason step-by-step:
1. What is the core health claim?
2. What evidence supports or refutes it?
3. Is there nuance that requires a MIXTURE verdict?

RULES:
- TRUE    → the claim is clearly supported by medical evidence.
- FALSE   → the claim is clearly contradicted by medical evidence.
- MIXTURE → the claim has both true and false elements, or is context-dependent.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B or C"}
Where A=TRUE, B=FALSE, C=MIXTURE."""

# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_medqa(count: Optional[int]):
    """
    Load MedQA-US test set (JSONL format).
    Each line: {"question": ..., "A": ..., "B": ..., "C": ..., "D": ...,
                 "E": ...,  "answer_idx": "C", "meta_info": "step1"}
    Options are top-level string keys A-E; answer is in "answer_idx".
    """
    path = DATASET_PATHS["medqa"]
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            # Collect option keys A-E that exist and have non-empty text
            opt_map = {k: row[k] for k in ("A", "B", "C", "D", "E") if k in row and row[k]}
            answer_key = str(row.get("answer_idx", "A")).strip().upper()
            if answer_key not in opt_map:
                answer_key = next(iter(opt_map), "A")
            questions.append({
                "id":       str(row.get("id", i)),
                "question": row["question"],
                "options":  opt_map,
                "answer":   answer_key,
            })
    if count:
        questions = questions[:count]
    log.info(f"[MedQA] Loaded {len(questions)} questions")
    return questions


def load_medchangeqa(count: Optional[int]):
    """Load the full MedChangeQA dataset."""
    LABEL_MAP = {"SUPPORTED": "A", "REFUTED": "B", "NOT ENOUGH INFORMATION": "C"}
    OPTIONS   = {"A": "SUPPORTED", "B": "REFUTED", "C": "NOT ENOUGH INFORMATION"}

    df = pd.read_csv(DATASET_PATHS["medchangeqa"])
    df = df.dropna(subset=["Question", "Newest Label"])
    df = df[df["Newest Label"].isin(LABEL_MAP)].reset_index(drop=True)

    questions = []
    for i, row in df.iterrows():
        questions.append({
            "id":       f"q_{i}",
            "question": str(row["Question"]),
            "options":  OPTIONS,
            "answer":   LABEL_MAP[row["Newest Label"]],
            "label":    row["Newest Label"],
        })
    if count:
        questions = questions[:count]
    log.info(f"[MedChangeQA] Loaded {len(questions)} questions")
    return questions


def load_bioasq(count: Optional[int]):
    """Load BioASQ-7b yes/no questions."""
    with open(DATASET_PATHS["bioasq"], "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for i, q in enumerate(data.get("questions", [])):
        if q.get("type") != "yesno":
            continue
        exact = q.get("exact_answer", "")
        if isinstance(exact, list):
            exact = exact[0] if exact else ""
        exact = str(exact).lower().strip()
        if exact not in ("yes", "no"):
            continue
        questions.append({
            "id":       q.get("id", str(i)),
            "question": q["body"],
            "options":  {"A": "Yes", "B": "No"},
            "answer":   "A" if exact == "yes" else "B",
        })
    if count:
        questions = questions[:count]
    log.info(f"[BioASQ] Loaded {len(questions)} yes/no questions")
    return questions


def load_scifact(split: str = "test", count: Optional[int] = None):
    """
    Load SciFact claims.
    Labels: SUPPORT → A, CONTRADICT → B, (no evidence) → C (NEI)
    """
    LABEL_MAP = {"SUPPORT": "A", "CONTRADICT": "B", "": "C"}
    OPTIONS   = {"A": "SUPPORTED", "B": "CONTRADICTED", "C": "NOT ENOUGH INFORMATION"}

    path = DATASET_PATHS[f"scifact_{split}"]
    df = pd.read_csv(path)

    # SciFact may have duplicate claim IDs with multiple evidence rows → deduplicate
    # Keep the row with a label if one exists, else keep the first row
    def pick_row(grp):
        labelled = grp[grp["evidence_label"].notna() & (grp["evidence_label"] != "")]
        return labelled.iloc[0] if len(labelled) else grp.iloc[0]

    df = df.groupby("id", as_index=False, group_keys=False).apply(pick_row).reset_index(drop=True)

    questions = []
    for _, row in df.iterrows():
        raw_label = str(row.get("evidence_label", "")).strip().upper()
        # Map to canonical
        if raw_label == "SUPPORT":
            ans = "A"
        elif raw_label == "CONTRADICT":
            ans = "B"
        else:
            ans = "C"
        questions.append({
            "id":       str(row["id"]),
            "question": str(row["claim"]),
            "options":  OPTIONS,
            "answer":   ans,
            "label":    raw_label if raw_label in ("SUPPORT", "CONTRADICT") else "NEI",
        })
    if count:
        questions = questions[:count]
    log.info(f"[SciFact-{split}] Loaded {len(questions)} claims")
    return questions


def load_healthfc(split: str = "test", count: Optional[int] = None):
    """
    Load HealthFC claims.
    Labels mapped to: True → A, False → B, Mixture → C
    """
    OPTIONS = {"A": "TRUE", "B": "FALSE", "C": "MIXTURE"}

    # Label normalisation
    def map_label(raw: str) -> str:
        r = str(raw).strip().lower()
        if r in ("true", "supported", "support", "correct", "0"):
            return "A"
        if r in ("false", "refuted", "refute", "incorrect", "2"):
            return "B"
        return "C"   # mixture, partially true, unproven, etc. (includes "1")

    path = DATASET_PATHS[f"healthfc_{split}"]
    df = pd.read_csv(path)
    df = df.dropna(subset=["en_claim", "label"]).reset_index(drop=True)

    questions = []
    for i, row in df.iterrows():
        questions.append({
            "id":       f"hfc_{split}_{i}",
            "question": str(row["en_claim"]),
            "options":  OPTIONS,
            "answer":   map_label(row["label"]),
            "label":    str(row["label"]),
        })
    if count:
        questions = questions[:count]
    log.info(f"[HealthFC-{split}] Loaded {len(questions)} claims")
    return questions


# ─── Hierarchical Retriever (BM25 → MedCPT) ──────────────────────────────────

class HierarchicalRetriever:
    """
    Two-stage retrieval:
      Stage 1: BM25Retriever → broad candidate pool (bm25_candidates)
      Stage 2: MoERetriever (wrapping MedCPT) → re-rank to top-k
    """

    def __init__(self, moe: MoERetriever, bm25: Optional[BM25Retriever],
                 use_bm25: bool = True):
        self.moe      = moe
        self.bm25     = bm25
        self.use_bm25 = use_bm25 and (bm25 is not None)

    def retrieve(self, query: str, k: int = 25, bm25_candidates: int = 100,
                 options=None, enable_pubmed: bool = True, pubmed_mode: str = "pubmed",
                 recency_alpha: float = 0.0):
        """
        If BM25 is active:
          1. BM25 retrieves `bm25_candidates` docs (optionally with recency boost).
          2. MoE (MedCPT/graph) re-ranks those candidates to top-k.
        Else:
          MoE retrieves directly.
        """
        if self.use_bm25 and self.bm25 and self.bm25.bm25:
            bm25_docs, _ = self.bm25.retrieve(
                query, k=bm25_candidates, recency_alpha=recency_alpha
            )
            # Inject bm25 docs as a pre-fetched pool into the moe's pubmed retriever
            # For simplicity, we pass the candidate pool to a lightweight re-ranker
            # using MoE's built-in RRF mechanism by running both in parallel
            moe_docs, moe_scores = self.moe.retrieve(
                query, k1=k, options=options,
                enable_pubmed=enable_pubmed, pubmed_mode=pubmed_mode,
            )
            # Merge: BM25 candidates + MoE results via RRF
            merged_docs, merged_scores = self._rrf_merge(
                [(bm25_docs, list(range(len(bm25_docs))), 0.4),
                 (moe_docs,  moe_scores,                  0.6)],
                k=k,
            )
            return merged_docs, merged_scores
        else:
            return self.moe.retrieve(
                query, k1=k, options=options,
                enable_pubmed=enable_pubmed, pubmed_mode=pubmed_mode,
            )

    @staticmethod
    def _rrf_merge(pools, k: int = 25, rrf_k: int = 60):
        rrf_scores, registry = {}, {}
        for docs, scores, weight in pools:
            for rank, doc in enumerate(docs):
                did = (doc.get("PMID") or doc.get("pmid") or
                       doc.get("id")   or doc.get("title", "")[:50])
                rrf_scores[did] = rrf_scores.get(did, 0.0) + weight / (rrf_k + rank + 1)
                if did not in registry:
                    registry[did] = doc
        top = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
        return [registry[i] for i in top], [rrf_scores[i] for i in top]


# ─── Core Evaluation Logic ────────────────────────────────────────────────────

def answer_with_calibration(llm_list, question, options, context, system_prompt,
                             votes=1, is_ensemble=False, has_maybe=False):
    """Run LLM with optional self-consistency voting."""
    candidates, thoughts = [], []

    for i in range(votes):
        llm = llm_list[i % len(llm_list)] if is_ensemble else llm_list[0]
        r   = llm._single_pass(question, options, context, system_prompt, temperature=0.35)
        candidates.append(r.get("answer_choice", list(options.keys())[0]))
        thoughts.append({
            "model":               llm.model_name,
            "vote_idx":            i + 1,
            "step_by_step_thinking": r.get("step_by_step_thinking", ""),
            "choice":              r.get("answer_choice", "?"),
        })

    vote_counts  = Counter(candidates)
    most_common, _ = vote_counts.most_common(1)[0]
    final_ans = most_common

    # 3-way unanimous split on 3-class problem → conservative fallback
    if has_maybe and len(vote_counts) == len(options) == 3:
        final_ans = "C"

    return {
        "final_answer":     final_ans,
        "vote_distribution": dict(vote_counts),
        "thoughts":         thoughts,
    }


def _save_checkpoint(out_path, results, correct, total, config_snap, final=False):
    acc = correct / total if total else 0.0
    payload = {
        "config":   config_snap,
        "complete": final,
        "progress": f"{len(results)}/{total}",
        "accuracy": acc,
        "results":  results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_eval(
    name:          str,
    questions:     list,
    llm_list:      list,
    retriever,
    compressor,
    decomposer,
    sys_prompt:    str,
    out_path:      str,
    config_snap:   dict,
    has_maybe:     bool = False,
    enable_pubmed: bool = True,
    pubmed_mode:   str  = "pubmed",
    label_inv:     dict = None,
) -> float:

    print(f"\n{'='*70}")
    print(f"  {name}  |  {len(questions)} questions")
    print(f"  Output → {out_path}")
    print(f"{'='*70}\n")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    done = {}
    if config_snap.get("resume") and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                done = {r["id"]: r for r in json.load(f).get("results", [])}
            except Exception:
                pass
        if done:
            print(f"  Resuming from checkpoint ({len(done)} already completed)")

    results = list(done.values())
    correct = sum(1 for r in results if r["correct"])
    chk_every = config_snap.get("checkpoint_every", 1)  # changed to 1 so results are always saved

    try:
        # ── Main evaluation loop ───────────────────────────────────────────────
        for qi, q in enumerate(tqdm(questions, desc=name, unit="q")):
            if q["id"] in done:
                continue

            t = time.time()
            try:
                raw_docs, _ = retriever.retrieve(
                    q["question"],
                    k=config_snap["k"],
                    bm25_candidates=config_snap["bm25_candidates"],
                    options=q["options"],
                    enable_pubmed=enable_pubmed,
                    pubmed_mode=pubmed_mode,
                    recency_alpha=config_snap.get("recency_alpha", 0.0),
                )
                
                # Standalone Atomic Decomposition
                # Run this even if GraphRetriever (Knowledge Graph) is disabled.
                props_text = "None"
                if decomposer:
                    entities, propositions = decomposer.decompose(q["question"], q.get("options"))
                    if propositions:
                        props_text = "Key medical claims to verify:\n" + "\n".join(f"- {p}" for p in propositions)
                        # Remove existing atomic propositions from graph retriever if present
                        raw_docs = [d for d in raw_docs if d.get("title") != "Atomic Propositions"]
                        # Prepend the standalone atomic propositions document
                        raw_docs.insert(0, {
                            "title": "Atomic Propositions",
                            "content": props_text,
                            "hop": 0,
                            "PMID": "atomic_decomp"
                        })

                ctx = compressor.compress(
                    q["question"], raw_docs,
                    context_length=config_snap["context_len"]
                )
                props = props_text
                res  = answer_with_calibration(
                    llm_list, q["question"], q["options"], ctx, sys_prompt,
                    votes=config_snap["votes"],
                    is_ensemble=config_snap["ensemble_mode"],
                    has_maybe=has_maybe,
                )
                pred = res.get("final_answer", "?")
            except Exception as e:
                log.error(f"ERROR on Q{qi}: {e}")
                pred, res, ctx, props = "?", {}, "Error", "Error"

            ok      = pred == q["answer"]
            elapsed = time.time() - t
            if ok:
                correct += 1

            pred_label = label_inv.get(pred, pred) if label_inv else pred
            gt_label   = q.get("label", q.get("answer", ""))
            tqdm.write(
                f"  [{'✅' if ok else '❌'}] Q{qi+1:>4} | pred={pred_label:<24} "
                f"gt={gt_label:<24} ({elapsed:.1f}s)"
            )

            rec = {
                "id":                  q["id"],
                "question":            q["question"],
                "gt_answer":           q["answer"],
                "gt_label":            gt_label,
                "pred_answer":         pred,
                "pred_label":          pred_label,
                "correct":             ok,
                "time_seconds":        elapsed,
                "atomic_propositions": props,
                "retrieved_context":   ctx,
                "vote_distribution":   res.get("vote_distribution", {}),
                "reasoning_traces":    res.get("thoughts", []),
            }
            results.append(rec)
            done[q["id"]] = rec

            if len(results) % chk_every == 0:
                _save_checkpoint(out_path, results, correct, len(questions), config_snap)
    except KeyboardInterrupt:
        print("\n  [Interrupt] Checkpoint saving on interruption...")

    acc = correct / len(questions) if questions else 0.0
    _save_checkpoint(out_path, results, correct, len(questions), config_snap, final=True)
    print(f"\n  {name} Final Accuracy: {correct}/{len(questions)} ({acc*100:.1f}%)")
    return acc


# ─── Entry Point ─────────────────────────────────────────────────────────────
DATASET_CHOICES = ["all", "medqa", "medchangeqa", "bioasq", "scifact", "healthfc"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClinProof Comprehensive Evaluation (BM25 + MedCPT)"
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all",
                        help="Dataset(s) to evaluate")
    parser.add_argument("--model",   type=str, default=None,
                        help="Ollama model tag for answering (overrides config)")
    parser.add_argument("--decomp-model", type=str, default=None,
                        help="Ollama model tag for atomic proposition decomposition")
    parser.add_argument("--count",   type=int, default=None,
                        help="Max questions per dataset")
    parser.add_argument("--k",       type=int, default=None,
                        help="Final retrieval top-k")
    parser.add_argument("--votes",   type=int, default=None,
                        help="Self-consistency votes")
    parser.add_argument("--tag",     type=str, default=None,
                        help="Output file tag suffix")
    parser.add_argument("--no-bm25", action="store_true",
                        help="Disable BM25 stage-1 pre-filtering")
    parser.add_argument("--no-pubmed", action="store_true",
                        help="Disable MedCPT/PubMed dense retriever")
    parser.add_argument("--use-graph", action="store_true",
                        help="Enable GraphRAG KG retriever")
    parser.add_argument("--no-kg", action="store_true",
                        help="Disable GraphRAG KG retriever (explicit; overrides --use-graph)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (ignore checkpoint)")
    parser.add_argument("--no-decomp", action="store_true",
                        help="Disable atomic proposition decomposition")
    # ── Ensemble: multiple models for voting (comma-separated) ────────────────
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated Ollama model tags for ensemble voting "
                             "(e.g. 'meditron:7b,medllama2:7b,qwen2.5:14b'). "
                             "Overrides --model. Votes are round-robined across models.")
    # ── Recency-weighted BM25 (for MedChangeQA temporal evaluation) ──────────
    parser.add_argument("--recency-bm25", action="store_true",
                        help="Enable recency-weighted BM25 scoring")
    parser.add_argument("--recency-alpha", type=float, default=0.3,
                        help="Recency boost strength (default=0.3; used when --recency-bm25 is set)")
    # ── Experiment ID for cross-run tracking ─────────────────────────────────
    parser.add_argument("--experiment-id", type=str, default="",
                        help="Unique experiment identifier saved in result JSON for cross-run analysis")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override default results directory")
    args = parser.parse_args()

    # ── Apply CLI overrides ────────────────────────────────────────────────
    if args.model:         CONFIG["model"]         = args.model
    if args.decomp_model:  CONFIG["decomp_model"]  = args.decomp_model
    if args.count:         CONFIG["count"]         = args.count
    if args.k:             CONFIG["k"]             = args.k
    if args.votes:         CONFIG["votes"]         = args.votes
    if args.tag:           CONFIG["tag"]           = args.tag
    if args.no_bm25:       CONFIG["use_bm25"]      = False
    if args.no_pubmed:     CONFIG["use_pubmed"]    = False
    if args.use_graph:     CONFIG["use_graph"]     = True
    if args.no_kg:         CONFIG["use_graph"]     = False   # --no-kg wins
    if args.no_resume:     CONFIG["resume"]        = False
    if args.recency_bm25:  CONFIG["recency_alpha"] = args.recency_alpha
    if args.experiment_id: CONFIG["experiment_id"] = args.experiment_id
    if args.results_dir:   CONFIG["results_dir"]   = args.results_dir

    # ── Build model list: --models overrides --model ───────────────────────
    if args.models:
        model_tags = [m.strip() for m in args.models.split(",") if m.strip()]
        CONFIG["model"]  = model_tags[0]   # primary model shown in summary
        CONFIG["models"] = model_tags
        CONFIG["ensemble_mode"] = True
        log.info(f"Ensemble mode: {model_tags}")
    else:
        CONFIG["models"] = [CONFIG["model"]]

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    cfg = yaml.safe_load(open(f"{PROJECT_ROOT}/config/default.yaml"))
    cfg["compression"]["enabled"] = True

    print("\n" + "="*70)
    print("  ClinProof Comprehensive Evaluation")
    print(f"  Dataset     : {args.dataset.upper()}")
    print(f"  Models      : {CONFIG['models']}")
    print(f"  k={CONFIG['k']}  bm25_candidates={CONFIG['bm25_candidates']}  votes={CONFIG['votes']}")
    print(f"  BM25={CONFIG['use_bm25']}  PubMed/MedCPT={CONFIG['use_pubmed']}  Graph={CONFIG['use_graph']}")
    print(f"  RecencyAlpha={CONFIG['recency_alpha']}  ExpID={CONFIG['experiment_id'] or '(none)'}")
    print("="*70)

    # ── LLM(s) ────────────────────────────────────────────────────────────
    ll_models = []
    for model_tag in CONFIG["models"]:
        m_cfg = dict(cfg)
        m_cfg["model"] = dict(cfg["model"])
        m_cfg["model"]["name"] = model_tag
        ll_models.append(OllamaLLM(m_cfg))
    log.info(f"Loaded {len(ll_models)} LLM(s): {[m.model_name for m in ll_models]}")

    # ── Decomposer LLM ────────────────────────────────────────────────────
    log.info(f"Loading Decomposer LLM ({CONFIG['decomp_model']}) for atomic questions...")
    decomp_cfg = dict(cfg)
    decomp_cfg["model"] = dict(cfg.get("model", {}))
    decomp_cfg["model"]["name"] = CONFIG["decomp_model"]
    decomp_llm = OllamaLLM(decomp_cfg)

    # ── Graph (optional — requires kg_graph.pkl to exist) ─────────────────
    graph = None
    if CONFIG["use_graph"]:
        kg_path = cfg.get("kg", {}).get("graph_path", "")
        if kg_path and os.path.exists(kg_path):
            try:
                log.info(f"Loading GraphRetriever from {kg_path} ...")
                graph = GraphRetriever(kg_path, cfg, llm=decomp_llm)
            except Exception as e:
                log.warning(f"GraphRetriever failed ({e}). Continuing without Graph.")
        else:
            log.warning(f"kg_graph.pkl not found at '{kg_path}'. Skipping GraphRAG.")

    # ── MoE Retriever ─────────────────────────────────────────────────────
    moe = MoERetriever(graph, None, None, cfg,
                       ollama_client=ll_models[0].client)

    # ── MedCPT / PubMed FAISS (Stage-2 semantic) ──────────────────────────
    if CONFIG["use_pubmed"]:
        log.info("Loading PubMed MedCPT dense retriever (Stage-2)...")
        # Override stale cache path from default.yaml
        cfg.setdefault("pubmed", {})
        cfg["pubmed"]["cache_dir"] = f"{PROJECT_ROOT}/data/pubmed_cache"
        os.makedirs(cfg["pubmed"]["cache_dir"], exist_ok=True)
        moe.pubmed = PubMedDenseRetriever(cfg)

    # ── BM25 (Stage-1 keyword) ────────────────────────────────────────────
    bm25 = None
    if CONFIG["use_bm25"]:
        # Always use PROJECT_ROOT — ignore stale path in default.yaml
        corpus_dir = f"{PROJECT_ROOT}/data/corpus"
        try:
            log.info(f"Loading BM25 retriever (Stage-1) from {corpus_dir} ...")
            bm25 = BM25Retriever(corpus_dir, corpus_name="textbooks", cache=True)
            if bm25.bm25 is None:
                log.warning("BM25 index is empty (no textbook chunks found). Skipping Stage-1.")
                bm25 = None
        except Exception as e:
            log.warning(f"BM25 retriever could not be loaded: {e}. Continuing without Stage-1.")
            bm25 = None

    # ── Hierarchical retriever & compressor ───────────────────────────────
    retriever  = HierarchicalRetriever(moe, bm25, use_bm25=CONFIG["use_bm25"])
    compressor = ExtractiveCompressor(cfg)
    decomposer = None
    if not args.no_decomp:
        decomposer = AtomicDecomposer(decomp_llm)

    # ── Run selected datasets ─────────────────────────────────────────────
    summary_lines = ["\n== EVALUATION SUMMARY =="]
    tag = CONFIG["tag"]

    def _path(suffix: str) -> str:
        return os.path.join(CONFIG["results_dir"], f"{tag}_{suffix}.json")

    run = args.dataset

    # ── MedQA ─────────────────────────────────────────────────────────────
    if run in ("all", "medqa"):
        qs  = load_medqa(CONFIG["count"])
        acc = run_eval(
            "MedQA-US", qs, ll_models, retriever, compressor, decomposer,
            MEDQA_PROMPT, _path("medqa"), CONFIG,
            has_maybe=False, enable_pubmed=CONFIG["use_pubmed"],
        )
        summary_lines.append(f"  MedQA-US    : {acc:.1%}")

    # ── MedChangeQA ───────────────────────────────────────────────────────
    if run in ("all", "medchangeqa"):
        MEDCHG_INV = {"A": "SUPPORTED", "B": "REFUTED", "C": "NOT ENOUGH INFORMATION"}
        qs  = load_medchangeqa(CONFIG["count"])
        acc = run_eval(
            "MedChangeQA", qs, ll_models, retriever, compressor, decomposer,
            MEDCHANGEQA_PROMPT, _path("medchangeqa"), CONFIG,
            has_maybe=True, enable_pubmed=CONFIG["use_pubmed"],
            label_inv=MEDCHG_INV,
        )
        summary_lines.append(f"  MedChangeQA : {acc:.1%}")

    # ── BioASQ ────────────────────────────────────────────────────────────
    if run in ("all", "bioasq"):
        BIOASQ_INV = {"A": "Yes", "B": "No"}
        qs  = load_bioasq(CONFIG["count"])
        acc = run_eval(
            "BioASQ-7b (Y/N)", qs, ll_models, retriever, compressor, decomposer,
            BIOASQ_PROMPT, _path("bioasq"), CONFIG,
            has_maybe=False, enable_pubmed=CONFIG["use_pubmed"],
            label_inv=BIOASQ_INV,
        )
        summary_lines.append(f"  BioASQ      : {acc:.1%}")

    # ── SciFact (test split) ──────────────────────────────────────────────
    if run in ("all", "scifact"):
        SCIFACT_INV = {"A": "SUPPORT", "B": "CONTRADICT", "C": "NEI"}
        qs  = load_scifact(split="test", count=CONFIG["count"])
        acc = run_eval(
            "SciFact-Test", qs, ll_models, retriever, compressor, decomposer,
            SCIFACT_PROMPT, _path("scifact_test"), CONFIG,
            has_maybe=True, enable_pubmed=CONFIG["use_pubmed"],
            label_inv=SCIFACT_INV,
        )
        summary_lines.append(f"  SciFact     : {acc:.1%}")

    # ── HealthFC (test split) ─────────────────────────────────────────────
    if run in ("all", "healthfc"):
        HEALTHFC_INV = {"A": "True", "B": "False", "C": "Mixture"}
        qs  = load_healthfc(split="test", count=CONFIG["count"])
        acc = run_eval(
            "HealthFC-Test", qs, ll_models, retriever, compressor, decomposer,
            HEALTHFC_PROMPT, _path("healthfc_test"), CONFIG,
            has_maybe=True, enable_pubmed=CONFIG["use_pubmed"],
            label_inv=HEALTHFC_INV,
        )
        summary_lines.append(f"  HealthFC    : {acc:.1%}")

    print("\n".join(summary_lines))
    print()
