"""
ClinProof MedChangeQA Evaluation Script
Evaluates on MedChangeQA.csv (SUPPORTED / REFUTED / NOT ENOUGH INFORMATION).
Saves a rolling JSON checkpoint that analyze_medrevqa.py can read at any time.

Run:
  conda activate aolm_project
  python eval_medchangeqa.py [--count N] [--tag myrun]
"""

import sys, os, json, yaml, time, argparse
from collections import Counter
import pandas as pd

sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/clinproof")
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.moe_retriever import MoERetriever
from src.retrieval.pubmed_dense_retriever import PubMedDenseRetriever
from src.compression.extractor import ExtractiveCompressor
from src.generation.ollama_llm import OllamaLLM

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    "count": None,                  # Number of questions to evaluate (None = all)
    "k": 25,                        # Retrieval top-k
    "context_len": 8000,            # Context window size
    "models": ["mistral:7b"],
    "ensemble_mode": False,          # If True votes split across models; else models[0]
    "votes": 1,                     # Number of votes per question
    "use_pubmed": False,
    "use_graph": False,
    "resume": True,                 # Resume from JSON checkpoint
    "tag": "medchangeqa_v1",        # Suffix for output file name
    "checkpoint_every": 5,          # Save every N questions
    "dataset_path": "/mnt/d/Harsha/AoLM/project/data/MedChange-main/Datasets/MedChangeQA.csv",
}

# ─── Prompt ──────────────────────────────────────────────────────────────────
# MEDREVQA_PROMPT = """You are a critical biomedical expert evaluating a factual medical claim.
# 
# You are given a QUESTION and RETRIEVED EVIDENCE from medical knowledge sources.
# Your task: determine whether the claim in the question is SUPPORTED, REFUTED, or
# lacks sufficient evidence (NOT ENOUGH INFORMATION).
# 
# Before answering, explicitly reason through BOTH sides:
# - Evidence supporting the claim: (list what in the context supports it)
# - Evidence against the claim: (list what in the context contradicts it)
# 
# CRITICAL RULES:
# - Base your answer ONLY on the retrieved evidence provided.
# - Do NOT use your internal medical knowledge or assumptions.
# - General plausibility does NOT equal SUPPORTED.
# - SUPPORTED   → the evidence clearly and explicitly confirms the claim.
# - REFUTED     → the evidence clearly and explicitly contradicts the claim.
# - NOT ENOUGH INFORMATION → the evidence is absent, ambiguous, or only tangentially related.
# 
# Respond with valid JSON only:
# {"step_by_step_thinking": "...", "answer_choice": "A, B, or C"}
# Where A=SUPPORTED, B=REFUTED, C=NOT ENOUGH INFORMATION."""

MEDREVQA_PROMPT = """You are a critical biomedical expert evaluating a factual medical claim.

You are given a QUESTION.
Your task: using your internal medical knowledge, determine whether the claim in the question is SUPPORTED, REFUTED, or lacks sufficient evidence (NOT ENOUGH INFORMATION).

Before answering, explicitly reason through BOTH sides based on your clinical knowledge.

CRITICAL RULES:
- Use your internal medical knowledge to answer.
- SUPPORTED   → established medical facts clearly confirm the claim.
- REFUTED     → established medical facts clearly contradict the claim.
- NOT ENOUGH INFORMATION → there is no clear consensus or insufficient evidence in medical science.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A, B, or C"}
Where A=SUPPORTED, B=REFUTED, C=NOT ENOUGH INFORMATION."""

LABEL_MAP = {"SUPPORTED": "A", "REFUTED": "B", "NOT ENOUGH INFORMATION": "C"}
LABEL_INV  = {v: k for k, v in LABEL_MAP.items()}
OPTIONS    = {"A": "SUPPORTED", "B": "REFUTED", "C": "NOT ENOUGH INFORMATION"}

# ─── Core Logic ──────────────────────────────────────────────────────────────
def answer_with_calibration(llm_list, question, options, context, system_prompt,
                             votes=3, is_ensemble=False):
    candidates = []
    thoughts   = []

    for i in range(votes):
        llm = llm_list[i % len(llm_list)] if is_ensemble else llm_list[0]
        r   = llm._single_pass(question, options, context, system_prompt, temperature=0.35)
        candidates.append(r.get("answer_choice", "C"))
        thoughts.append({
            "model": llm.model_name,
            "vote_idx": i + 1,
            "step_by_step_thinking": r.get("step_by_step_thinking",
                                           "Could not parse JSON reasoning"),
            "choice": r.get("answer_choice", "?"),
        })

    vote_counts = Counter(candidates)
    most_common, _ = vote_counts.most_common(1)[0]
    final_ans = most_common

    # Unanimous 3-way split → fall back to NOT ENOUGH INFORMATION (most conservative)
    if len(vote_counts) == 3:
        final_ans = "C"

    return {
        "final_answer": final_ans,
        "vote_distribution": dict(vote_counts),
        "thoughts": thoughts,
    }


def load_dataset(path, count):
    """Load MedChangeQA CSV and return list of question dicts."""
    df = pd.read_csv(path)
    # Drop rows with missing essential fields
    df = df.dropna(subset=["Question", "Newest Label"])
    df = df[df["Newest Label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

    questions = []
    for i, row in df.iterrows():
        # Treat the index as ID if missing
        pmid = str(i)
        
        abstract_prefix = ""
        questions.append({
            "id":           f"q_{i}",
            "pmid":         pmid,
            "question":     str(row["Question"]),
            "raw_question": str(row["Question"]),
            "abstract":     abstract_prefix,        # stored for logging only
            "options":      OPTIONS,
            "answer":       LABEL_MAP[row["Newest Label"]],
            "label":        row["Newest Label"],
            "author":       "N/A",
            "doi_date":     "N/A",
        })

    if count:
        questions = questions[:count]
    return questions


def run_eval(questions, llm_list, moe, compressor, out_path):
    print(f"\n{'='*70}")
    print(f"  MedChangeQA  |  {len(questions)} questions")
    print(f"{'='*70}\n")

    # ── Checkpoint resume ─────────────────────────────────────────────────
    done = {}
    if CONFIG["resume"] and os.path.exists(out_path):
        with open(out_path, "r") as f:
            try:
                done = {r["id"]: r for r in json.load(f).get("results", [])}
            except Exception:
                pass
        if done:
            print(f"Resuming from checkpoint ({len(done)} already completed)")

    results = list(done.values())
    correct = sum(1 for r in results if r["correct"])

    for qi, q in enumerate(questions):
        if q["id"] in done:
            continue

        print(f"[{qi+1}/{len(questions)}] {q['raw_question'][:60]}...", end=" ", flush=True)
        t = time.time()

        try:
            raw_docs, _ = moe.retrieve(
                q["question"], k1=CONFIG["k"],
                options=q["options"],
                enable_pubmed=CONFIG["use_pubmed"],
                pubmed_mode="pubmed",
            )
            ctx = compressor.compress(q["question"], raw_docs,
                                      context_length=CONFIG["context_len"])

            props = (raw_docs[0]["content"]
                     if raw_docs and raw_docs[0].get("title") == "Atomic Propositions"
                     else "None")

            res  = answer_with_calibration(
                llm_list, q["question"], q["options"], ctx, MEDREVQA_PROMPT,
                votes=CONFIG["votes"], is_ensemble=CONFIG["ensemble_mode"],
            )
            pred = res.get("final_answer", "?")
        except Exception as e:
            print(f"ERROR: {e}")
            pred, res, ctx, props = "?", {}, "Error", "Error"

        ok      = pred == q["answer"]
        elapsed = time.time() - t
        if ok:
            correct += 1
        print(f"{'✅' if ok else '❌'} pred={LABEL_INV.get(pred, pred):<24}"
              f" gt={q['label']:<24} ({elapsed:.1f}s)")

        rec = {
            "id":               q["id"],
            "pmid":             q["pmid"],
            "question":         q["raw_question"],
            "abstract":         q["abstract"],
            "gt_label":         q["label"],
            "gt_answer":        q["answer"],
            "pred_answer":      pred,
            "pred_label":       LABEL_INV.get(pred, pred),
            "correct":          ok,
            "time_seconds":     elapsed,
            "atomic_propositions": props,
            "retrieved_context":   ctx,
            "vote_distribution":   res.get("vote_distribution", {}),
            "reasoning_traces":    res.get("thoughts", []),
            "author":           q["author"],
            "doi_date":         q["doi_date"],
        }
        results.append(rec)
        done[q["id"]] = rec

        # ── Rolling checkpoint ────────────────────────────────────────────
        if len(results) % CONFIG["checkpoint_every"] == 0:
            _save(out_path, results, correct, len(results))

    # ── Final save ────────────────────────────────────────────────────────
    acc = correct / len(questions) if questions else 0.0
    _save(out_path, results, correct, len(questions), final=True)
    print(f"\n  MedChangeQA Final Accuracy: {correct}/{len(questions)} ({acc*100:.1f}%)")
    return acc


def _save(path, results, correct, total, final=False):
    acc = correct / total if total else 0.0
    payload = {
        "config":   CONFIG,
        "complete": final,
        "progress": f"{len(results)}/{total}",
        "accuracy": acc,
        "results":  results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClinProof MedChangeQA Evaluator")
    parser.add_argument("--count", type=int, default=None,
                        help="Limit questions (default: all)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override result file tag")
    args = parser.parse_args()

    if args.count:
        CONFIG["count"] = args.count
    if args.tag:
        CONFIG["tag"] = args.tag

    os.makedirs("/mnt/d/Harsha/AoLM/project/clinproof/results", exist_ok=True)
    cfg = yaml.safe_load(open("/mnt/d/Harsha/AoLM/project/clinproof/config/default.yaml"))
    cfg["compression"]["enabled"] = True

    out_path = f"/mnt/d/Harsha/AoLM/project/clinproof/results/{CONFIG['tag']}.json"

    print("\nStarting ClinProof MedChangeQA Evaluation")
    print(f"Output: {out_path}")
    print(json.dumps(CONFIG, indent=2))

    ll_models = []
    for m in CONFIG["models"]:
        m_cfg = dict(cfg)
        m_cfg["model"]["name"] = m
        ll_models.append(OllamaLLM(m_cfg))

    graph = (GraphRetriever(cfg["kg"]["graph_path"], cfg, llm=ll_models[0])
             if CONFIG["use_graph"] else None)
    moe   = MoERetriever(graph, None, None, cfg, ollama_client=ll_models[0].client)

    if CONFIG["use_pubmed"]:
        print("Loading PubMed MedCPT retriever...")
        moe.pubmed = PubMedDenseRetriever(cfg)

    compressor = ExtractiveCompressor(cfg)

    questions = load_dataset(CONFIG["dataset_path"], CONFIG["count"])
    run_eval(questions, ll_models, moe, compressor, out_path)
