"""
ClinProof Universal Evaluation Script
Consolidates all previous eval scripts into one configurable runner.
Saves comprehensive logs (propositions, full context, LLM reasoning, votes) to JSON.
"""
from src.generation.ollama_llm import OllamaLLM
from src.compression.extractor import ExtractiveCompressor
from src.retrieval.pubmed_dense_retriever import PubMedDenseRetriever
from src.retrieval.moe_retriever import MoERetriever
from src.retrieval.graph_retriever import GraphRetriever
import sys
import os
import json
import yaml
import time
from collections import Counter
import pandas as pd

sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/clinproof")

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    # Number of questions per dataset (None for all)
    "count": None,
    "k": 25,                      # Retrieval top-k
    "context_len": 8000,          # Context window size
    # List of models for ensemble. Use 1 model for standard vote
    "models": ["llama3.1:8b", "mistral:7b"],
    # If True, votes are split across models. If False, votes are cast by models[0]
    "ensemble_mode": True,
    "votes": 3,                   # Number of votes
    "use_pubmed": True,           # Enable PubMed MedCPT dense retriever
    "use_graph": True,            # Enable GraphRAG
    "resume": True,               # Resume from interrupted JSON checkpoint
    "tag": "sota_v2"              # Name for the output log file
}

# ─── Prompts ──────────────────────────────────────────────────────────────────
BIOASQ_PROMPT = """You are evaluating a yes/no biomedical claim strictly using the provided documents.

STRICT RULE: Base your answer ONLY on the retrieved documents below.
Do NOT use your training knowledge, general medical expertise, or assumptions.
If the documents do not contain clear evidence either way, answer B (No).

Before answering, reason through BOTH sides using ONLY the documents:
- Evidence supporting YES: (cite specific document content)
- Evidence supporting NO: (cite specific document content, or state "not found in documents")

CRITICAL RULES:
- Do NOT default to Yes because a concept sounds medically plausible.
- Absence of supporting evidence in the documents = No.
- Only answer Yes if a retrieved document explicitly supports it.
- NEGATIVE RESULTS ARE EVIDENCE: Document titles or conclusions containing phrases
  like "not associated", "no significant", "no benefit", "failed to show", "did not",
  "was not" are direct evidence for No. Do not ignore them.
- TESTED IS NOT EFFECTIVE: A document saying drug X was tested or evaluated in a
  trial does NOT mean it is effective. Only answer Yes if the document explicitly
  states the drug showed benefit or was effective.
- READ FULLY: Do not stop at the first positive signal. If the document later
  contradicts or qualifies it, the net result may be No.

Respond with valid JSON only: {"step_by_step_thinking": "...", "answer_choice": "A or B"}
Where A=Yes, B=No."""

PUBMEDQA_PROMPT = """You are evaluating a biomedical research question strictly using the provided documents.

STRICT RULE: Base your answer ONLY on the retrieved documents below.
Do NOT use your training knowledge, general medical expertise, or assumptions.

Before answering, explicitly reason through:
1. What is the specific study, intervention, or clinical outcome the question is asking about?
2. Does the retrieved context provide the ACTUAL RESULTS of this specific study, or just general medical background?
3. What is the evidence supporting YES or NO, citing specific documents?
4. If the documents do not contain the specific study result, or results are inconclusive, you MUST answer MAYBE.

CRITICAL RULES:
- General biological plausibility is NOT sufficient — you need the actual study result.
- Do NOT answer Yes or No based on what seems medically likely.
- NEGATIVE RESULTS ARE EVIDENCE: "not associated", "no significant benefit", "did not
  improve" in a document is direct evidence for No, not Maybe.
- TESTED IS NOT EFFECTIVE: drug X being evaluated in a trial is not evidence it works.
- When in doubt, answer MAYBE.

Respond with valid JSON only: {"step_by_step_thinking": "...", "answer_choice": "A, B, or C"}
Where A=Yes, B=No, C=Maybe."""


def _sanitize_choice(raw_choice, valid_options, default="B"):
    """Clamp model output to a valid option key.
    Handles: "A", "A.", "A=Yes", "Yes", "yes", "B or C", stray letters, etc.
    Falls back to default (No) when nothing valid can be extracted.
    """
    if not raw_choice:
        return default
    # Strip whitespace and take first char uppercased
    cleaned = str(raw_choice).strip().upper()
    # Direct match
    if cleaned in valid_options:
        return cleaned
    # First character match (handles "A." "A=" "A or B" etc.)
    if cleaned and cleaned[0] in valid_options:
        return cleaned[0]
    # Try to match spelled-out words → option values
    lower = str(raw_choice).strip().lower()
    for key, val in valid_options.items():
        if lower == val.lower() or lower.startswith(val.lower()):
            return key
    return default


# ─── Core Logic ──────────────────────────────────────────────────────────────
def answer_with_calibration(llm_list, question, options, context, system_prompt, votes=3, has_maybe=False, is_ensemble=False):
    """Generates answers and saves raw reasoning for the logs."""
    candidates = []
    thoughts = []
    valid_options = set(options.keys())  # e.g. {"A","B"} or {"A","B","C"}
    default_choice = "B"                 # default to No when model hallucinates

    for i in range(votes):
        llm = llm_list[i % len(llm_list)] if is_ensemble else llm_list[0]
        r = llm._single_pass(question, options, context,
                             system_prompt, temperature=0.35)
        raw_choice = r.get("answer_choice", default_choice)
        choice = _sanitize_choice(raw_choice, options, default=default_choice)
        candidates.append(choice)
        thoughts.append({
            "model": llm.model_name,
            "vote_idx": i+1,
            "step_by_step_thinking": r.get("step_by_step_thinking", "Could not parse JSON reasoning"),
            "choice": choice,
            "raw_choice": raw_choice,  # log original for debugging
        })

    vote_counts = Counter(candidates)
    most_common, _ = vote_counts.most_common(1)[0]
    final_ans = most_common

    # Calibrated Maybe: Unanimous disagreement among 3 different answers → Maybe
    if has_maybe and len(vote_counts) == 3:
        final_ans = "C"

    return {
        "final_answer": final_ans,
        "vote_distribution": dict(vote_counts),
        "thoughts": thoughts
    }


def load_dataset(name, count):
    if name == "BioASQ":
        with open("/mnt/d/Harsha/AoLM/project/data/BioASQ-training13b/training13b.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = [{"id": q.get("id", str(i)), "question": q["body"], "options": {"A": "Yes", "B": "No"},
               "answer": "A" if q.get("exact_answer", "").lower() == "yes" else "B"}
              for i, q in enumerate(data["questions"]) if q.get("type") == "yesno" and q.get("exact_answer", "").lower() in ["yes", "no"]]
        return qs[:count] if count else qs
    else:
        df = pd.read_parquet(
            "/mnt/d/Harsha/AoLM/project/data/pubmed_qa_pga_labeled.parquet")
        vm = {"yes": "A", "no": "B", "maybe": "C"}
        qs = [{"id": str(r.get("pubid", i)), "question": r["question"], "options": {"A": "Yes", "B": "No", "C": "Maybe"},
               "answer": vm.get(r["final_decision"].strip().lower(), "C")}
              for i, r in df.iterrows()]
        return qs[:count] if count else qs


def run_eval(name, questions, llm_list, moe, compressor, sys_prompt, out_path, has_maybe=False, enable_pubmed=True, pubmed_mode="pmc"):
    print(f"\n{'='*70}")
    print(f"  {name}  |  {len(questions)} questions")
    print(f"{'='*70}\n")

    # Load checkpoint
    done = {}
    if CONFIG["resume"] and os.path.exists(out_path):
        with open(out_path, "r") as f:
            try:
                done = {r["id"]: r for r in json.load(f).get("results", [])}
            except:
                pass
        if done:
            print(f"Resuming from checkpoint ({len(done)} completed)")

    results = list(done.values())
    correct = sum(1 for r in results if r["correct"])

    for qi, q in enumerate(questions):
        if q["id"] in done:
            continue

        print(f"[{qi+1}/{len(questions)}] {q['question'][:50]}...",
              end=" ", flush=True)
        t = time.time()

        try:
            # Note: MoE retriever now returns (docs, propositions_if_any) based on our GraphRAG changes
            raw_docs, _ = moe.retrieve(q["question"], k1=CONFIG["k"], options=q["options"],
                                       enable_pubmed=enable_pubmed, pubmed_mode=pubmed_mode)
            ctx = compressor.compress(
                q["question"], raw_docs, context_length=CONFIG["context_len"])

            # Extract propositions from context if they were added as doc 0 (GraphRAG trick)
            props = raw_docs[0]["content"] if raw_docs and raw_docs[0].get(
                "title") == "Atomic Propositions" else "None"

            res = answer_with_calibration(llm_list, q["question"], q["options"], ctx, sys_prompt,
                                          votes=CONFIG["votes"], has_maybe=has_maybe, is_ensemble=CONFIG["ensemble_mode"])
            pred = res.get("final_answer", "?")
        except Exception as e:
            print(f"ERROR: {e}")
            pred, res, ctx, props = "?", {}, "Error", "Error"

        ok = pred == q["answer"]
        if ok:
            correct += 1
        elapsed = time.time() - t
        print(
            f"{'✅' if ok else '❌'} pred={pred} gt={q['answer']} ({elapsed:.1f}s)")

        # Comprehensive Log Object
        rec = {
            "id": q["id"],
            "question": q["question"],
            "gt_answer": q["answer"],
            "pred_answer": pred,
            "correct": ok,
            "time_seconds": elapsed,
            "atomic_propositions": props,
            "retrieved_context": ctx,
            "vote_distribution": res.get("vote_distribution", {}),
            "reasoning_traces": res.get("thoughts", [])
        }
        results.append(rec)
        done[q["id"]] = rec

        # Checkpoint every 5 questions
        if len(results) % 5 == 0:
            with open(out_path, "w") as f:
                json.dump({"config": CONFIG, "accuracy": correct /
                          len(results), "results": results}, f, indent=2)

    # Final Save
    acc = correct / len(questions) if questions else 0
    with open(out_path, "w") as f:
        json.dump({"config": CONFIG, "accuracy": acc,
                  "results": results}, f, indent=2)

    print(f"\n  {name} Final Accuracy: {correct}/{len(questions)} ({acc*100:.1f}%)")
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ClinProof Universal Evaluation Runner")
    parser.add_argument("--dataset", type=str, choices=["all", "bioasq", "pubmedqa"], default="all",
                        help="Which dataset to evaluate")
    args = parser.parse_args()

    os.makedirs("/mnt/d/Harsha/AoLM/project/clinproof/results", exist_ok=True)
    cfg = yaml.safe_load(
        open("/mnt/d/Harsha/AoLM/project/clinproof/config/default.yaml"))
    cfg["compression"]["enabled"] = True

    print("\nStarting ClinProof Universal Evaluation")
    print(f"Dataset selected: {args.dataset.upper()}")
    print(json.dumps(CONFIG, indent=2))

    ll_models = []
    for m in CONFIG["models"]:
        m_cfg = dict(cfg)
        m_cfg["model"]["name"] = m
        ll_models.append(OllamaLLM(m_cfg))

    graph = GraphRetriever(cfg["kg"]["graph_path"], cfg,
                           llm=ll_models[0]) if CONFIG["use_graph"] else None
    moe = MoERetriever(graph, None, None, cfg,
                       ollama_client=ll_models[0].client)

    if CONFIG["use_pubmed"]:
        print("Loading PubMed MedCPT retriever...")
        moe.pubmed = PubMedDenseRetriever(cfg)

    compressor = ExtractiveCompressor(cfg)

    # --- Run Evaluations ---
    b_path = f"/mnt/d/Harsha/AoLM/project/clinproof/results/univ_{CONFIG['tag']}_bioasq.json"
    p_path = f"/mnt/d/Harsha/AoLM/project/clinproof/results/univ_{CONFIG['tag']}_pubmedqa.json"

    b_acc, p_acc = None, None
    summary = "\n== Summary =="

    if args.dataset in ["all", "bioasq"]:
        b_acc = run_eval("BioASQ-Y/N", load_dataset("BioASQ", CONFIG["count"]), ll_models, moe,
                         compressor, BIOASQ_PROMPT, b_path, has_maybe=False, enable_pubmed=True, pubmed_mode="pubmed")
        summary += f"\nBioASQ: {b_acc:.1%}"

    if args.dataset in ["all", "pubmedqa"]:
        p_acc = run_eval("PubMedQA", load_dataset(
            "PubMedQA", CONFIG["count"]), ll_models, moe, compressor, PUBMEDQA_PROMPT, p_path, has_maybe=True, enable_pubmed=True, pubmed_mode="pmc")
        summary += f"\nPubMedQA: {p_acc:.1%}"

    print(summary)
