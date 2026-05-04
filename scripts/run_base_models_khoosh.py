"""
Khoosh base-model inference runner.

This runner intentionally performs direct model inference only:
- no retrieval
- no atomic decomposition
- no KG

It reuses the same dataset loaders and prompts as eval_all.py so Stage 1
matches the main evaluation setup as closely as possible without RAG.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable
    tqdm.write = print
    sys.modules["tqdm"] = SimpleNamespace(tqdm=tqdm)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ollama_llm import OllamaLLM
from src.evaluation.metrics import compute_classification_metrics
from src.utils.paths import load_yaml_config, project_path


DATASET_CHOICES = ["bioasq", "healthfc"]

DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"

DATASET_PATHS = {
    "bioasq": [
        DATA_ROOT / "BioASQ-training13b" / "test.json",
        DATA_ROOT / "BioASQ-training13b" / "training13b.json",
        PROCESSED_ROOT / "BioASQ-training7b" / "test.json",
    ],
    "healthfc_test": [
        DATA_ROOT / "HealthFC.csv",
        DATA_ROOT / "healthfc_test.csv",
        PROCESSED_ROOT / "healthfc_test.csv",
    ],
}

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


def resolve_dataset_path(name: str) -> str:
    for candidate in DATASET_PATHS[name]:
        if candidate.exists():
            return str(candidate)
    return str(DATASET_PATHS[name][0])


def load_bioasq() -> list:
    with open(resolve_dataset_path("bioasq"), "r", encoding="utf-8") as f:
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
            "id": q.get("id", str(i)),
            "question": q["body"],
            "options": {"A": "Yes", "B": "No"},
            "answer": "A" if exact == "yes" else "B",
        })
    return questions


def load_healthfc() -> list:
    def map_label(raw: str) -> str:
        r = str(raw).strip().lower()
        if r in ("true", "supported", "support", "correct", "0"):
            return "A"
        if r in ("false", "refuted", "refute", "incorrect", "2"):
            return "B"
        return "C"

    questions = []
    with open(resolve_dataset_path("healthfc_test"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if not row.get("en_claim") or not row.get("label"):
                continue
            questions.append({
                "id": f"hfc_test_{i}",
                "question": str(row["en_claim"]),
                "options": {"A": "TRUE", "B": "FALSE", "C": "MIXTURE"},
                "answer": map_label(row["label"]),
                "label": str(row["label"]),
            })
    return questions


def subset_questions(questions: list, percent: float | None) -> list:
    if percent is None:
        return questions
    limit = math.ceil(len(questions) * (percent / 100.0)) if questions else 0
    return questions[:limit]


def answer_without_retrieval(
    llm: OllamaLLM,
    question: str,
    options: dict,
    system_prompt: str,
    votes: int,
    has_maybe: bool,
) -> dict:
    candidates, thoughts = [], []

    for i in range(votes):
        temperature = 0.35 if votes > 1 else None
        result = llm._single_pass(
            question,
            options,
            context="",
            system_prompt=system_prompt,
            temperature=temperature,
        )
        choice = result.get("answer_choice", list(options.keys())[0])
        candidates.append(choice)
        thoughts.append({
            "model": llm.model_name,
            "vote_idx": i + 1,
            "step_by_step_thinking": result.get("step_by_step_thinking", ""),
            "choice": choice,
        })

    vote_counts = Counter(candidates)
    final_answer, _ = vote_counts.most_common(1)[0]
    if has_maybe and len(vote_counts) == len(options) == 3:
        final_answer = "C"

    return {
        "final_answer": final_answer,
        "vote_distribution": dict(vote_counts),
        "thoughts": thoughts,
    }


def save_checkpoint(out_path: str, results: list, correct: int, total: int, config: dict, final: bool = False) -> None:
    cls = compute_classification_metrics(results)
    payload = {
        "config": config,
        "complete": final,
        "progress": f"{len(results)}/{total}",
        "accuracy": (correct / total) if total else 0.0,
        "classification_metrics": cls,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_dataset_bundle(name: str, percent: float | None) -> tuple[list, str, dict | None, bool]:
    if name == "bioasq":
        questions = subset_questions(load_bioasq(), percent)
        return questions, BIOASQ_PROMPT, {"A": "Yes", "B": "No"}, False
    if name == "healthfc":
        questions = subset_questions(load_healthfc(), percent)
        return questions, HEALTHFC_PROMPT, {"A": "True", "B": "False", "C": "Mixture"}, True
    raise ValueError(f"Unsupported dataset: {name}")


def run_eval(
    dataset: str,
    model: str,
    votes: int,
    tag: str,
    experiment_id: str,
    results_dir: str,
    percent: float | None,
) -> int:
    questions, prompt, label_inv, has_maybe = load_dataset_bundle(dataset, percent)
    out_path = os.path.join(results_dir, f"{tag}_{dataset}.json")

    cfg = load_yaml_config(project_path("config", "default.yaml"))
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["name"] = model
    llm = OllamaLLM(cfg)

    run_config = {
        "mode": "base_model_no_retrieval",
        "dataset": dataset,
        "model": model,
        "votes": votes,
        "tag": tag,
        "experiment_id": experiment_id,
        "percent": percent,
        "retrieval_enabled": False,
        "atomic_decomposition_enabled": False,
    }

    print(f"\n{'=' * 70}")
    print(f"  {dataset.upper()} | {model} | {len(questions)} questions")
    print(f"  Output → {out_path}")
    print(f"{'=' * 70}\n")

    done = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                done = {r["id"]: r for r in json.load(f).get("results", [])}
            except Exception:
                done = {}
        if done:
            print(f"  Resuming from checkpoint ({len(done)} already completed)")

    results = list(done.values())
    correct = sum(1 for r in results if r.get("correct"))

    for qi, q in enumerate(tqdm(questions, desc=f"{dataset}:{model}", unit="q")):
        if q["id"] in done:
            continue

        start = time.time()
        try:
            response = answer_without_retrieval(
                llm=llm,
                question=q["question"],
                options=q["options"],
                system_prompt=prompt,
                votes=votes,
                has_maybe=has_maybe,
            )
            pred = response.get("final_answer", "?")
        except Exception as e:
            pred = "?"
            response = {"vote_distribution": {}, "thoughts": []}
            print(f"[ERROR] Q{qi}: {e}")

        elapsed = time.time() - start
        gt_label = q.get("label", q.get("answer", ""))
        pred_label = label_inv.get(pred, pred) if label_inv else pred
        ok = pred == q["answer"]
        if ok:
            correct += 1

        tqdm.write(
            f"  [{'✅' if ok else '❌'}] Q{qi+1:>4} | pred={pred_label:<24} "
            f"gt={gt_label:<24} ({elapsed:.1f}s)"
        )

        rec = {
            "id": q["id"],
            "question": q["question"],
            "gt_answer": q["answer"],
            "gt_label": gt_label,
            "pred_answer": pred,
            "pred_label": pred_label,
            "correct": ok,
            "time_seconds": elapsed,
            "atomic_propositions": "Disabled",
            "retrieved_context": "",
            "vote_distribution": response.get("vote_distribution", {}),
            "reasoning_traces": response.get("thoughts", []),
            "retrieval_metadata": {
                "mode": "base_model_no_retrieval",
                "warnings": [],
            },
        }
        results.append(rec)
        done[q["id"]] = rec
        save_checkpoint(out_path, results, correct, len(questions), run_config)

    save_checkpoint(out_path, results, correct, len(questions), run_config, final=True)
    cls = compute_classification_metrics(results)
    print(
        f"\n  Final Metrics: "
        f"Acc={cls['accuracy']*100:.1f}% "
        f"P={cls['precision']*100:.1f}% "
        f"R={cls['recall']*100:.1f}% "
        f"F1={cls['f1']*100:.1f}%"
    )
    print(f"  Correct: {correct}/{len(questions)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run direct base-model inference for Khoosh ablations"
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--percent", type=float, default=None)
    args = parser.parse_args()

    if args.percent is not None and not (0 < args.percent <= 100):
        parser.error("--percent must be in the range (0, 100].")

    os.makedirs(args.results_dir, exist_ok=True)
    sys.exit(
        run_eval(
            dataset=args.dataset,
            model=args.model,
            votes=args.votes,
            tag=args.tag,
            experiment_id=args.experiment_id,
            results_dir=args.results_dir,
            percent=args.percent,
        )
    )
