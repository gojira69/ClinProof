#!/usr/bin/env python3
"""
Run local Ollama medical fact-checking experiments on:
  - HealthFC
  - MedChangeQA
  - BioASQ yes/no only

Two evaluation sets are supported in one script:
  1. Base models only:
     - qwen2.5:7b
     - medllama2:7b
  2. Full pipeline:
     - atomic decomposition
     - KG-RAG
     - extractive compression
     - majority voting across:
       meditron:latest, qwen2.5:7b, medllama2:7b

This script talks to the Ollama HTTP API directly, so it does not require the
Python `ollama` package. It assumes `ollama serve` is already running.

Example:
  python run_ollama_factcheck.py --n 10
  python run_ollama_factcheck.py --datasets healthfc,medchangeqa,bioasq --n 25
  python run_ollama_factcheck.py --ollama-host http://127.0.0.1:11434
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_yaml_config, project_path, resolve_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("run_ollama_factcheck")


MEDCHANGEQA_PROMPT = """You are a critical biomedical expert evaluating whether a medical claim is current and accurate based on the provided evidence.

Your task: evaluate whether the core medical claim is SUPPORTED, REFUTED, or if there is NOT ENOUGH INFORMATION.

Reason step-by-step:
1. What is the core medical claim?
2. Does the provided evidence clearly confirm this claim? If yes, SUPPORTED.
3. Does the provided evidence contradict the claim, or state that the intervention is not recommended, ineffective, or harmful? If yes, REFUTED.
4. Only if the evidence is completely tangential and unrelated to the topic, choose NOT ENOUGH INFORMATION.

CRITICAL RULES:
- REFUTED -> If the evidence states a treatment/practice is "not recommended", "ineffective", or "advises against" it, the claim is REFUTED.
- NOT ENOUGH INFORMATION -> Use ONLY as a last resort when the documents do not discuss the core components of the claim at all. DO NOT use this just because the evidence lacks detail; make the best clinical judgment possible.

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


HEALTHFC_PROMPT = """You are a health claim fact-checker with expertise in evidence-based medicine.

Your task: evaluate whether the health claim is TRUE, FALSE, or a MIXTURE of true and false elements.

Reason step-by-step:
1. What is the core health claim?
2. What evidence supports or refutes it?
3. Is there nuance that requires a MIXTURE verdict?

RULES:
- TRUE -> the claim is clearly supported by medical evidence.
- FALSE -> the claim is clearly contradicted by medical evidence.
- MIXTURE -> the claim has both true and false elements, or is context-dependent.

Respond with valid JSON only:
{"step_by_step_thinking": "...", "answer_choice": "A or B or C"}
Where A=TRUE, B=FALSE, C=MIXTURE."""


DATASET_PATHS = {
    "medchangeqa": [
        PROJECT_ROOT / "data" / "MedChangeQA.csv",
        PROJECT_ROOT / "data" / "processed" / "MedChange-main" / "Datasets" / "MedChangeQA.csv",
    ],
    "bioasq": [
        PROJECT_ROOT / "data" / "BioASQ-training13b" / "test.json",
        PROJECT_ROOT / "data" / "BioASQ-training13b" / "training13b.json",
        PROJECT_ROOT / "data" / "processed" / "BioASQ-training7b" / "test.json",
    ],
    "healthfc": [
        PROJECT_ROOT / "data" / "HealthFC.csv",
        PROJECT_ROOT / "data" / "healthfc_test.csv",
        PROJECT_ROOT / "data" / "processed" / "healthfc_test.csv",
    ],
}


@dataclass
class DatasetSpec:
    name: str
    prompt: str
    labels: dict[str, str]
    tie_break_choice: str | None


DATASET_SPECS = {
    "healthfc": DatasetSpec(
        name="healthfc",
        prompt=HEALTHFC_PROMPT,
        labels={"A": "TRUE", "B": "FALSE", "C": "MIXTURE"},
        tie_break_choice="C",
    ),
    "medchangeqa": DatasetSpec(
        name="medchangeqa",
        prompt=MEDCHANGEQA_PROMPT,
        labels={"A": "SUPPORTED", "B": "REFUTED", "C": "NOT ENOUGH INFORMATION"},
        tie_break_choice="C",
    ),
    "bioasq": DatasetSpec(
        name="bioasq",
        prompt=BIOASQ_PROMPT,
        labels={"A": "YES", "B": "NO"},
        tie_break_choice=None,
    ),
}


def resolve_dataset_path(name: str) -> Path:
    for candidate in DATASET_PATHS[name]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset for {name}: {DATASET_PATHS[name]}")


def load_medchangeqa(limit: int | None, start_idx: int = 0) -> list[dict[str, Any]]:
    label_map = {
        "SUPPORTED": "A",
        "REFUTED": "B",
        "NOT ENOUGH INFORMATION": "C",
    }
    options = {"A": "SUPPORTED", "B": "REFUTED", "C": "NOT ENOUGH INFORMATION"}
    rows: list[dict[str, Any]] = []
    with open(resolve_dataset_path("medchangeqa"), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            question = (row.get("Question") or "").strip()
            label = (row.get("Newest Label") or "").strip()
            if not question or label not in label_map:
                continue
            rows.append(
                {
                    "id": f"medchangeqa_{row_idx}",
                    "question": question,
                    "options": options,
                    "answer": label_map[label],
                    "label": label,
                }
            )
    return slice_rows(rows, limit, start_idx)


def load_bioasq(limit: int | None, start_idx: int = 0) -> list[dict[str, Any]]:
    with open(resolve_dataset_path("bioasq"), encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: list[dict[str, Any]] = []
    for idx, question in enumerate(payload.get("questions", [])):
        if question.get("type") != "yesno":
            continue
        exact = question.get("exact_answer", "")
        if isinstance(exact, list):
            exact = exact[0] if exact else ""
        exact = str(exact).strip().lower()
        if exact not in {"yes", "no"}:
            continue
        rows.append(
            {
                "id": str(question.get("id", f"bioasq_{idx}")),
                "question": question.get("body", "").strip(),
                "options": {"A": "Yes", "B": "No"},
                "answer": "A" if exact == "yes" else "B",
                "label": exact.upper(),
            }
        )
    return slice_rows(rows, limit, start_idx)


def load_healthfc(limit: int | None, start_idx: int = 0) -> list[dict[str, Any]]:
    options = {"A": "TRUE", "B": "FALSE", "C": "MIXTURE"}

    def map_label(raw: str) -> str:
        normalized = str(raw).strip().lower()
        if normalized in {"true", "supported", "support", "correct", "0"}:
            return "A"
        if normalized in {"false", "refuted", "refute", "incorrect", "2"}:
            return "B"
        return "C"

    rows: list[dict[str, Any]] = []
    with open(resolve_dataset_path("healthfc"), newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            claim = (row.get("en_claim") or row.get("\ufeffen_claim") or "").strip()
            label = (row.get("label") or "").strip()
            if not claim or not label:
                continue
            rows.append(
                {
                    "id": f"healthfc_{row_idx}",
                    "question": claim,
                    "options": options,
                    "answer": map_label(label),
                    "label": label,
                }
            )
    return slice_rows(rows, limit, start_idx)


def slice_rows(rows: list[dict[str, Any]], limit: int | None, start_idx: int) -> list[dict[str, Any]]:
    if start_idx:
        rows = rows[start_idx:]
    if limit is not None:
        rows = rows[:limit]
    return rows


class LocalOllamaLLM:
    def __init__(
        self,
        model_name: str,
        host: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        timeout_sec: int = 180,
        context_length: int = 32768,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec
        self.context_length = context_length

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.pop("temperature", self.temperature),
                "num_predict": kwargs.pop("num_predict", self.max_new_tokens),
            },
        }
        format_out = kwargs.pop("format", None)
        if format_out:
            payload["format"] = format_out
        if kwargs:
            payload["options"].update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return raw.get("message", {}).get("content", "").strip()
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code} for model={self.model_name}: {detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from exc

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}


def ask_model(
    llm: LocalOllamaLLM,
    question: str,
    options: dict[str, str],
    system_prompt: str,
    context: str = "",
    temperature: float = 0.0,
) -> dict[str, Any]:
    options_text = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
    if context:
        user_content = (
            f"Relevant evidence:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            'Respond with JSON only: {"step_by_step_thinking": "...", "answer_choice": "A"}'
        )
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            'Respond with JSON only: {"step_by_step_thinking": "...", "answer_choice": "A"}'
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw = llm.generate(messages, format="json", temperature=temperature)
    parsed = llm.extract_json(raw)
    answer_choice = str(parsed.get("answer_choice", "")).strip().upper()
    if answer_choice not in options:
        match = re.search(rf"\b({'|'.join(map(re.escape, options.keys()))})\b", raw, re.IGNORECASE)
        answer_choice = match.group(1).upper() if match else next(iter(options.keys()))
    return {
        "answer_choice": answer_choice,
        "step_by_step_thinking": parsed.get("step_by_step_thinking", raw),
        "raw_output": raw,
    }


def run_voting(
    models: list[LocalOllamaLLM],
    question: str,
    options: dict[str, str],
    system_prompt: str,
    context: str,
    votes: int,
    tie_break_choice: str | None = None,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    candidates: list[str] = []
    for vote_idx in range(votes):
        llm = models[vote_idx % len(models)]
        result = ask_model(
            llm=llm,
            question=question,
            options=options,
            system_prompt=system_prompt,
            context=context,
            temperature=0.25 if votes > 1 else 0.0,
        )
        choice = result["answer_choice"]
        candidates.append(choice)
        traces.append(
            {
                "vote_index": vote_idx + 1,
                "model": llm.model_name,
                "choice": choice,
                "step_by_step_thinking": result["step_by_step_thinking"],
                "raw_output": result["raw_output"],
            }
        )

    counts = Counter(candidates)
    best_choice, best_count = counts.most_common(1)[0]

    if tie_break_choice and len(counts) > 1:
        tied = [choice for choice, count in counts.items() if count == best_count]
        if len(tied) > 1 and tie_break_choice in options:
            best_choice = tie_break_choice

    return {
        "final_answer": best_choice,
        "vote_distribution": dict(counts),
        "traces": traces,
    }


def build_pipeline_components(
    graph_path: str,
    config_path: str,
    decomp_model: str,
    ollama_host: str,
    max_new_tokens: int,
    timeout_sec: int,
):
    try:
        from src.compression.extractor import ExtractiveCompressor
        from src.retrieval.graph_retriever import GraphRetriever
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Full pipeline dependencies are missing. Install at least: "
            "numpy, scikit-learn, networkx."
        ) from exc

    config = load_yaml_config(config_path)
    config["compression"]["enabled"] = True
    config["kg"]["graph_path"] = resolve_path(graph_path, PROJECT_ROOT)
    decomp_llm = LocalOllamaLLM(
        model_name=decomp_model,
        host=ollama_host,
        max_new_tokens=max_new_tokens,
        timeout_sec=timeout_sec,
        context_length=4096,
    )
    retriever = GraphRetriever(config["kg"]["graph_path"], config, llm=decomp_llm)
    compressor = ExtractiveCompressor(config)
    return config, retriever, compressor


def load_dataset(name: str, limit: int | None, start_idx: int) -> list[dict[str, Any]]:
    if name == "healthfc":
        return load_healthfc(limit=limit, start_idx=start_idx)
    if name == "medchangeqa":
        return load_medchangeqa(limit=limit, start_idx=start_idx)
    if name == "bioasq":
        return load_bioasq(limit=limit, start_idx=start_idx)
    raise ValueError(f"Unsupported dataset: {name}")


def accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for item in records if item["correct"]) / len(records)


def macro_f1(records: list[dict[str, Any]], labels: list[str]) -> float:
    if not records or not labels:
        return 0.0
    f1_scores = []
    for label in labels:
        tp = sum(1 for item in records if item["pred_answer"] == label and item["gt_answer"] == label)
        fp = sum(1 for item in records if item["pred_answer"] == label and item["gt_answer"] != label)
        fn = sum(1 for item in records if item["pred_answer"] != label and item["gt_answer"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def build_confusion_matrix(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}
    for item in records:
        gt = item.get("gt_answer")
        pred = item.get("pred_answer")
        if gt not in matrix:
            matrix[gt] = {label: 0 for label in labels}
        if pred not in matrix[gt]:
            for row_gt in matrix:
                matrix[row_gt].setdefault(pred, 0)
            if pred not in labels:
                labels = labels + [pred]
        matrix[gt][pred] += 1
    return {"labels": labels, "matrix": matrix}


def format_confusion_matrix(confusion: dict[str, Any], label_names: dict[str, str] | None = None) -> str:
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", {})
    if not labels:
        return "(empty confusion matrix)"

    display_names = {
        label: f"{label}:{label_names.get(label, label)}" if label_names else label
        for label in labels
    }
    row_header = "GT \\ Pred"
    col_width = max(
        len(row_header),
        max(len(display_names[label]) for label in labels),
        max(
            len(str(matrix.get(gt, {}).get(pred, 0)))
            for gt in labels
            for pred in labels
        ),
    )

    header = row_header.ljust(col_width) + " | " + " | ".join(
        display_names[label].rjust(col_width) for label in labels
    )
    divider = "-" * len(header)
    rows = [header, divider]
    for gt in labels:
        row_label = display_names[gt].ljust(col_width)
        counts = " | ".join(
            str(matrix.get(gt, {}).get(pred, 0)).rjust(col_width)
            for pred in labels
        )
        rows.append(f"{row_label} | {counts}")
    return "\n".join(rows)


def extract_atomic_propositions(raw_docs: list[dict[str, Any]]) -> str:
    for doc in raw_docs:
        if doc.get("title") == "Atomic Propositions":
            return doc.get("content", "")
    return ""


def evaluate_system_on_dataset(
    *,
    system_name: str,
    dataset_name: str,
    questions: list[dict[str, Any]],
    models: list[LocalOllamaLLM],
    prompt: str,
    output_dir: Path,
    votes: int,
    tie_break_choice: str | None,
    retriever: Any = None,
    compressor: Any = None,
    context_length: int = 32768,
    top_k: int = 12,
) -> dict[str, Any]:
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_path = dataset_dir / f"{system_name}.json"
    labels = sorted(questions[0]["options"].keys()) if questions else []

    records: list[dict[str, Any]] = []
    log.info("[%s | %s] evaluating %d samples", system_name, dataset_name, len(questions))

    for idx, item in enumerate(questions, start=1):
        started = time.time()
        raw_docs: list[dict[str, Any]] = []
        context = ""
        entities: list[str] = []
        propositions: list[str] = []

        if retriever and compressor:
            retrieval_output = retriever.retrieve(item["question"], k=top_k, options=item["options"])
            if isinstance(retrieval_output, tuple):
                if len(retrieval_output) >= 4:
                    raw_docs, _, entities, propositions = retrieval_output[:4]
                elif len(retrieval_output) >= 2:
                    raw_docs = retrieval_output[0]
            context = compressor.compress(
                query=item["question"],
                docs=raw_docs,
                context_length=context_length,
            )

        vote_result = run_voting(
            models=models,
            question=item["question"],
            options=item["options"],
            system_prompt=prompt,
            context=context,
            votes=votes,
            tie_break_choice=tie_break_choice,
        )

        pred = vote_result["final_answer"]
        gt = item["answer"]
        record = {
            "id": item["id"],
            "question": item["question"],
            "options": item["options"],
            "gt_answer": gt,
            "gt_label": item.get("label", gt),
            "pred_answer": pred,
            "correct": pred == gt,
            "time_seconds": round(time.time() - started, 3),
            "atomic_propositions": propositions or extract_atomic_propositions(raw_docs),
            "linked_entities": entities,
            "retrieved_doc_count": len(raw_docs),
            "retrieved_context": context,
            "vote_distribution": vote_result["vote_distribution"],
            "reasoning_traces": vote_result["traces"],
        }
        records.append(record)
        log.info(
            "[%s | %s] %d/%d pred=%s gt=%s correct=%s",
            system_name,
            dataset_name,
            idx,
            len(questions),
            pred,
            gt,
            record["correct"],
        )

    confusion = build_confusion_matrix(records, labels=labels[:])
    summary = {
        "system_name": system_name,
        "dataset": dataset_name,
        "n": len(records),
        "accuracy": accuracy(records),
        "macro_f1": macro_f1(records, labels=labels),
        "labels": labels,
        "label_names": questions[0]["options"] if questions else {},
        "confusion_matrix": confusion,
        "results": records,
    }
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_overall_summary(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "runs": summaries,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system_name", "dataset", "n", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "system_name": row["system_name"],
                    "dataset": row["dataset"],
                    "n": row["n"],
                    "accuracy": round(row["accuracy"], 4),
                    "macro_f1": round(row["macro_f1"], 4),
                }
            )


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def check_ollama_alive(host: str) -> None:
    req = urlrequest.Request(f"{host.rstrip('/')}/api/tags", method="GET")
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {host}. Start it with `ollama serve` first."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run base models and full KG-RAG ensemble for medical fact-checking."
    )
    parser.add_argument("--datasets", default="healthfc,medchangeqa,bioasq")
    parser.add_argument("--n", type=int, default=1, help="Number of samples per dataset.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start offset inside each dataset.")
    parser.add_argument("--baseline-models", default="qwen2.5:7b,medllama2:7b")
    parser.add_argument("--pipeline-models", default="meditron:latest,qwen2.5:7b,medllama2:7b")
    parser.add_argument("--decomp-model", default="medllama2:7b")
    parser.add_argument("--pipeline-votes", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--graph-path", default=project_path("data", "kg_graph.pkl"))
    parser.add_argument("--config", default=project_path("config", "default.yaml"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--output-dir", default=project_path("results", "ollama_factcheck"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)
    invalid = [name for name in datasets if name not in DATASET_SPECS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Choose from {list(DATASET_SPECS)}")

    baseline_model_names = parse_csv_arg(args.baseline_models)
    pipeline_model_names = parse_csv_arg(args.pipeline_models)
    if not baseline_model_names and not args.skip_baselines:
        raise ValueError("At least one baseline model is required unless --skip-baselines is set.")
    if not pipeline_model_names and not args.skip_pipeline:
        raise ValueError("At least one pipeline model is required unless --skip-pipeline is set.")

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(resolve_path(args.output_dir, PROJECT_ROOT)) / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache = {
        name: load_dataset(name, limit=args.n, start_idx=args.start_idx)
        for name in datasets
    }

    if args.dry_run:
        print(json.dumps(
            {
                "datasets": {name: len(rows) for name, rows in dataset_cache.items()},
                "baseline_models": baseline_model_names,
                "pipeline_models": pipeline_model_names,
                "output_dir": str(output_dir),
                "graph_path": resolve_path(args.graph_path, PROJECT_ROOT),
                "ollama_host": args.ollama_host,
            },
            indent=2,
        ))
        return 0

    check_ollama_alive(args.ollama_host)

    baseline_models = [
        LocalOllamaLLM(
            model_name=name,
            host=args.ollama_host,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
        )
        for name in baseline_model_names
    ]
    pipeline_models = [
        LocalOllamaLLM(
            model_name=name,
            host=args.ollama_host,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
        )
        for name in pipeline_model_names
        
    ]

    summaries: list[dict[str, Any]] = []

    if not args.skip_baselines:
        for llm in baseline_models:
            system_name = f"baseline__{llm.model_name.replace(':', '_')}"
            for dataset_name in datasets:
                questions = dataset_cache[dataset_name]
                spec = DATASET_SPECS[dataset_name]
                summary = evaluate_system_on_dataset(
                    system_name=system_name,
                    dataset_name=dataset_name,
                    questions=questions,
                    models=[llm],
                    prompt=spec.prompt,
                    output_dir=output_dir,
                    votes=1,
                    tie_break_choice=spec.tie_break_choice,
                    retriever=None,
                    compressor=None,
                    top_k=args.top_k,
                )
                summaries.append(summary)

    if not args.skip_pipeline:
        graph_path = resolve_path(args.graph_path, PROJECT_ROOT)
        if not Path(graph_path).exists():
            raise FileNotFoundError(f"KG graph file not found: {graph_path}")

        config, retriever, compressor = build_pipeline_components(
            graph_path=graph_path,
            config_path=args.config,
            decomp_model=args.decomp_model,
            ollama_host=args.ollama_host,
            max_new_tokens=args.max_new_tokens,
            timeout_sec=args.timeout_sec,
        )
        context_length = int(config.get("model", {}).get("context_length", 32768))

        for dataset_name in datasets:
            questions = dataset_cache[dataset_name]
            spec = DATASET_SPECS[dataset_name]
            summary = evaluate_system_on_dataset(
                system_name="pipeline__atomic_kgrag_voting",
                dataset_name=dataset_name,
                questions=questions,
                models=pipeline_models,
                prompt=spec.prompt,
                output_dir=output_dir,
                votes=args.pipeline_votes,
                tie_break_choice=spec.tie_break_choice,
                retriever=retriever,
                compressor=compressor,
                context_length=context_length,
                top_k=args.top_k,
            )
            summaries.append(summary)

    write_overall_summary(summaries, output_dir)

    print("\nSaved results to:", output_dir)
    for row in summaries:
        print(
            f"{row['system_name']:>32} | {row['dataset']:<12} | "
            f"n={row['n']:<3} | acc={row['accuracy']:.3f} | macro_f1={row['macro_f1']:.3f}"
        )
        print(format_confusion_matrix(row["confusion_matrix"], row.get("label_names")))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
