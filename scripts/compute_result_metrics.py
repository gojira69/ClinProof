#!/usr/bin/env python3
"""
Compute summary classification metrics for a ClinProof results JSON.

Usage:
  python scripts/compute_result_metrics.py path/to/results.json
  python scripts/compute_result_metrics.py path/to/results.json --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = str(answer).strip().upper()
    match = re.search(r"\b([ABCDE])\b", answer)
    if match:
        return match.group(1)
    if "YES" in answer:
        return "A"
    if "NO" in answer:
        return "B"
    return answer[0] if answer else ""


def compute_classification_metrics(
    results: list[dict],
    pred_field: str = "pred_answer",
    gt_field: str = "gt_answer",
) -> dict:
    pairs = []
    for result in results:
        gt = normalize_answer(result.get(gt_field, ""))
        pred = normalize_answer(result.get(pred_field, ""))
        if gt and pred:
            pairs.append((gt, pred))

    labels = sorted({gt for gt, _ in pairs})
    per_class = {}

    for label in labels:
        tp = sum(1 for gt, pred in pairs if gt == label and pred == label)
        fp = sum(1 for gt, pred in pairs if gt != label and pred == label)
        fn = sum(1 for gt, pred in pairs if gt == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
        }

    total = len(pairs)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    macro_precision = (
        sum(metric["precision"] for metric in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    macro_recall = (
        sum(metric["recall"] for metric in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    macro_f1 = (
        sum(metric["f1"] for metric in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )

    return {
        "accuracy": correct / total if total else 0.0,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


def compute_confusion_matrix(
    results: list[dict],
    pred_field: str = "pred_answer",
    gt_field: str = "gt_answer",
) -> dict:
    pairs = []
    for result in results:
        gt = normalize_answer(result.get(gt_field, ""))
        pred = normalize_answer(result.get(pred_field, ""))
        if gt and pred:
            pairs.append((gt, pred))

    labels = sorted({gt for gt, _ in pairs} | {pred for _, pred in pairs})
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}

    for gt, pred in pairs:
        matrix[gt][pred] += 1

    return {"labels": labels, "matrix": matrix}


def build_model_results(results: list[dict]) -> dict[str, list[dict]]:
    per_model: dict[str, list[dict]] = {}

    for result in results:
        gt_answer = result.get("gt_answer", "")
        for trace in result.get("reasoning_traces", []):
            model_name = trace.get("model")
            model_choice = trace.get("choice", "")
            if not model_name:
                continue
            per_model.setdefault(model_name, []).append(
                {
                    "gt_answer": gt_answer,
                    "pred_answer": model_choice,
                }
            )

    return per_model


def compute_model_metrics(results: list[dict]) -> dict[str, dict]:
    model_results = build_model_results(results)
    output: dict[str, dict] = {}

    for model_name, rows in sorted(model_results.items()):
        classification = compute_classification_metrics(rows)
        confusion = compute_confusion_matrix(rows)
        output[model_name] = {
            "n_results": len(rows),
            "accuracy": classification.get("accuracy", 0.0),
            "precision": classification.get("precision", 0.0),
            "recall": classification.get("recall", 0.0),
            "macro_f1": classification.get("f1", 0.0),
            "correct": classification.get("correct", 0),
            "total": classification.get("total", len(rows)),
            "unanimity": None,
            "confusion_matrix": confusion,
        }

    return output


def compute_unanimity(results: list[dict]) -> dict:
    unanimous = 0
    evaluated = 0

    for result in results:
        vote_distribution = result.get("vote_distribution") or {}
        total_votes = sum(vote_distribution.values())
        if total_votes <= 0:
            continue

        evaluated += 1
        if max(vote_distribution.values()) == total_votes:
            unanimous += 1

    return {
        "unanimity": (unanimous / evaluated) if evaluated else 0.0,
        "unanimous_count": unanimous,
        "evaluated_count": evaluated,
    }


def load_payload(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path, help="Path to results JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of a text summary",
    )
    args = parser.parse_args()

    payload = load_payload(args.results_json)
    results = payload.get("results", [])

    classification = payload.get("classification_metrics")
    if not classification:
        classification = compute_classification_metrics(results)
    confusion = compute_confusion_matrix(results)
    per_model = compute_model_metrics(results)

    unanimity = compute_unanimity(results)
    output = {
        "file": str(args.results_json),
        "n_results": len(results),
        "accuracy": classification.get("accuracy", 0.0),
        "precision": classification.get("precision", 0.0),
        "recall": classification.get("recall", 0.0),
        "macro_f1": classification.get("f1", 0.0),
        "unanimity": unanimity["unanimity"],
        "unanimous_count": unanimity["unanimous_count"],
        "evaluated_count": unanimity["evaluated_count"],
        "correct": classification.get("correct", 0),
        "total": classification.get("total", len(results)),
        "confusion_matrix": confusion,
        "per_model_metrics": per_model,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"File:        {output['file']}")
    print(f"Results:     {output['n_results']}")
    print(f"Accuracy:    {output['accuracy']:.6f} ({output['correct']}/{output['total']})")
    print(f"Precision:   {output['precision']:.6f}")
    print(f"Recall:      {output['recall']:.6f}")
    print(f"Macro-F1:    {output['macro_f1']:.6f}")
    print(
        f"Unanimity:   {output['unanimity']:.6f} "
        f"({output['unanimous_count']}/{output['evaluated_count']})"
    )
    print("\nConfusion Matrix (rows=ground truth, cols=prediction):")
    labels = output["confusion_matrix"]["labels"]
    matrix = output["confusion_matrix"]["matrix"]
    header = "GT\\Pred".ljust(10) + "".join(label.rjust(8) for label in labels)
    print(header)
    for gt in labels:
        row = gt.ljust(10) + "".join(str(matrix[gt][pred]).rjust(8) for pred in labels)
        print(row)

    if output["per_model_metrics"]:
        print("\nPer-model metrics:")
        for model_name, metrics in output["per_model_metrics"].items():
            print(f"\n[{model_name}]")
            print(
                f"Accuracy:    {metrics['accuracy']:.6f} "
                f"({metrics['correct']}/{metrics['total']})"
            )
            print(f"Precision:   {metrics['precision']:.6f}")
            print(f"Recall:      {metrics['recall']:.6f}")
            print(f"Macro-F1:    {metrics['macro_f1']:.6f}")
            print("Unanimity:   N/A (single-model vote per question)")
            model_labels = metrics["confusion_matrix"]["labels"]
            model_matrix = metrics["confusion_matrix"]["matrix"]
            header = "GT\\Pred".ljust(10) + "".join(label.rjust(8) for label in model_labels)
            print("Confusion Matrix (rows=ground truth, cols=prediction):")
            print(header)
            for gt in model_labels:
                row = gt.ljust(10) + "".join(
                    str(model_matrix[gt][pred]).rjust(8) for pred in model_labels
                )
                print(row)


if __name__ == "__main__":
    main()
