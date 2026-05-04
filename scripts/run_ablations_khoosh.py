"""
Khoosh Ablation Runner
======================
Runs:
  1. Base-model baselines for:
     - medllama2:7b
     - meditron:latest
     - qwen2.5:7b
  2. RAG pipeline with atomic decomposition + KG + ensemble
  3. RAG pipeline with atomic decomposition + KG + live web search + ensemble

Datasets:
  - healthfc
  - bioasq (yes/no only; handled inside eval_all.py)

Examples:
  python scripts/run_ablations_khoosh.py --dry-run
  python scripts/run_ablations_khoosh.py --dataset-percent 10
  python scripts/run_ablations_khoosh.py --skip-base-models --trace-rag-pipelines
  python scripts/run_ablations_khoosh.py --skip-base-models --trace-rag-pipelines --dataset-percent 1
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "eval_all.py")
BASE_INFER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_base_models_khoosh.py")
RESULTS_SUBDIR = "khoosh_ablations"
DEFAULT_DATASETS = ["healthfc", "bioasq"]
ENSEMBLE_MODELS = ["medllama2:7b", "meditron:latest", "qwen2.5:7b"]


@dataclass
class Experiment:
    tag: str
    experiment_id: str
    description: str
    models: List[str]
    votes: int
    use_graph: bool = False
    use_pubmed: bool = False
    enable_live_search: bool = False
    runner_script: str = EVAL_SCRIPT
    extra_flags: List[str] = field(default_factory=list)


BASE_EXPERIMENTS = [
    Experiment(
        tag="khoosh_base_medllama2_7b",
        experiment_id="K1",
        description="Base model baseline: medllama2:7b (direct inference, no retrieval, no KG, no atomic decomp, 1 vote)",
        models=["medllama2:7b"],
        votes=1,
        use_graph=False,
        use_pubmed=False,
        runner_script=BASE_INFER_SCRIPT,
    ),
    Experiment(
        tag="khoosh_base_meditron_latest",
        experiment_id="K2",
        description="Base model baseline: meditron:latest (direct inference, no retrieval, no KG, no atomic decomp, 1 vote)",
        models=["meditron:latest"],
        votes=1,
        use_graph=False,
        use_pubmed=False,
        runner_script=BASE_INFER_SCRIPT,
    ),
    Experiment(
        tag="khoosh_base_qwen25_7b",
        experiment_id="K3",
        description="Base model baseline: qwen2.5:7b (direct inference, no retrieval, no KG, no atomic decomp, 1 vote)",
        models=["qwen2.5:7b"],
        votes=1,
        use_graph=False,
        use_pubmed=False,
        runner_script=BASE_INFER_SCRIPT,
    ),
]


RAG_KG_ENSEMBLE_EXPERIMENT = Experiment(
    tag="khoosh_rag_kg_med_ensemble",
    experiment_id="K4",
    description="RAG with atomic decomp + KG + ensemble (3 votes round-robin)",
    models=ENSEMBLE_MODELS,
    votes=3,
    use_graph=True,
    use_pubmed=True,
)

RAG_KG_LIVE_WEB_ENSEMBLE_EXPERIMENT = Experiment(
    tag="khoosh_rag_kg_live_web_med_ensemble",
    experiment_id="K5",
    description="RAG with atomic decomp + KG + live web search + ensemble (3 votes round-robin)",
    models=ENSEMBLE_MODELS,
    votes=3,
    use_graph=True,
    use_pubmed=True,
    enable_live_search=True,
)


def build_cmd(exp: Experiment,
              dataset: str,
              results_dir: str,
              dataset_percent: float | None,
              trace_pipeline: bool) -> str:
    parts = [
        f"python3 {exp.runner_script}",
        f"--dataset {dataset}",
        f"--tag {exp.tag}",
        f"--votes {exp.votes}",
        f"--experiment-id {exp.experiment_id}",
        f"--results-dir {results_dir}",
    ]

    if len(exp.models) == 1:
        parts.append(f"--model {exp.models[0]}")
    else:
        parts.append(f"--models '{','.join(exp.models)}'")

    if exp.runner_script == EVAL_SCRIPT:
        if exp.use_graph:
            parts.append("--use-graph")
        else:
            parts.append("--no-kg")

        # if exp.use_pubmed:
            # parts.append("--use-pubmed")
        # else:
        parts.append("--no-pubmed")

        if exp.enable_live_search:
            parts.append("--enable-live-search")

    if dataset_percent is not None:
        parts.append(f"--percent {dataset_percent}")

    if trace_pipeline:
        parts.append("--trace-pipeline")

    parts.extend(exp.extra_flags)
    return " ".join(parts)


def run_experiment(exp: Experiment,
                   dataset: str,
                   results_dir: str,
                   dataset_percent: float | None,
                   trace_pipeline: bool,
                   dry_run: bool) -> int:
    cmd = build_cmd(
        exp=exp,
        dataset=dataset,
        results_dir=results_dir,
        dataset_percent=dataset_percent,
        trace_pipeline=trace_pipeline,
    )
    env = os.environ.copy()
    env["CLINPROOF_RESULTS"] = results_dir

    print(f"\n{'-' * 70}")
    print(f"[{exp.experiment_id}] {exp.description}")
    print(f"Dataset : {dataset}")
    print(f"Command : {cmd}")
    print(f"{'-' * 70}")

    if dry_run:
        return 0

    ret = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )
    return ret.returncode


def run_stage(stage_name: str,
              experiments: List[Experiment],
              datasets: List[str],
              results_dir: str,
              dataset_percent: float | None,
              trace_pipeline: bool,
              dry_run: bool) -> None:
    print(f"\n{'=' * 70}")
    print(f"{stage_name}")
    print(f"{'=' * 70}")

    for exp in experiments:
        for dataset in datasets:
            code = run_experiment(
                exp=exp,
                dataset=dataset,
                results_dir=results_dir,
                dataset_percent=dataset_percent,
                trace_pipeline=trace_pipeline,
                dry_run=dry_run,
            )
            if code != 0:
                print(
                    f"[ERROR] {exp.experiment_id} failed on {dataset} "
                    f"with exit code {code}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Khoosh ablations on HealthFC and BioASQ"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=["healthfc", "bioasq"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--dataset-percent",
        type=float,
        default=None,
        help="Evaluate only the first N percent of each dataset (0 < N <= 100)",
    )
    parser.add_argument(
        "--skip-base-models",
        action="store_true",
        help="Skip the single-model baseline stage",
    )
    parser.add_argument(
        "--skip-full-pipeline",
        action="store_true",
        help="Skip the KG + ensemble RAG stage",
    )
    parser.add_argument(
        "--skip-live-web-pipeline",
        action="store_true",
        help="Skip the KG + live web search + ensemble RAG stage",
    )
    parser.add_argument(
        "--trace-rag-pipelines",
        action="store_true",
        help="Show question, atomic decomp, retrieval, voting, and final output during the RAG stages",
    )
    parser.add_argument(
        "--trace-full-pipeline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Root directory for result JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    if args.dataset_percent is not None and not (0 < args.dataset_percent <= 100):
        parser.error("--dataset-percent must be in the range (0, 100].")

    if args.skip_base_models and args.skip_full_pipeline and args.skip_live_web_pipeline:
        parser.error("All stages are skipped. Enable at least one stage to run.")

    trace_rag_pipelines = args.trace_rag_pipelines or args.trace_full_pipeline

    results_dir = os.path.join(args.results_root, RESULTS_SUBDIR)
    if not args.dry_run:
        os.makedirs(results_dir, exist_ok=True)

    if not args.skip_base_models:
        run_stage(
            stage_name="Stage 1: Base Models",
            experiments=BASE_EXPERIMENTS,
            datasets=args.datasets,
            results_dir=results_dir,
            dataset_percent=args.dataset_percent,
            trace_pipeline=False,
            dry_run=args.dry_run,
        )

    if not args.skip_full_pipeline:
        run_stage(
            stage_name="Stage 2: RAG + Atomic Decomp + KG + Ensemble",
            experiments=[RAG_KG_ENSEMBLE_EXPERIMENT],
            datasets=args.datasets,
            results_dir=results_dir,
            dataset_percent=args.dataset_percent,
            trace_pipeline=trace_rag_pipelines,
            dry_run=args.dry_run,
        )

    if not args.skip_live_web_pipeline:
        run_stage(
            stage_name="Stage 3: RAG + Atomic Decomp + KG + Live Web Search + Ensemble",
            experiments=[RAG_KG_LIVE_WEB_ENSEMBLE_EXPERIMENT],
            datasets=args.datasets,
            results_dir=results_dir,
            dataset_percent=args.dataset_percent,
            trace_pipeline=trace_rag_pipelines,
            dry_run=args.dry_run,
        )

    sys.exit(0)
