"""
ClinProof Ablation Experiment Orchestrator
==========================================
Defines all ablation configurations (EXP A–D) and generates or executes
the corresponding eval_all.py commands.

Usage:
    # Print all commands (review before running)
    python scripts/run_ablations.py --dry-run

    # Run a specific experiment group
    python scripts/run_ablations.py --group A --datasets medchangeqa bioasq

    # Run all experiments sequentially
    python scripts/run_ablations.py --group all --datasets medchangeqa bioasq

Notes:
  - Each experiment saves results to results/v5_ablations/<tag>_<dataset>.json
  - Use --dry-run to print commands without executing
  - Use --parallel N to launch N experiments in parallel (background processes)
"""
import argparse
import os
import subprocess
import sys
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/mnt/d/Harsha/AoLM/ClinProof"
EVAL_SCRIPT = f"{PROJECT_ROOT}/eval_all.py"
RESULTS_SUBDIR = "v5_ablations"

# ── Experiment Definition ─────────────────────────────────────────────────────


@dataclass
class Experiment:
    tag:           str
    experiment_id: str
    group:         str            # A / B / C / D
    description:   str
    # Model(s)
    models:        List[str]      # empty = use default (qwen2.5:14b)
    votes:         int = 3
    # Retrieval
    no_kg:         bool = False   # disable GraphRAG
    no_bm25:       bool = False   # disable BM25
    use_pubmed:    bool = False   # enable MedCPT dense retrieval
    # Recency
    recency_bm25:  bool = False
    recency_alpha: float = 0.3
    # Decomposition
    no_decomp:     bool = False
    # MedChangeQA split (all=512, test=103, val=102, train=307)
    medchangeqa_split: str = "all"
    # Extra datasets
    datasets:      List[str] = field(default_factory=lambda: [
        "bioasq", "healthfc", "scifact"])
    enable_live_search: bool = False
    extra_flags:   List[str] = field(default_factory=list)


EXPERIMENTS: List[Experiment] = [
    # ==========================================================================
    # EXP BEST: Optimal Configurations (Homo vs Hetero)
    # ==========================================================================
    Experiment(
        tag="BEST1_homo_dense_bm25",
        experiment_id="BEST1",
        group="BEST",
        description="Best Homogeneous: Qwen2.5 (3-vote), Dense+BM25, Decomp",
        models=["qwen2.5:14b"],
        votes=3,
        no_kg=True,
        no_bm25=False,
        use_pubmed=True,
        no_decomp=False,
        medchangeqa_split="all",
        datasets=["bioasq", "healthfc", "medchangeqa"],
    ),
    Experiment(
        tag="BEST2_hetero_dense_bm25",
        experiment_id="BEST2",
        group="BEST",
        description="Best Heterogeneous: Qwen+Llama+Meditron (3-vote), Dense+BM25, Decomp",
        models=["qwen2.5:14b", "meditron:7b", "llama3.1:8b"],
        votes=3,
        no_kg=True,
        no_bm25=False,
        use_pubmed=True,
        no_decomp=False,
        medchangeqa_split="all",
        datasets=["bioasq", "healthfc", "medchangeqa"],
    ),
    # ==========================================================================
    # EXP LIVE: Internet Search Ablations
    # ==========================================================================
    Experiment(
        tag="LIVE1_base_homo",
        experiment_id="LIVE1",
        group="LIVE",
        description="Base Model + Live Web Search (No RAG, No Decomp, No Compression)",
        models=["qwen2.5:14b"],
        votes=1,
        no_kg=True,
        no_bm25=True,
        use_pubmed=False,
        enable_live_search=True,
        no_decomp=True,
        medchangeqa_split="all",
        datasets=["healthfc"],
        extra_flags=["--no-compression"],
    ),
    Experiment(
        tag="LIVE2_live_only_rag",
        experiment_id="LIVE2",
        group="LIVE",
        description="Full Pipeline with Live Web Search Only (Decomp, Compression, Voting)",
        models=["qwen2.5:14b"],
        votes=3,
        no_kg=True,
        no_bm25=True,
        use_pubmed=False,
        enable_live_search=True,
        no_decomp=False,
        medchangeqa_split="all",
        datasets=["healthfc"],
        extra_flags=["--live-search-k", "15"],
    ),
    # ==========================================================================
    # EXP ZERO: Pure Zero-Shot Baseline
    # No RAG, no context, no atomic decomp, no compression.
    # Answers the question: What is the base model performance without ClinProof?
    # ==========================================================================
    Experiment(
        tag="Z1_zeroshot_baseline",
        experiment_id="Z1",
        group="Z",
        description="Pure zero-shot baseline (no-RAG, no-decomp, empty-evidence)",
        models=["qwen2.5:14b"],
        votes=1,
        no_kg=True,
        no_bm25=True,
        no_decomp=True,
        datasets=["healthfc", "bioasq"],
        extra_flags=["--empty-evidence", "--no-compression"],
    ),
    # ==========================================================================
    # EXP A: Medical Fine-tuned vs Base Models
    # All use BM25 only (no KG) with 1 vote for speed.
    # Answers the question: Does medical fine-tuning help?
    # ==========================================================================
    # Experiment(
    #     tag="A1_meditron7b",
    #     experiment_id="A1",
    #     group="A",
    #     description="Medical fine-tuned: meditron:7b (BM25, no-KG, 1 vote)",
    #     models=["meditron:7b"],
    #     votes=1, no_kg=True,
    # ),
    # Experiment(
    #     tag="A2_medllama7b",
    #     experiment_id="A2",
    #     group="A",
    #     description="Medical fine-tuned: medllama2:7b (BM25, no-KG, 1 vote)",
    #     models=["medllama2:7b"],
    #     votes=1, no_kg=True,
    # ),
    Experiment(
        tag="A3_biomistral",
        experiment_id="A3",
        group="A",
        description="Medical fine-tuned: cniongolo/biomistral (BM25, no-KG, 1 vote)",
        models=["cniongolo/biomistral:latest"],
        votes=1, no_kg=True,
    ),
    # Experiment(
    #     tag="A4_llama31_8b",
    #     experiment_id="A4",
    #     group="A",
    #     description="Base model: llama3.1:8b (BM25, no-KG, 1 vote)",
    #     models=["llama3.1:8b"],
    #     votes=1, no_kg=True,
    # ),
    Experiment(
        tag="A5_mistral7b",
        experiment_id="A5",
        group="A",
        description="Base model: mistral:7b (BM25, no-KG, 1 vote) — v1 baseline",
        models=["mistral:7b"],
        votes=1, no_kg=True,
    ),
    Experiment(
        tag="A6_qwen14b_nokgbm25",
        experiment_id="A6",
        group="A",
        description="Large base: qwen2.5:14b (BM25, no-KG, 3 votes) — best model, no KG",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
    ),

    # ==========================================================================
    # EXP B: Is the KG Helping? Is BM25 Necessary?
    # All use qwen2.5:14b, 3 votes.
    # Answers the user's specific slide questions.
    # ==========================================================================
    Experiment(
        tag="B1_dense_only",
        experiment_id="B1",
        group="B",
        description="Dense RAG Only (MedCPT, no KG, no BM25)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
    ),
    Experiment(
        tag="B2_dense_kg",
        experiment_id="B2",
        group="B",
        description="RAG + KG (MedCPT + GraphRAG, no BM25)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=False, no_bm25=True, use_pubmed=True,
    ),
    Experiment(
        tag="B3_dense_bm25",
        experiment_id="B3",
        group="B",
        description="RAG + BM25 (MedCPT + BM25, no KG)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=False, use_pubmed=True,
    ),
    Experiment(
        tag="B4_full_pipeline",
        experiment_id="B4",
        group="B",
        description="Full Pipeline (MedCPT + KG + BM25)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=False, no_bm25=False, use_pubmed=True,
    ),

    # ==========================================================================
    # EXP C: Recency-Weighted BM25 (MedChangeQA ONLY)
    # All use qwen2.5:14b, 3 votes, no KG.
    # Answers the question: Does recency weighting help on temporal evaluation?
    # ==========================================================================
    Experiment(
        tag="C1_bm25_flat",
        experiment_id="C1",
        group="C",
        description="BM25 (no recency weighting) — flat baseline",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="C2_bm25_recency_a0.3",
        experiment_id="C2",
        group="C",
        description="BM25 + recency alpha=0.3 (mild boost for newer docs)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
        recency_bm25=True, recency_alpha=0.3,
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="C3_bm25_recency_a0.7",
        experiment_id="C3",
        group="C",
        description="BM25 + recency alpha=0.7 (strong boost for newer docs)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
        recency_bm25=True, recency_alpha=0.7,
        datasets=["medchangeqa"],
    ),

    # ==========================================================================
    # EXP D: Ensemble Strategies
    # Answers: Does more voting help? Does mixing medical+general models help?
    # ==========================================================================
    Experiment(
        tag="D1_qwen14b_1vote",
        experiment_id="D1",
        group="D",
        description="Single vote: qwen2.5:14b (no-KG, BM25)",
        models=["qwen2.5:14b"],
        votes=1, no_kg=True,
    ),
    Experiment(
        tag="D2_qwen14b_3vote",
        experiment_id="D2",
        group="D",
        description="3 votes: qwen2.5:14b (no-KG, BM25)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
    ),
    # Experiment(
    #     tag="D3_qwen14b_5vote",
    #     experiment_id="D3",
    #     group="D",
    #     description="5 votes: qwen2.5:14b (no-KG, BM25)",
    #     models=["qwen2.5:14b"],
    #     votes=5, no_kg=True,
    # ),
    # Experiment(
    #     tag="D4_medensemble_3",
    #     experiment_id="D4",
    #     group="D",
    #     description="Medical ensemble (meditron+medllama2+biomistral), 3 votes",
    #     models=["meditron:7b", "medllama2:7b", "cniongolo/biomistral:latest"],
    #     votes=3, no_kg=True,
    # ),
    Experiment(
        tag="D5_hybridensemble_3",
        experiment_id="D5",
        group="D",
        description="Hybrid ensemble (qwen2.5:14b+meditron:7b+llama3.1:8b), 3 votes",
        models=["qwen2.5:14b", "meditron:7b", "llama3.1:8b"],
        votes=3, no_kg=True,
    ),

    # ==========================================================================
    # EXP E: Atomic Decomposition Impact (HealthFC)
    # Answers: Does breaking claims into atomic propositions help or hurt?
    # ==========================================================================
    Experiment(
        tag="E1_qwen14b_3vote_with_decomp",
        experiment_id="E1",
        group="E",
        description="Best Combo: qwen2.5:14b, 3 votes, BM25-only, WITH atomic decomp",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_decomp=False,
        datasets=["healthfc"],
    ),
    Experiment(
        tag="E2_qwen14b_3vote_no_decomp",
        experiment_id="E2",
        group="E",
        description="Best Combo: qwen2.5:14b, 3 votes, BM25-only, NO atomic decomp",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_decomp=True,
        datasets=["healthfc"],
    ),
    Experiment(
        tag="BEST1_no_decomp",
        experiment_id="E3",
        group="E",
        description="Best Config (Dense+BM25): Qwen2.5, 3-vote, NO atomic decomp",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, use_pubmed=True, no_decomp=True,
        datasets=["healthfc"],
    ),

    # ==========================================================================
    # EXP F: Best Ensemble + Recency (MedChangeQA)
    # Answers: Does combining the strongest ensemble with recency help?
    # ==========================================================================
    # Experiment(
    #     tag="F1_hybridensemble_recency0.3",
    #     experiment_id="F1",
    #     group="F",
    #     description="Hybrid Ensemble (qwen+meditron+llama) + 0.3 Recency + Atomic Decomp",
    #     models=["qwen2.5:14b", "meditron:7b", "llama3.1:8b"],
    #     votes=3, no_kg=True, no_decomp=False,
    #     recency_bm25=True, recency_alpha=0.3,
    #     datasets=["medchangeqa"],
    # ),
    Experiment(
        tag="F2_qwen14b_recency0.3_with_decomp",
        experiment_id="F2",
        group="F",
        description="Single Best: qwen2.5:14b + 3 votes + 0.3 Recency + Atomic Decomp",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_decomp=False,
        recency_bm25=True, recency_alpha=0.3,
        datasets=["medchangeqa"],
    ),

    # ==========================================================================
    # EXP G1: PubMed Recency Re-Ranking (MedChangeQA ONLY)
    # -----------------------------------------------------------------------
    # APPROACH: Use PubMed dense retrieval ONLY (no BM25 textbooks, no KG).
    #   PubMed articles have chronologically ordered PMIDs (~30k–33.3M, 1966–2020).
    #   After FAISS retrieves 3×k candidates, re-rank by:
    #       final_score = cosine_sim + alpha × pmid_recency_weight
    #   where pmid_recency_weight ∈ [0,1] maps PMID to recency.
    #
    # Hypothesis: mild alpha (0.1–0.3) improves REFUTED recall by preferring
    #   newer PubMed articles that reflect current medical consensus.
    # ==========================================================================
    Experiment(
        tag="G1a_pubmed_recency_a0.1",
        experiment_id="G1a",
        group="G1",
        description="PubMed-only + recency alpha=0.1 (very mild PMID boost)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=True, recency_alpha=0.1,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G1b_pubmed_recency_a0.2",
        experiment_id="G1b",
        group="G1",
        description="PubMed-only + recency alpha=0.2 (mild)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=True, recency_alpha=0.2,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G1c_pubmed_recency_a0.3",
        experiment_id="G1c",
        group="G1",
        description="PubMed-only + recency alpha=0.3 (moderate)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=True, recency_alpha=0.3,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G1d_pubmed_recency_a0.5",
        experiment_id="G1d",
        group="G1",
        description="PubMed-only + recency alpha=0.5 (strong)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=True, recency_alpha=0.5,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G1e_pubmed_recency_a0.7",
        experiment_id="G1e",
        group="G1",
        description="PubMed-only + recency alpha=0.7 (very strong)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=True, recency_alpha=0.7,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G1f_pubmed_flat",
        experiment_id="G1f",
        group="G1",
        description="PubMed-only BASELINE (no recency, alpha=0.0)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True, no_bm25=True, use_pubmed=True,
        recency_bm25=False, recency_alpha=0.0,
        medchangeqa_split="test",
        datasets=["medchangeqa"],
    ),

    # ==========================================================================
    # EXP G2: NEI Calibration (MedChangeQA ONLY)
    # -----------------------------------------------------------------------
    # Problem: NEI class has 0% recall across ALL MedChangeQA runs.
    # The model never predicts "NOT ENOUGH INFORMATION" despite it being 31%
    # of the gold labels. This is the single biggest bottleneck on this dataset.
    #
    # Fix: prompt-level forcing via --nei-threshold flag in eval_all.py.
    # G2a: explicit prompt instruction to use NEI when evidence is contradictory.
    # G2b: confidence-based abstention — if max_vote_frac < 0.67 → predict NEI.
    # ==========================================================================
    Experiment(
        tag="G2a_nei_prompt_forced",
        experiment_id="G2a",
        group="G2",
        description="NEI calibration: explicit prompt instruction to use NEI on insufficient evidence",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
        extra_flags=["--nei-forced"],
        datasets=["medchangeqa"],
    ),
    Experiment(
        tag="G2b_nei_confidence_threshold",
        experiment_id="G2b",
        group="G2",
        description="NEI calibration: predict NEI when ensemble confidence < 0.67 (split vote)",
        models=["qwen2.5:14b"],
        votes=3, no_kg=True,
        extra_flags=["--nei-threshold", "0.67"],
        datasets=["medchangeqa"],
    ),
    # ==========================================================================
    # EXP S: Context Compression Ablations (Section 7.4)
    # Answers: Does MMR compression improve reasoning over raw documents?
    # ==========================================================================
    Experiment(
        tag="S2_comp_on",
        experiment_id="S2",
        group="S",
        description="Compression ON (MMR): Qwen2.5, 1-vote, BM25-only (Fast)",
        models=["qwen2.5:14b"],
        votes=1, no_kg=True, use_pubmed=False,
        datasets=["healthfc"],
    ),
    Experiment(
        tag="S3_comp_off",
        experiment_id="S3",
        group="S",
        description="Compression OFF (Raw Docs): Qwen2.5, 1-vote, BM25-only (Fast)",
        models=["qwen2.5:14b"],
        votes=1, no_kg=True, use_pubmed=False,
        extra_flags=["--no-compression"],
        datasets=["healthfc"],
    ),
]

# ── Command Builder ───────────────────────────────────────────────────────────


def build_cmd(exp: Experiment, dataset: str, results_dir: str) -> str:
    """Build a shell command string for one (experiment, dataset) pair."""
    tag = f"{exp.tag}"
    cmd_parts = [
        f"conda run -n aolm_project python3 {EVAL_SCRIPT}",
        f"--dataset {dataset}",
        f"--tag {tag}",
        f"--votes {exp.votes}",
        f"--experiment-id {exp.experiment_id}",
        f"--results-dir {results_dir}",
        # --use-pubmed / --no-pubmed is added conditionally below based on exp.use_pubmed
    ]

    # Models: single or ensemble
    if len(exp.models) == 1:
        cmd_parts.append(f"--model {exp.models[0]}")
    else:
        models_csv = ",".join(exp.models)
        cmd_parts.append(f"--models '{models_csv}'")

    # Retrieval flags
    if exp.no_kg:
        cmd_parts.append("--no-kg")
    else:
        cmd_parts.append("--use-graph")

    if exp.no_bm25:
        cmd_parts.append("--no-bm25")

    if exp.use_pubmed:
        cmd_parts.append("--use-pubmed")
    else:
        cmd_parts.append("--no-pubmed")

    # Recency
    if exp.recency_bm25:
        cmd_parts.append("--recency-bm25")
        cmd_parts.append(f"--recency-alpha {exp.recency_alpha}")

    # Extra flags
    if exp.no_decomp:
        cmd_parts.append("--no-decomp")

    # MedChangeQA split
    if exp.medchangeqa_split != "all" and dataset == "medchangeqa":
        cmd_parts.append(f"--medchangeqa-split {exp.medchangeqa_split}")

    # Live search flag
    if exp.enable_live_search:
        cmd_parts.append("--enable-live-search")

    if exp.extra_flags:
        cmd_parts.extend(exp.extra_flags)

    return " ".join(cmd_parts)


# ── Runner ───────────────────────────────────────────────────────────────────


def run_experiment(exp: Experiment, dataset: str, results_dir: str,
                   dry_run: bool = False) -> None:
    cmd = build_cmd(exp, dataset, results_dir)
    env = os.environ.copy()
    env["CLINPROOF_RESULTS"] = results_dir

    print(f"\n{'-'*70}")
    print(f"  [{exp.experiment_id}] {exp.description}")
    print(f"  Dataset : {dataset}")
    print(f"  CMD     : {cmd}")
    print(f"{'-'*70}")

    if not dry_run:
        full_cmd = f"cd {PROJECT_ROOT} && CLINPROOF_RESULTS={results_dir} {cmd}"
        ret = subprocess.run(
            ["wsl", "-e", "bash", "-ic", full_cmd],
            text=True,
            env=env,
        )
        if ret.returncode != 0:
            print(f"  [ERROR] Experiment {exp.experiment_id} on {dataset} failed "
                  f"(exit {ret.returncode})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClinProof Ablation Experiment Runner"
    )
    parser.add_argument("--group",    default="A",
                        help="Experiment group(s) to run: A, B, C, D, E, F, G1, G2, or 'all' (default: A)")
    parser.add_argument("--datasets", nargs="+",
                        default=["bioasq", "healthfc", "scifact"],
                        choices=["medchangeqa", "bioasq",
                                 "healthfc", "scifact", "medqa"],
                        help="Datasets to evaluate (default: medchangeqa bioasq)")
    parser.add_argument("--exp-ids",  nargs="+", default=None,
                        help="Run only specific experiment IDs (e.g. A1 A4 B2)")
    parser.add_argument("--results-root", default=f"{PROJECT_ROOT}/results",
                        help="Root results directory")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of experiments to run in parallel")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--list",     action="store_true",
                        help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'ID':<5} {'Group':<6} {'Tag':<35} Description")
        print("-" * 100)
        for exp in EXPERIMENTS:
            print(
                f"{exp.experiment_id:<5} {exp.group:<6} {exp.tag:<35} {exp.description}")
        sys.exit(0)

    results_dir = os.path.join(args.results_root, RESULTS_SUBDIR)
    results_dir_wsl = results_dir.replace("\\", "/")
    os.makedirs(results_dir, exist_ok=True) if not args.dry_run else None

    # Filter experiments
    groups = [g.strip().upper() for g in args.group.split(
        ",")] if args.group.lower() != "all" else ["BEST", "LIVE", "Z", "A", "B", "C", "D", "E", "F", "G1", "G2", "S"]
    selected = [
        exp for exp in EXPERIMENTS
        if exp.group in groups
        and (args.exp_ids is None or exp.experiment_id in args.exp_ids)
    ]

    print(f"\nClinProof Ablation Runner")
    print(f"  Groups   : {groups}")
    print(f"  Datasets : {args.datasets}")
    print(f"  Results  : {results_dir_wsl}")
    print(f"  Parallel : {args.parallel} workers")
    print(f"  Dry Run  : {args.dry_run}")
    print(f"  Selected : {len(selected)} experiments × {len(args.datasets)} datasets "
          f"= {len(selected) * len(args.datasets)} maximum possible runs\n")

    tasks = []
    for exp in selected:
        for dataset in args.datasets:
            # Skip datasets not in exp.datasets (e.g. EXP C is MedChangeQA only)
            if dataset not in exp.datasets:
                print(f"  [SKIP] {exp.experiment_id} skipped for dataset={dataset} "
                      f"(not in exp.datasets={exp.datasets})")
                continue
            tasks.append((exp, dataset, results_dir_wsl, args.dry_run))

    if args.parallel > 1 and not args.dry_run:
        print(
            f"  [Running {len(tasks)} tasks via ThreadPoolExecutor with {args.parallel} parallel workers...]")
        print(f"  (Note: Output logs from different evaluations will overlap, but JSON results will save cleanly)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(run_experiment, *t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  [ERROR] Task failed with exception: {e}")
    else:
        for t in tasks:
            run_experiment(*t)

    print(f"\n{'='*70}")
    print(f"  Done! Results in: {results_dir}")
    print(f"  Run analysis with:")
    print(
        f"    python scripts/analyze_comprehensive.py --results-dir {results_dir}")
    print(f"{'='*70}\n")
