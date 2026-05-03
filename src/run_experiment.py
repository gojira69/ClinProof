"""
ClinProof CLI Entry Point
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ClinProof
from src.evaluation.benchmarks import load_dataset_by_name
from src.evaluation.metrics import evaluate_results, print_results_table
from src.utils.paths import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("run_experiment")


def main():
    parser = argparse.ArgumentParser(description="ClinProof Experiment Runner")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--dataset", default="medqa_us",
                        choices=["medqa_us", "medmcqa", "pubmedqa", "mmlu_med", "bioasq"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--no_citations", action="store_true")
    parser.add_argument("--eval_only", default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    config_name = Path(args.config).stem
    results_dir = cfg.get("evaluation", {}).get("results_dir", str(PROJECT_ROOT / "results"))
    save_dir = args.save_dir or f"{results_dir}/{config_name}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    log.info(f"Config: {args.config} | Dataset: {args.dataset} | Model: {cfg['model']['name']}")
    log.info(f"Retrieval: {cfg['retrieval']['mode']} | Save: {save_dir}")

    # Load dataset
    dataset = load_dataset_by_name(args.dataset, split=args.split)
    if args.n_samples:
        end = min(args.start_idx + args.n_samples, len(dataset))
        dataset = dataset[args.start_idx:end]
    log.info(f"Loaded {len(dataset)} questions from {args.dataset}")

    if args.eval_only:
        results = []
        for fname in sorted(Path(args.eval_only).glob("test_*.json")):
            with open(fname) as f:
                results.append(json.load(f))
        for i, r in enumerate(results):
            if i < len(dataset):
                r["ground_truth"] = dataset[i]["answer"]
    else:
        clinproof = ClinProof(config_path=args.config)
        results = clinproof.batch_verify(dataset, save_dir, start_idx=args.start_idx)
        for i, r in enumerate(results):
            if i < len(dataset):
                r["ground_truth"] = dataset[i]["answer"]

    # Evaluate
    log.info("Computing metrics...")
    metrics = evaluate_results(results, config=cfg, evaluate_citations=not args.no_citations)

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved: {metrics_path}")

    print_results_table(metrics, system_name=f"ClinProof [{config_name}] on {args.dataset.upper()}")


if __name__ == "__main__":
    main()
