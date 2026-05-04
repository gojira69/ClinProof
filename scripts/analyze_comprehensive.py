"""
ClinProof Comprehensive Results Analyzer
=========================================
Analyzes ALL result JSON files in a results directory.
Produces:
  1. Accuracy table (all runs × all datasets)
  2. Per-class Precision / Recall / Macro-F1 (critical for MedChangeQA)
  3. Vote confidence analysis
  4. Error categorization (retrieval failure vs reasoning failure vs label ambiguity)
  5. Subjective error sampling (top-20 wrong predictions with reasoning)
  6. Comparison table vs SOTA (from literature)

Usage:
    # Analyze v4 (existing fullpower runs)
    python scripts/analyze_comprehensive.py --results-dir results/v4

    # Analyze ablation runs
    python scripts/analyze_comprehensive.py --results-dir results/v5_ablations

    # Also compute reasoning metrics (NLI-based, slow)
    python scripts/analyze_comprehensive.py --results-dir results/v4 --reasoning-metrics

    # Export markdown table
    python scripts/analyze_comprehensive.py --results-dir results/v4 --markdown
"""
import os
import sys
import json
import re
import argparse
import textwrap
from collections import Counter, defaultdict
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

# ── SOTA Reference Table (from literature) ───────────────────────────────────
SOTA = {
    "bioasq": [
        {"system": "Vladika and Matthes [44]", "accuracy": 0.0, "f1": 0.617},
        {"system": "Lan et al. [30]",           "accuracy": 0.0, "f1": 0.601},
        {"system": "Bekoulis et al. [8]",       "accuracy": 0.0, "f1": 0.498},
    ],
    "medchangeqa": [
        {"system": "BioMistral 7B (Latest Labels)",  "accuracy": 0.354, "f1": 0.353},
        {"system": "Llama 3.3 70B (Latest Labels)",  "accuracy": 0.428, "f1": 0.341},
        {"system": "Mistral 24B (Latest Labels)",    "accuracy": 0.369, "f1": 0.337},
        {"system": "OLMo 2 13B (Latest Labels)",     "accuracy": 0.355, "f1": 0.332},
        {"system": "GPT-4o (Latest Labels)",          "accuracy": 0.352, "f1": 0.311},
    ],
    "healthfc": [
        {"system": "Vladika et al. [45] (Best)", "accuracy": 0.0, "f1": 0.675},
        {"system": "Bekoulis et al. [8]",       "accuracy": 0.0, "f1": 0.452},
        {"system": "Vladika and Matthes [44]",  "accuracy": 0.0, "f1": 0.406},
    ],
    "scifact": [
        {"system": "Bekoulis et al. [8] (Best)", "accuracy": 0.0, "f1": 0.526},
        {"system": "Vladika and Matthes [44]",  "accuracy": 0.0, "f1": 0.441},
        {"system": "Zaheer et al. [53]",        "accuracy": 0.0, "f1": 0.369},
    ],
    "medqa": [
        {"system": "MedRAG (GPT-3.5)",               "accuracy": 0.743, "f1": 0.0},
        {"system": "GPT-4 zero-shot",                "accuracy": 0.870, "f1": 0.0},
    ],
}
 
MERMAID_DIAGRAM = """
```mermaid
graph TD
    A[Medical Claim] --> B{Atomic Decomposition}
    B -->|Propositions| C[Multi-Stage Retrieval]
    C --> D[Stage 1: Keyword BM25]
    C --> E[Stage 1: GraphRAG / KG]
    D --> F[Stage 2: MedCPT Semantic PubMed]
    E --> F
    F --> G[Extractive Context Compression]
    G --> H[Ensemble LLM Reasoning]
    H --> I[Self-Consistency Voting]
    I --> J[Verified Veridicality Label]
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```
"""

# ── Label normalisation ───────────────────────────────────────────────────────

ANSWER_MAP = {
    "A": {"bioasq": "Yes",  "medchangeqa": "SUPPORTED",    "healthfc": "True",  "scifact": "SUPPORT",    "medqa": "A"},
    "B": {"bioasq": "No",   "medchangeqa": "REFUTED",      "healthfc": "False", "scifact": "CONTRADICT", "medqa": "B"},
    "C": {                   "medchangeqa": "NEI",          "healthfc": "Mixture","scifact": "NEI",       "medqa": "C"},
    "D": {                                                                                                "medqa": "D"},
}

def dataset_from_fname(fname: str) -> str:
    """Infer dataset name from result filename."""
    f = fname.lower()
    if   "bioasq"      in f: return "bioasq"
    elif "medchangeqa" in f: return "medchangeqa"
    elif "healthfc"    in f: return "healthfc"
    elif "scifact"     in f: return "scifact"
    elif "medqa"       in f: return "medqa"
    return "unknown"


def normalize_label(label: str, dataset: str) -> str:
    """Normalise a raw label (A/B/C or 0/1/2 or text) to canonical name."""
    s = str(label).strip().upper()
    
    # If it's a numeric string (used in HealthFC)
    if s in ("0", "1", "2"):
        if dataset == "healthfc":
            # HealthFC: 0=True(A), 1=Mixture(C), 2=False(B)
            mapped_key = {"0": "A", "1": "C", "2": "B"}.get(s, "C")
            return ANSWER_MAP[mapped_key].get(dataset, s)
        
    # Already a letter key A/B/C/D
    if s in ANSWER_MAP:
        mapping = ANSWER_MAP[s].get(dataset, s)
        return mapping if mapping else s
    
    return s


def compute_per_class(results, dataset: str, pred_field="pred_answer", gt_field="gt_answer"):
    """Compute precision, recall, F1, support per class. Returns dict."""
    # Use gt_label if available (more reliable for HealthFC right now)
    preds = [normalize_label(r.get(pred_field, "?"), dataset) for r in results]
    gts   = [normalize_label(r.get("gt_label", r.get(gt_field, "?")), dataset) for r in results]
    
    classes = sorted(set(gts))
    # Filter out unknown/unprocessed classes if any
    classes = [c for c in classes if c != "?"]
    
    per_class = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(preds, gts) if g == cls and p == cls)
        fp = sum(1 for p, g in zip(preds, gts) if g != cls and p == cls)
        fn = sum(1 for p, g in zip(preds, gts) if g == cls and p != cls)
        P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
        per_class[cls] = {"P": P, "R": R, "F1": F1, "support": tp+fn, "TP": tp}
    
    macro_p  = np.mean([v["P"]  for v in per_class.values()]) if per_class else 0.0
    macro_r  = np.mean([v["R"]  for v in per_class.values()]) if per_class else 0.0
    macro_f1 = np.mean([v["F1"] for v in per_class.values()]) if per_class else 0.0
    accuracy = sum(1 for p,g in zip(preds,gts) if p==g) / len(gts) if gts else 0.0
    
    return {
        "per_class": per_class,
        "macro_p":  float(macro_p),
        "macro_r":  float(macro_r),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "preds_norm": preds,
        "gts_norm": gts
    }


def vote_confidence_stats(results, votes: int = 3) -> dict:
    """Analyse self-consistency vote distributions."""
    unanimous, total, winner_fracs = 0, 0, []
    for r in results:
        vd = r.get("vote_distribution", {})
        if not vd: continue
        total += 1
        max_v = max(vd.values())
        if max_v == votes: unanimous += 1
        winner_fracs.append(max_v / votes)
    return {
        "unanimous_rate":   unanimous / total if total else 0.0,
        "avg_winner_frac":  float(np.mean(winner_fracs)) if winner_fracs else 0.0,
        "n": total
    }


def error_analysis(results, dataset: str, n_sample: int = 20) -> dict:
    """
    Categorize wrong predictions into:
      - retrieval_empty: no context retrieved
      - retrieval_short: very short context (<200 chars)
      - unanimous_wrong: all votes agreed on wrong answer (high confidence error)
      - split_vote_wrong: votes were split (model uncertain)
      - correct (reference)

    Also samples wrong cases for qualitative review.
    """
    wrong  = [r for r in results if not r.get("correct", False)]
    right  = [r for r in results if r.get("correct", False)]

    cats = Counter()
    for r in wrong:
        ctx = r.get("retrieved_context", "")
        vd  = r.get("vote_distribution", {})
        if not ctx or len(ctx.strip()) < 50:
            cats["retrieval_empty"] += 1
        elif len(ctx) < 200:
            cats["retrieval_short"] += 1
        elif vd and max(vd.values()) == sum(vd.values()):
            cats["unanimous_wrong"] += 1
        else:
            cats["split_vote_wrong"] += 1

    # Confusion matrix
    pred_dist  = Counter(r.get("pred_answer","?") for r in wrong)
    gt_dist    = Counter(r.get("gt_answer","?")   for r in wrong)
    conf_wrong = defaultdict(Counter)
    for r in wrong:
        conf_wrong[r.get("gt_answer","?")][r.get("pred_answer","?")] += 1

    # Sample wrong cases for qualitative review
    sample = wrong[:n_sample]
    sample_out = []
    for r in sample:
        reasoning = ""
        traces = r.get("reasoning_traces", [])
        if traces:
            reasoning = traces[0].get("step_by_step_thinking", "")[:300]
        ctx_snippet = (r.get("retrieved_context","")[:200] or "(empty)")
        sample_out.append({
            "id":        r.get("id","?"),
            "question":  r.get("question","")[:120],
            "gt":        r.get("gt_label", r.get("gt_answer", "?")),
            "pred":      r.get("pred_label", r.get("pred_answer", "?")),
            "votes":     r.get("vote_distribution", {}),
            "context_snippet": ctx_snippet,
            "reasoning_snippet": reasoning,
        })

    return {
        "total_wrong": len(wrong),
        "total_correct": len(right),
        "error_categories": dict(cats),
        "wrong_pred_dist": dict(pred_dist),
        "wrong_gt_dist":   dict(gt_dist),
        "confusion_wrong": {k: dict(v) for k,v in conf_wrong.items()},
        "sample_wrong_cases": sample_out,
    }


# ── File loader ───────────────────────────────────────────────────────────────

def load_result_files(results_dir: str, filter_tag: Optional[str] = None):
    """Load all JSON result files from directory. Returns list of (fname, data, dataset)."""
    files = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"): continue
        if filter_tag and filter_tag not in fname: continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            dataset = dataset_from_fname(fname)
            files.append((fname, data, dataset))
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
    return files


# ── Markdown output ───────────────────────────────────────────────────────────

def md_table(headers: list, rows: list) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    def _row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i,c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-"*(w+2) for w in widths) + "|"
    lines = [_row(headers), sep] + [_row(r) for r in rows]
    return "\n".join(lines)


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(results_dir: str, use_reasoning_metrics: bool = False,
            markdown: bool = False, max_nli: int = 50,
            filter_tag: Optional[str] = None) -> None:

    print(f"\n{'='*70}")
    print(f"  ClinProof Comprehensive Analysis")
    print(f"  Results dir: {results_dir}")
    print(f"{'='*70}\n")
 
    if markdown:
        print("## CLINPROOF ARCHITECTURE\n")
        print(MERMAID_DIAGRAM)
        print("\n---\n")

    files = load_result_files(results_dir, filter_tag)
    if not files:
        print(f"  [ERROR] No JSON files found in {results_dir}")
        return

    # ── 1. Summary accuracy table ─────────────────────────────────────────
    print("## 1. ACCURACY SUMMARY\n")
    summary_rows = []
    all_metrics  = {}   # fname → detailed metrics

    for fname, data, dataset in files:
        results    = data.get("results", [])
        cfg        = data.get("config", {})
        n_votes    = cfg.get("votes", "?")
        
        # Determine model display
        models_list = cfg.get("models", [])
        if not models_list:
            model = cfg.get("model", "?")
        else:
            # For ensembles, show initials or short names
            model = ",".join([m.split(":")[0].split("/")[-1] for m in models_list])

        use_kg     = cfg.get("use_graph", False)
        use_bm25   = cfg.get("use_bm25", True)
        use_pubmed = cfg.get("use_pubmed", False)
        no_decomp  = cfg.get("no_decomp", False)
        exp_id     = cfg.get("experiment_id", "")
        r_alpha    = cfg.get("recency_alpha", 0.0)

        if not results:
            print(f"  [SKIP] {fname}: 0 results")
            continue

        m = compute_per_class(results, dataset=dataset)
        vc = vote_confidence_stats(results, votes=n_votes if isinstance(n_votes, int) else 3)

        tag_col = fname.replace(".json","")
        row = [
            tag_col,
            dataset,
            model,
            "✗" if no_decomp else "✓",
            "✓" if use_kg     else "✗",
            "✓" if use_bm25   else "✗",
            "✓" if use_pubmed else "✗",
            f"{r_alpha:.1f}",
            n_votes,
            len(results),
            f"{m['accuracy']*100:.1f}%",
            f"{m['macro_p']*100:.1f}%",
            f"{m['macro_r']*100:.1f}%",
            f"{m['macro_f1']*100:.1f}%",
            f"{vc['unanimous_rate']*100:.0f}%",
        ]
        summary_rows.append(row)
        all_metrics[fname] = {
            "dataset": dataset,
            "per_class": m["per_class"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "vote_conf": vc,
            "results": results,
            "cfg": cfg,
            "model": model,
            "exp_id": exp_id,
        }

    headers = ["Tag","Dataset","Model","Decomp","KG","BM25","PMed","Rec","Votes","N","Acc","P","R","F1","Unan%"]
    if markdown:
        print(md_table(headers, summary_rows))
    else:
        try:
            df = pd.DataFrame(summary_rows, columns=headers)
            print(df.to_string(index=False))
        except Exception:
            for row in summary_rows:
                print("  " + "  ".join(str(c) for c in row))

    # ── 2. Per-class breakdown ─────────────────────────────────────────────
    print("\n\n## 2. PER-CLASS PRECISION / RECALL / F1\n")
    for fname, info in all_metrics.items():
        pc = info["per_class"]
        if not pc: continue
        print(f"\n  [{info['exp_id'] or fname}] {info['dataset'].upper()}  "
              f"Model={info['model'][:25]}  n={len(info['results'])}")
        cls_rows = [
            [cls, f"{v['P']*100:.1f}%", f"{v['R']*100:.1f}%",
             f"{v['F1']*100:.1f}%", v['support'], v['TP']]
            for cls, v in sorted(pc.items())
        ]
        if markdown:
            print(md_table(["Class","Precision","Recall","F1","Support","TP"], cls_rows))
        else:
            try:
                df_cls = pd.DataFrame(cls_rows,
                                      columns=["Class","Precision","Recall","F1","Support","TP"])
                print(df_cls.to_string(index=False))
            except Exception:
                for r in cls_rows:
                    print("    " + "  ".join(str(c) for c in r))
        print(f"  → Macro-F1: {info['macro_f1']*100:.1f}%  |  "
              f"Unanimous: {info['vote_conf']['unanimous_rate']*100:.0f}%  |  "
              f"AvgWinnerFrac: {info['vote_conf']['avg_winner_frac']:.2f}")

    # ── 3. Error Analysis ──────────────────────────────────────────────────
    print("\n\n## 3. ERROR ANALYSIS (Wrong Predictions)\n")
    for fname, info in all_metrics.items():
        ea = error_analysis(info["results"], info["dataset"])
        print(f"\n  [{info['exp_id'] or fname}] {info['dataset'].upper()}")
        print(f"  Wrong={ea['total_wrong']}  Correct={ea['total_correct']}")
        print(f"  Error categories: {ea['error_categories']}")
        print(f"  Wrong pred dist:  {ea['wrong_pred_dist']}")
        print(f"  Confusion (gt→pred): {dict(ea['confusion_wrong'])}")

        # Print top-3 sample wrong cases
        print(f"\n  Sample wrong cases (first 3):")
        for case in ea["sample_wrong_cases"][:3]:
            print(f"\n    ID={case['id']}")
            print(f"    Q : {case['question']}")
            print(f"    GT: {case['gt']}  PRED: {case['pred']}  Votes: {case['votes']}")
            print(f"    Context: {case['context_snippet'][:100]}...")
            print(f"    Reasoning: {case['reasoning_snippet'][:150]}...")

    # ── 4. SOTA Comparison ─────────────────────────────────────────────────
    print("\n\n## 4. SOTA COMPARISON\n")

    # Group our results by dataset, pick best accuracy per dataset
    our_best: dict[str, dict] = {}
    for fname, info in all_metrics.items():
        ds = info["dataset"]
        if ds not in our_best or info["accuracy"] > our_best[ds]["accuracy"]:
            our_best[ds] = {
                "accuracy": info["accuracy"],
                "macro_f1": info["macro_f1"],
                "tag":   fname.replace(".json",""),
                "model": info["model"],
            }

    for ds, sota_list in SOTA.items():
        print(f"\n  Dataset: {ds.upper()}")
        rows = []
        for s in sota_list:
            rows.append([
                s["system"],
                f"{s['accuracy']*100:.1f}%" if s["accuracy"] > 0 else "—",
                f"{s['f1']*100:.1f}%" if s.get("f1",0) > 0 else "—"
            ])
        if ds in our_best:
            ob = our_best[ds]
            rows.append([f"ClinProof ({ob['model'][:20]}) [BEST]",
                         f"{ob['accuracy']*100:.1f}%",
                         f"{ob['macro_f1']*100:.1f}%"])
        if markdown:
            print(md_table(["System","Accuracy","Macro-F1"], rows))
        else:
            try:
                df_sota = pd.DataFrame(rows, columns=["System","Accuracy","Macro-F1"])
                print(df_sota.to_string(index=False))
            except Exception:
                for r in rows:
                    print("  " + "  ".join(str(c) for c in r))

    # ── 5. Reasoning Metrics (optional) ────────────────────────────────────
    if use_reasoning_metrics:
        print("\n\n## 5. REASONING METRICS\n")
        try:
            from src.evaluation.reasoning_metrics import ReasoningMetrics
            rm = ReasoningMetrics(use_nli=True, use_bertscore=False)
        except Exception as e:
            print(f"  [ERROR] Could not load ReasoningMetrics: {e}")
            return

        for fname, info in all_metrics.items():
            print(f"\n  [{info['exp_id'] or fname}] Computing reasoning metrics "
                  f"(max_nli={max_nli}) ...")
            scores = rm.score_results(
                info["results"],
                n_votes = info["cfg"].get("votes", 3),
                max_nli_samples = max_nli,
            )
            print(f"  ROUGE-L mean            : {scores.get('rouge_l', {}).get('mean', 'N/A')}")
            print(f"  Faithfulness mean       : {scores.get('faithfulness', {}).get('mean', 'N/A')}")
            print(f"  Faithfulness correct    : {scores.get('faithfulness', {}).get('correct_mean', 'N/A')}")
            print(f"  Faithfulness wrong      : {scores.get('faithfulness', {}).get('wrong_mean', 'N/A')}")
            print(f"  Evidence Grounding mean : {scores.get('evidence_grounding', {}).get('mean', 'N/A')}")
            print(f"  Citation Recall         : {scores.get('citation', {}).get('recall', 'N/A')}")
            print(f"  Citation Precision      : {scores.get('citation', {}).get('precision', 'N/A')}")
            print(f"  Vote Info               : {scores.get('vote_confidence', {})}")

    # ── 6. Prompt Bias Diagnosis ───────────────────────────────────────────
    print("\n\n## 6. NEI / CONSERVATIVE BIAS DIAGNOSIS\n")
    print("  (Checking if model over-predicts 'safe' answers: NEI, No, False)\n")

    for fname, info in all_metrics.items():
        results = info["results"]
        all_preds = Counter(r.get("pred_answer","?") for r in results)
        all_gts   = Counter(r.get("gt_answer","?")   for r in results)
        total = len(results)
        print(f"  [{info['exp_id'] or fname}] {info['dataset'].upper()}  n={total}")
        for choice in sorted(all_preds.keys()):
            pred_n = all_preds[choice]
            gt_n   = all_gts.get(choice, 0)
            bias   = (pred_n - gt_n) / total * 100 if total else 0
            marker = " ← BIASED" if abs(bias) > 10 else ""
            print(f"    {choice:>2}: gt={gt_n:>4} ({gt_n/total*100:.0f}%)  "
                  f"pred={pred_n:>4} ({pred_n/total*100:.0f}%)  "
                  f"bias={bias:+.0f}%{marker}")
        print()

    print(f"\n{'═'*70}")
    print("  Analysis complete.")
    print(f"{'═'*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ClinProof Comprehensive Results Analyzer"
    )
    parser.add_argument("--results-dir", default=project_path("results", "v4"),
                        help="Directory containing result JSON files")
    parser.add_argument("--filter-tag",  type=str, default=None,
                        help="Only analyze files containing this string")
    parser.add_argument("--reasoning-metrics", action="store_true",
                        help="Also compute NLI-based reasoning metrics (slow)")
    parser.add_argument("--max-nli",   type=int, default=50,
                        help="Max samples for NLI metrics (default=50)")
    parser.add_argument("--markdown",  action="store_true",
                        help="Print tables in markdown format")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output file path (writes directly in UTF-8 to bypass Windows console encoding issues)")
    args = parser.parse_args()

    if args.output:
        import sys
        sys.stdout = open(args.output, "w", encoding="utf-8")

    analyze(
        results_dir          = args.results_dir,
        use_reasoning_metrics= args.reasoning_metrics,
        markdown             = args.markdown,
        max_nli              = args.max_nli,
        filter_tag           = args.filter_tag,
    )
