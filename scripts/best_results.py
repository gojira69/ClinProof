"""Extract best result per dataset from the results folder."""
import json, os, sys
import numpy as np

sys.path.insert(0, ".")

ANSWER_MAP = {
    "A": {"bioasq": "Yes",  "medchangeqa": "SUPPORTED",    "healthfc": "True",  "scifact": "SUPPORT",    "medqa": "A"},
    "B": {"bioasq": "No",   "medchangeqa": "REFUTED",      "healthfc": "False", "scifact": "CONTRADICT", "medqa": "B"},
    "C": {                   "medchangeqa": "NEI",          "healthfc": "Mixture","scifact": "NEI",       "medqa": "C"},
}

def dataset_from_fname(fname):
    f = fname.lower()
    if "bioasq"      in f: return "bioasq"
    elif "medchangeqa" in f: return "medchangeqa"
    elif "healthfc"    in f: return "healthfc"
    elif "scifact"     in f: return "scifact"
    elif "medqa"       in f: return "medqa"
    return "unknown"

def normalize_label(label, dataset):
    s = str(label).strip().upper()
    if s in ("0","1","2") and dataset == "healthfc":
        mapped_key = {"0":"A","1":"C","2":"B"}.get(s,"C")
        return ANSWER_MAP.get(mapped_key, {}).get(dataset, s)
    if s in ANSWER_MAP:
        return ANSWER_MAP[s].get(dataset, s) or s
    return s

results_dir = "results"
min_n = 50
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--min-n" and i+1 < len(sys.argv):
        min_n = int(sys.argv[i+1])
    elif not arg.startswith("--") and sys.argv[i-1] != "--min-n":
        results_dir = arg
best_per_ds = {}

for fname in os.listdir(results_dir):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(results_dir, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    dataset = dataset_from_fname(fname)
    if dataset == "unknown":
        continue
    results = data.get("results", [])
    if not results or len(results) < min_n:
        continue
    cfg = data.get("config", {})
    preds = [normalize_label(r.get("pred_answer","?"), dataset) for r in results]
    gts   = [normalize_label(r.get("gt_label", r.get("gt_answer","?")), dataset) for r in results]
    classes = sorted(set(g for g in gts if g != "?"))
    per_class = {}
    for cls in classes:
        tp = sum(1 for p,g in zip(preds,gts) if g==cls and p==cls)
        fp = sum(1 for p,g in zip(preds,gts) if g!=cls and p==cls)
        fn = sum(1 for p,g in zip(preds,gts) if g==cls and p!=cls)
        P  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        R  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
        per_class[cls] = {"P":P,"R":R,"F1":F1,"support":tp+fn,"TP":tp}
    macro_f1 = float(np.mean([v["F1"] for v in per_class.values()])) if per_class else 0.0
    macro_p  = float(np.mean([v["P"]  for v in per_class.values()])) if per_class else 0.0
    macro_r  = float(np.mean([v["R"]  for v in per_class.values()])) if per_class else 0.0
    accuracy = sum(1 for p,g in zip(preds,gts) if p==g)/len(gts) if gts else 0.0

    if dataset not in best_per_ds or accuracy > best_per_ds[dataset]["accuracy"]:
        best_per_ds[dataset] = {
            "fname"    : fname,
            "accuracy" : accuracy,
            "macro_f1" : macro_f1,
            "macro_p"  : macro_p,
            "macro_r"  : macro_r,
            "per_class": per_class,
            "n"        : len(results),
            "cfg"      : cfg,
        }

for ds, info in sorted(best_per_ds.items()):
    cfg    = info["cfg"]
    models = cfg.get("models") or [cfg.get("model","?")]
    print(f"\n{'='*60}")
    print(f"  DATASET  : {ds.upper()}")
    print(f"{'='*60}")
    print(f"  File     : {info['fname']}")
    print(f"  Exp ID   : {cfg.get('experiment_id','?')}")
    print(f"  N (qs)   : {info['n']}")
    print()
    print(f"  ── Metrics ──────────────────────────────────")
    print(f"  Accuracy : {info['accuracy']*100:.1f}%")
    print(f"  Macro-P  : {info['macro_p']*100:.1f}%")
    print(f"  Macro-R  : {info['macro_r']*100:.1f}%")
    print(f"  Macro-F1 : {info['macro_f1']*100:.1f}%")
    print()
    print(f"  ── Per-Class ────────────────────────────────")
    print(f"  {'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'n':>5}")
    for cls, v in sorted(info["per_class"].items()):
        print(f"  {cls:<12} {v['P']*100:>6.1f}% {v['R']*100:>6.1f}% {v['F1']*100:>6.1f}% {v['support']:>5}")
    print()
    print(f"  ── Config ───────────────────────────────────")
    print(f"  Models    : {models}")
    print(f"  Votes     : {cfg.get('votes','?')}")
    print(f"  KG        : {cfg.get('use_graph', False)}")
    print(f"  BM25      : {cfg.get('use_bm25', True)}")
    print(f"  PubMed    : {cfg.get('use_pubmed', False)}")
    print(f"  Decomp    : {not cfg.get('no_decomp', False)}")
    print(f"  LiveSearch: {cfg.get('enable_live_search', False)}")
    print(f"  LiveK     : {cfg.get('live_search_k', 'N/A')}")
    print(f"  RecAlpha  : {cfg.get('recency_alpha', 0.0)}")
    print(f"  k (top-k) : {cfg.get('k', '?')}")
    print(f"  Compression: {not cfg.get('no_compression', False)}")
