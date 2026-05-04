"""Analyze a single result file: BEST2_hetero_dense_bm25_healthfc_test.json"""
import json, sys, numpy as np
sys.path.insert(0, ".")

ANSWER_MAP = {
    "A": {"bioasq": "Yes",  "medchangeqa": "SUPPORTED", "healthfc": "True",  "scifact": "SUPPORT"},
    "B": {"bioasq": "No",   "medchangeqa": "REFUTED",   "healthfc": "False", "scifact": "CONTRADICT"},
    "C": {                   "medchangeqa": "NEI",       "healthfc": "Mixture","scifact": "NEI"},
}

def nl(label, dataset):
    s = str(label).strip().upper()
    if s in ("0","1","2") and dataset == "healthfc":
        mk = {"0":"A","1":"C","2":"B"}.get(s,"C")
        return ANSWER_MAP.get(mk, {}).get(dataset, s)
    if s in ANSWER_MAP:
        return ANSWER_MAP[s].get(dataset, s) or s
    return s

fname = "results/BEST2_hetero_dense_bm25_healthfc_test.json"
with open(fname, encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
cfg = data.get("config", {})
dataset = "healthfc"
preds = [nl(r.get("pred_answer","?"), dataset) for r in results]
gts   = [nl(r.get("gt_label", r.get("gt_answer","?")), dataset) for r in results]
classes = sorted(set(g for g in gts if g != "?"))
per_class = {}
for cls in classes:
    tp = sum(1 for p,g in zip(preds,gts) if g==cls and p==cls)
    fp = sum(1 for p,g in zip(preds,gts) if g!=cls and p==cls)
    fn = sum(1 for p,g in zip(preds,gts) if g==cls and p!=cls)
    P = tp/(tp+fp) if (tp+fp)>0 else 0.0
    R = tp/(tp+fn) if (tp+fn)>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    per_class[cls] = {"P":P,"R":R,"F1":F1,"support":tp+fn,"TP":tp}

mf1 = float(np.mean([v["F1"] for v in per_class.values()]))
mp  = float(np.mean([v["P"]  for v in per_class.values()]))
mr  = float(np.mean([v["R"]  for v in per_class.values()]))
acc = sum(1 for p,g in zip(preds,gts) if p==g)/len(gts)
models = cfg.get("models") or [cfg.get("model","?")]

print("="*60)
print("  DATASET  : HEALTHFC")
print("="*60)
print(f"  File     : BEST2_hetero_dense_bm25_healthfc_test.json")
print(f"  Exp ID   : {cfg.get('experiment_id','BEST2')}")
print(f"  N (qs)   : {len(results)}")
print()
print(f"  -- Metrics --")
print(f"  Accuracy : {acc*100:.1f}%")
print(f"  Macro-P  : {mp*100:.1f}%")
print(f"  Macro-R  : {mr*100:.1f}%")
print(f"  Macro-F1 : {mf1*100:.1f}%")
print()
print(f"  -- Per-Class --")
print(f"  {'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'n':>5} {'TP':>5}")
for cls, v in sorted(per_class.items()):
    print(f"  {cls:<12} {v['P']*100:>6.1f}% {v['R']*100:>6.1f}% {v['F1']*100:>6.1f}% {v['support']:>5} {v['TP']:>5}")
print()
print(f"  -- Config --")
print(f"  Models    : {models}")
print(f"  Votes     : {cfg.get('votes','?')}")
print(f"  KG        : {cfg.get('use_graph', False)}")
print(f"  BM25      : {cfg.get('use_bm25', True)}")
print(f"  PubMed    : {cfg.get('use_pubmed', False)}")
print(f"  Decomp    : {not cfg.get('no_decomp', False)}")
print(f"  LiveSearch: {cfg.get('enable_live_search', False)}")
print(f"  RecAlpha  : {cfg.get('recency_alpha', 0.0)}")
print(f"  k (top-k) : {cfg.get('k','?')}")
print(f"  Compression: {not cfg.get('no_compression', False)}")
