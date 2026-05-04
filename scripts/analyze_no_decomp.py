import json
import numpy as np

ANSWER_MAP = {
    "A": {"healthfc": "True"},
    "B": {"healthfc": "False"},
    "C": {"healthfc": "Mixture"},
}

def nl(label):
    s = str(label).strip().upper()
    if s in ("0", "1", "2"):
        mk = {"0": "A", "1": "C", "2": "B"}.get(s, "C")
        return ANSWER_MAP.get(mk, {}).get("healthfc", s)
    if s in ANSWER_MAP:
        return ANSWER_MAP[s].get("healthfc", s) or s
    return s

fname = "results/v5_ablations/BEST1_no_decomp_healthfc_test.json"
with open(fname, encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
preds = [nl(r.get("pred_answer", "?")) for r in results]
gts = [nl(r.get("gt_label", r.get("gt_answer", "?"))) for r in results]
classes = sorted(set(g for g in gts if g != "?"))
per_class = {}
for cls in classes:
    tp = sum(1 for p, g in zip(preds, gts) if g == cls and p == cls)
    fp = sum(1 for p, g in zip(preds, gts) if g != cls and p == cls)
    fn = sum(1 for p, g in zip(preds, gts) if g == cls and p != cls)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    per_class[cls] = {"P": P, "R": R, "F1": F1, "n": tp + fn, "TP": tp}

mf1 = float(np.mean([v["F1"] for v in per_class.values()]))
acc = sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)

print("=" * 60)
print("  DATASET  : HEALTHFC (BEST1 NO DECOMP)")
print("=" * 60)
print(f"  Accuracy : {acc*100:.1f}%")
print(f"  Macro-F1 : {mf1*100:.1f}%")
print()
print(f"  -- Per-Class --")
print(f"  {'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'n':>5}")
for cls, v in sorted(per_class.items()):
    print(f"  {cls:<12} {v['P']*100:>6.1f}% {v['R']*100:>6.1f}% {v['F1']*100:>6.1f}% {v['n']:>5}")
