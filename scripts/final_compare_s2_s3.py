"""Compare S2 and S3 experiments on full HealthFC."""
import json
import numpy as np

AMAP = {"A": "TRUE", "B": "FALSE", "C": "MIXTURE"}


def nl(label):
    s = str(label).strip().upper()
    if s in ("0", "1", "2"):
        mk = {"0": "A", "1": "C", "2": "B"}.get(s, "C")
        return AMAP.get(mk, s)
    return AMAP.get(s, s)


def analyze(fname):
    with open(fname, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
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
        per_class[cls] = {"P": P, "R": R, "F1": F1, "n": tp + fn}
    
    acc = sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
    mf1 = float(np.mean([v["F1"] for v in per_class.values()]))
    return {"acc": acc, "mf1": mf1, "n": len(results), "per_class": per_class}


s2 = analyze("results/v5_ablations/S2_comp_on_healthfc_test.json")
s3 = analyze("results/v5_ablations/S3_comp_off_healthfc_test.json")

print("\n=== FINAL COMPRESSION COMPARISON (Full HealthFC N=75) ===")
if s2 and s3:
    print(f"{'Metric':<10} | {'S2 (Comp ON)':>12} | {'S3 (Comp OFF)':>12} | {'Delta':>8}")
    print("-" * 55)
    print(f"{'Accuracy':<10} | {s2['acc']*100:>11.1f}% | {s3['acc']*100:>11.1f}% | {(s2['acc']-s3['acc'])*100:>+7.1f}%")
    print(f"{'Macro-F1':<10} | {s2['mf1']*100:>11.1f}% | {s3['mf1']*100:>11.1f}% | {(s2['mf1']-s3['mf1'])*100:>+7.1f}%")
    
    print("\n--- Per-Class F1 ---")
    for cls in sorted(s2["per_class"].keys()):
        f2 = s2["per_class"][cls]["F1"] * 100
        f3 = s3["per_class"][cls]["F1"] * 100
        print(f"  {cls:<10}: S2={f2:>5.1f}% | S3={f3:>5.1f}% | Diff={(f2-f3):>+5.1f}%")
else:
    print("Error: Could not load one or both result files.")
