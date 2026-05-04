import json
import glob
import os
import numpy as np
from collections import defaultdict

def parse_label(dataset, raw_label):
    raw_label = str(raw_label).strip().lower()
    if dataset == "bioasq":
        if raw_label in ["yes", "a", "1", "true"]: return "yes"
        if raw_label in ["no", "b", "0", "false"]: return "no"
    elif dataset == "healthfc":
        if raw_label in ["true", "a", "0"]: return "true"
        if raw_label in ["false", "b", "1"]: return "false"
        if raw_label in ["mixture", "c", "2"]: return "mixture"
    elif dataset == "medchangeqa":
        if raw_label in ["supported", "a", "true"]: return "supported"
        if raw_label in ["refuted", "b", "false"]: return "refuted"
        if raw_label in ["not enough information", "c", "nei"]: return "nei"
    return raw_label

def calculate_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = "unknown"
    if "bioasq" in file_path.lower(): dataset = "bioasq"
    elif "healthfc" in file_path.lower(): dataset = "healthfc"
    elif "medchangeqa" in file_path.lower(): dataset = "medchangeqa"
    
    results = data.get("results", [])
    
    # M1 & M2 tracking
    bins = defaultdict(list)
    unanimous_correct = 0
    unanimous_total = 0
    
    # M3 tracking
    nei_tp = 0
    nei_fp = 0
    nei_fn = 0
    
    # M4 tracking
    recent_correct = 0
    recent_total = 0
    older_correct = 0
    older_total = 0
    
    total_acc = 0
    
    for r in results:
        gt_raw = r.get("gt_label", r.get("gt_answer"))
        pred_raw = r.get("pred_label", r.get("pred_answer"))
        gt = parse_label(dataset, gt_raw)
        pred = parse_label(dataset, pred_raw)
        
        is_correct = r.get("correct", False)
        if is_correct: total_acc += 1
        
        # M1: Confidence Binning
        vd = r.get("vote_distribution", {})
        if vd:
            total_votes = sum(vd.values())
            max_votes = max(vd.values())
            conf = max_votes / total_votes
            bins[conf].append(is_correct)
            
            # M2: Selective Accuracy
            if conf == 1.0:
                unanimous_total += 1
                if is_correct: unanimous_correct += 1
                
        # M3: NEI Tracking
        if dataset == "medchangeqa":
            if pred == "nei" and gt == "nei": nei_tp += 1
            if pred == "nei" and gt != "nei": nei_fp += 1
            if pred != "nei" and gt == "nei": nei_fn += 1
            
            # M4: Temporal Sensitivity
            q_info = r.get("question", "")
            if "change_year" in r:
                # If we have explicit metadata
                pass 
            # In MedChangeQA, we can usually extract metadata if provided, but we skip if not explicitly present.
            
    # Calculate M1: ECE
    ece = 0.0
    total_samples = len(results)
    for conf, acc_list in bins.items():
        bin_acc = np.mean(acc_list)
        weight = len(acc_list) / total_samples
        ece += weight * abs(bin_acc - conf)
        
    # Calculate M2: Selective Accuracy
    sel_acc = unanimous_correct / unanimous_total if unanimous_total > 0 else 0.0
    coverage = unanimous_total / total_samples if total_samples > 0 else 0.0
    
    # Calculate M3: NEI Metrics
    nei_prec = nei_tp / (nei_tp + nei_fp) if (nei_tp + nei_fp) > 0 else 0.0
    nei_rec = nei_tp / (nei_tp + nei_fn) if (nei_tp + nei_fn) > 0 else 0.0
    
    return {
        "n": total_samples,
        "acc": total_acc / total_samples if total_samples > 0 else 0.0,
        "ece": ece,
        "sel_acc": sel_acc,
        "coverage": coverage,
        "nei_prec": nei_prec,
        "nei_rec": nei_rec
    }

print("=== ClinProof Calibration & Selective Accuracy (M1-M2) ===")
print("| Config | Dataset | N | Base Acc | M1: ECE | M2: Sel. Acc | Unanimous Coverage |")
print("|--------|---------|---|----------|---------|--------------|--------------------|")

all_files = glob.glob("results/*.json") + glob.glob("results/v5_ablations/*.json")
all_files = list(set(all_files)) # deduplicate

results_list = []
for path in all_files:
    path = path.replace("\\", "/")
    if "sanity" in path.lower():
        continue # Skip sanity checks
        
    m = calculate_metrics(path)
    if m["n"] == 0: continue
    
    # Extract config name from filename
    name = os.path.basename(path).replace(".json", "").replace("_bioasq", "").replace("_healthfc_test", "").replace("_medchangeqa", "").replace("_all", "")
    dataset = "BioASQ" if "bioasq" in path.lower() else ("HealthFC" if "healthfc" in path.lower() else "MedChangeQA")
    
    results_list.append({
        "name": name,
        "dataset": dataset,
        "n": m["n"],
        "acc": m["acc"],
        "ece": m["ece"],
        "sel_acc": m["sel_acc"],
        "coverage": m["coverage"]
    })

# Sort by dataset then by config name
results_list.sort(key=lambda x: (x["dataset"], x["name"]))

for r in results_list:
    print(f"| {r['name']:<25} | {r['dataset']:<8} | {r['n']:<3} | {r['acc']*100:>7.1f}% | {r['ece']*100:>6.1f}% | {r['sel_acc']*100:>11.1f}% | {r['coverage']*100:>17.1f}% |")

