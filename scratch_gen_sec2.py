import json
import os
from collections import defaultdict

results_dir = 'results'
datasets = {
    'bioasq': {'classes': ['yes', 'no']},
    'healthfc': {'classes': ['true', 'false', 'mixture']},
    'medchangeqa': {'classes': ['supported', 'refuted', 'not enough info']}
}

experiments = {
    'bioasq': {},
    'healthfc': {},
    'medchangeqa': {}
}

for f in os.listdir(results_dir):
    if not f.endswith('.json'): continue
    path = os.path.join(results_dir, f)
    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            continue
    
    # identify dataset
    ds = None
    if 'bioasq' in f: ds = 'bioasq'
    elif 'healthfc' in f: ds = 'healthfc'
    elif 'medchangeqa' in f: ds = 'medchangeqa'
    else: continue

    exp_id = f.split('_')[0]
    
    # extract results
    results = data.get('results', [])
    if not results: continue

    metrics = {}
    
    y_true = []
    y_pred = []
    for r in results:
        gt = str(r.get('gt_label', '')).strip().lower()
        pred = str(r.get('pred_label', '')).strip().lower()
        
        if ds == 'bioasq':
            if gt in ['yes', 'supported', 'true', 'a', 'yes.']: gt = 'yes'
            if gt in ['no', 'refuted', 'false', 'b', 'no.']: gt = 'no'
            if pred in ['yes', 'supported', 'true', 'a', 'yes.']: pred = 'yes'
            if pred in ['no', 'refuted', 'false', 'b', 'no.']: pred = 'no'
            if pred == 'nei' or pred == 'not enough info': pred = 'no'
            
        if ds == 'healthfc':
            if gt in ['true', 'yes', 'supported', '0']: gt = 'true'
            if gt in ['false', 'no', 'refuted', '1']: gt = 'false'
            if gt in ['mixture', '2']: gt = 'mixture'
            if pred in ['true', 'yes', 'supported', '0']: pred = 'true'
            if pred in ['false', 'no', 'refuted', '1']: pred = 'false'
            if pred in ['mixture', '2']: pred = 'mixture'
            
        y_true.append(gt)
        y_pred.append(pred)
        
    experiments[ds][exp_id] = {'y_true': y_true, 'y_pred': y_pred, 'tag': f.replace('.json', '')}

def compute_metrics(y_true, y_pred, classes):
    # Confusion Matrix
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in classes and p in classes:
            cm[t][p] += 1
            
    # Class-wise metrics
    class_metrics = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        class_metrics[c] = {'P': precision, 'R': recall, 'F1': f1, 'Support': support}
    
    # Macro metrics
    macro_p = sum(m['P'] for m in class_metrics.values()) / len(classes)
    macro_r = sum(m['R'] for m in class_metrics.values()) / len(classes)
    macro_f1 = sum(m['F1'] for m in class_metrics.values()) / len(classes)
    correct = sum(cm[c][c] for c in classes)
    total = sum(sum(row.values()) for row in cm.values())
    acc = correct / total if total > 0 else 0
    
    return {'Acc': acc, 'Macro-P': macro_p, 'Macro-R': macro_r, 'Macro-F1': macro_f1, 'CM': cm, 'Class': class_metrics}

markdown = "# Section 2: Results (Full Granularity)\n\n"

for ds, exps in experiments.items():
    markdown += f"## {ds.upper()}\n\n"
    classes = datasets[ds]['classes']
    
    # Overall summary table
    markdown += "### Overall Metrics\n\n"
    markdown += "| ID | Tag | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |\n"
    markdown += "|---|---|---|---|---|---|\n"
    
    # Calculate and store for detailed breakdown
    computed = {}
    sorted_exps = sorted(exps.keys())
    for exp_id in sorted_exps:
        data = exps[exp_id]
        m = compute_metrics(data['y_true'], data['y_pred'], classes)
        computed[exp_id] = m
        markdown += f"| {exp_id} | {data['tag']} | {m['Acc']:.3f} | {m['Macro-P']:.3f} | {m['Macro-R']:.3f} | {m['Macro-F1']:.3f} |\n"
    
    markdown += "\n"
    
    # Detail per experiment
    for exp_id in sorted_exps:
        m = computed[exp_id]
        data = exps[exp_id]
        markdown += f"### {exp_id} - {data['tag']}\n\n"
        
        # Class-wise metrics
        markdown += "#### Class-wise Metrics\n\n"
        markdown += "| Class | Precision | Recall | F1 | Support |\n"
        markdown += "|---|---|---|---|---|\n"
        for c in classes:
            cmets = m['Class'][c]
            markdown += f"| {c} | {cmets['P']:.3f} | {cmets['R']:.3f} | {cmets['F1']:.3f} | {cmets['Support']} |\n"
        markdown += "\n"
        
        # Confusion matrix
        markdown += "#### Confusion Matrix (True \\ Pred)\n\n"
        header = "| True \\ Pred | " + " | ".join(classes) + " |\n"
        sep = "|---|" + "|".join(["---" for _ in classes]) + "|\n"
        markdown += header + sep
        for c_true in classes:
            row = f"| **{c_true}** | " + " | ".join(str(m['CM'][c_true][c_pred]) for c_pred in classes) + " |\n"
            markdown += row
        markdown += "\n"
        
        # Per-class error breakdown
        markdown += "#### Per-class Error Breakdown\n\n"
        for c_true in classes:
            errors = sum(m['CM'][c_true][c_pred] for c_pred in classes if c_pred != c_true)
            support = m['Class'][c_true]['Support']
            err_rate = errors / support if support > 0 else 0
            
            markdown += f"- **{c_true}**: {errors} errors out of {support} ({err_rate:.1%})\n"
            for c_pred in classes:
                if c_pred != c_true:
                    count = m['CM'][c_true][c_pred]
                    if count > 0:
                        markdown += f"  - Misclassified as **{c_pred}**: {count}\n"
        markdown += "\n---\n\n"

with open('section_2_results.md', 'w', encoding='utf-8') as f:
    f.write(markdown)
