import json
import os
import glob
from sklearn.metrics import f1_score, recall_score, accuracy_score

print("=== Sanity Ablation Metrics ===\n")

for file in sorted(glob.glob('results/sanity_*_healthfc_test.json')):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    ctx_len = []
    
    for r in data['results']:
        y_true.append(r['gt_answer'])
        y_pred.append(r['pred_answer'])
        # Approximate tokens based on characters / 4
        ctx_len.append(len(r.get('retrieved_context', '')) / 4.0)
    
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        avg_toks = sum(ctx_len) / len(ctx_len)
        
        name = os.path.basename(file).split('_')[1].upper()
        print(f"[{name}] (n={len(y_true)})")
        print(f"  Accuracy:   {acc*100:.1f}%")
        print(f"  Macro-F1:   {f1*100:.1f}%")
        print(f"  Macro-Rec:  {rec*100:.1f}%")
        print(f"  Avg Tokens: {avg_toks:.1f}")
        print("-" * 30)
