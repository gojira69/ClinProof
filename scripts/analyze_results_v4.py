import json, os

results_dir = "/mnt/d/Harsha/AoLM/ClinProof/results/v4"
files = os.listdir(results_dir)
print("Files:", files)

for f in sorted(files):
    path = os.path.join(results_dir, f)
    with open(path) as fp:
        data = json.load(fp)
    results = data.get("results", [])
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    acc = correct/total if total else 0
    complete = data.get("complete", False)
    cfg = data.get("config", {})
    print(f"\n--- {f} ---")
    print(f"  Completed: {complete}, Results: {total}, Correct: {correct}, Accuracy: {acc:.1%}")
    print(f"  Config: votes={cfg.get('votes')}, k={cfg.get('k')}, bm25_candidates={cfg.get('bm25_candidates')}, model={cfg.get('model')}, use_bm25={cfg.get('use_bm25')}, use_graph={cfg.get('use_graph')}")

    # Vote distribution analysis
    all_votes = []
    for r in results:
        vd = r.get("vote_distribution", {})
        if vd:
            max_votes = max(vd.values())
            all_votes.append(max_votes)
    if all_votes:
        unanimous = sum(1 for v in all_votes if v == cfg.get("votes", 1))
        print(f"  Unanimous votes: {unanimous}/{total} ({unanimous/total:.1%})")

    # Per-class breakdown for 3-class datasets
    if results and "gt_label" in results[0]:
        from collections import Counter, defaultdict
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            lbl = r.get("gt_label", r.get("gt_answer", "?"))
            per_class[lbl]["total"] += 1
            if r.get("correct"):
                per_class[lbl]["correct"] += 1
        for cls, stats in sorted(per_class.items()):
            c, t = stats["correct"], stats["total"]
            print(f"    [{cls}] {c}/{t} ({c/t:.1%})" if t else f"    [{cls}] 0/0")

    # Error pattern: what was predicted when wrong
    wrong = [r for r in results if not r.get("correct")]
    if wrong:
        from collections import Counter
        predicted_when_wrong = Counter(r.get("pred_label", r.get("pred_answer", "?")) for r in wrong)
        print(f"  Wrong predictions ({len(wrong)} errors): {dict(predicted_when_wrong.most_common(5))}")
