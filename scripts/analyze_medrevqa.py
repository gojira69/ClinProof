"""
MedRevQA Results Analyzer
Works on live (mid-eval) checkpoints and completed result files.

Usage:
  python analyze_medrevqa.py results.json                        # single run
  python analyze_medrevqa.py results_new.json results_old.json   # compare vs baseline
"""

import json, sys
from collections import Counter, defaultdict

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"]
LABEL_MAP = {"SUPPORTED": "A", "REFUTED": "B", "NOT ENOUGH INFORMATION": "C"}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
SHORT     = {"SUPPORTED": "SUP", "REFUTED": "REF", "NOT ENOUGH INFORMATION": "NEI"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def pct(n, d):
    return n / d * 100 if d else 0.0

def delta(new, old):
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.1f})"

def classify_context(ctx: str):
    has_kg     = "KG:" in ctx or "Atomic Propositions" in ctx or "Entity:" in ctx
    has_pubmed = "PMID" in ctx or "Abstract" in ctx or "PubMed" in ctx or "[Chunk" in ctx
    has_empty  = len(ctx.strip()) < 80
    if has_empty:             return "empty"
    if has_kg and has_pubmed: return "kg+pubmed"
    if has_kg:                return "kg_only"
    if has_pubmed:            return "pubmed_only"
    return "other"

def vote_type(vd: dict):
    total  = sum(vd.values())
    unique = len(vd)
    if unique == 1:     return "unanimous"
    if unique == total: return "all_split"
    return "majority"

def prop_quality(props: str):
    if not props or props.strip() == "None": return "none"
    lines = [l for l in props.splitlines() if l.strip().startswith("-")]
    if len(lines) == 0:  return "none"
    if len(lines) <= 2:  return "minimal"
    return "rich"

# ─── Stats Builder ───────────────────────────────────────────────────────────

def build_stats(results):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])

    # ── Per-class confusion matrix: true_label → predicted_label → count ──
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        gt_lbl   = r.get("gt_label",   LABEL_INV.get(r.get("gt_answer",   "?"), "?"))
        pred_lbl = r.get("pred_label", LABEL_INV.get(r.get("pred_answer", "?"), "?"))
        confusion[gt_lbl][pred_lbl] += 1

    # Per-class precision / recall / F1
    tp = {l: confusion[l].get(l, 0) for l in LABELS}
    fp = {l: sum(confusion[g].get(l, 0) for g in LABELS if g != l) for l in LABELS}
    fn = {l: sum(confusion[l].get(p, 0) for p in LABELS if p != l) for l in LABELS}

    precision = {l: pct(tp[l], tp[l] + fp[l]) for l in LABELS}
    recall    = {l: pct(tp[l], tp[l] + fn[l]) for l in LABELS}
    f1        = {}
    for l in LABELS:
        p, r2 = precision[l], recall[l]
        f1[l] = 2 * p * r2 / (p + r2) if (p + r2) > 0 else 0.0

    macro_f1 = sum(f1.values()) / len(LABELS)
    macro_p  = sum(precision.values()) / len(LABELS)
    macro_r  = sum(recall.values()) / len(LABELS)

    # Support (gt counts)
    support = {l: sum(confusion[l].values()) for l in LABELS}

    # Context source breakdown
    src_correct = defaultdict(int)
    src_total   = defaultdict(int)
    for r in results:
        src = classify_context(r.get("retrieved_context", ""))
        src_total[src]  += 1
        if r["correct"]: src_correct[src] += 1
    src_acc = {s: pct(src_correct[s], src_total[s]) for s in src_total}

    # KG contribution
    kg_res  = [r for r in results if "KG:" in r.get("retrieved_context", "")]
    no_kg   = [r for r in results if "KG:" not in r.get("retrieved_context", "")]
    kg_acc  = pct(sum(r["correct"] for r in kg_res),  len(kg_res))
    nkg_acc = pct(sum(r["correct"] for r in no_kg),   len(no_kg))

    # Atomic proposition quality
    prop_correct = defaultdict(int)
    prop_total   = defaultdict(int)
    for r in results:
        pq = prop_quality(r.get("atomic_propositions", ""))
        prop_total[pq]  += 1
        if r["correct"]: prop_correct[pq] += 1
    prop_acc = {pq: pct(prop_correct[pq], prop_total[pq]) for pq in prop_total}

    # Vote confidence
    vote_correct = defaultdict(int)
    vote_total   = defaultdict(int)
    for r in results:
        vt = vote_type(r.get("vote_distribution", {}))
        vote_total[vt]  += 1
        if r["correct"]: vote_correct[vt] += 1
    vote_acc = {vt: pct(vote_correct[vt], vote_total[vt]) for vt in vote_total}

    # Per-model vote accuracy
    model_right = defaultdict(int)
    model_seen  = defaultdict(int)
    for r in results:
        gt = r.get("gt_answer", "?")
        for t in r.get("reasoning_traces", []):
            m = t["model"]
            model_seen[m]  += 1
            if t["choice"] == gt: model_right[m] += 1
    model_acc    = {m: pct(model_right[m], model_seen[m]) for m in model_seen}
    model_counts = dict(model_seen)

    # Rescue / betrayal
    rescued  = sum(1 for r in results
                   if r["pred_answer"] == r["gt_answer"]
                   and any(t["choice"] != r["gt_answer"]
                           for t in r.get("reasoning_traces", [])))
    betrayed = sum(1 for r in results
                   if r["pred_answer"] != r["gt_answer"]
                   and any(t["choice"] == r["gt_answer"]
                           for t in r.get("reasoning_traces", [])))

    times = [r["time_seconds"] for r in results if "time_seconds" in r]

    return {
        "total": total, "correct": correct, "acc": pct(correct, total),
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "confusion": {g: dict(row) for g, row in confusion.items()},
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "src_acc": src_acc, "src_total": dict(src_total),
        "kg_count": len(kg_res), "kg_acc": kg_acc,
        "nkg_count": len(no_kg), "nkg_acc": nkg_acc,
        "prop_acc": prop_acc, "prop_total": dict(prop_total),
        "vote_acc": vote_acc, "vote_total": dict(vote_total),
        "model_acc": model_acc, "model_counts": model_counts,
        "rescued": rescued, "betrayed": betrayed,
        "times": times,
        "wrong": [r for r in results if not r["correct"]],
    }

# ─── Printer ─────────────────────────────────────────────────────────────────

def analyze(path, baseline_path=None):
    data    = load(path)
    config  = data.get("config", {})
    results = data["results"]
    s       = build_stats(results)

    complete  = data.get("complete", True)
    progress  = data.get("progress", f"{len(results)}/?")

    has_baseline = baseline_path is not None
    if has_baseline:
        bdata = load(baseline_path)
        b     = build_stats(bdata["results"])

    W = 76
    print(f"\n{'='*W}")
    print(f"  MedRevQA Results Analysis")
    print(f"  FILE     : {path}")
    if not complete:
        print(f"  STATUS   : IN PROGRESS  ({progress} questions done)")
    if has_baseline:
        print(f"  BASELINE : {baseline_path}")
    if config:
        print(f"  Models   : {config.get('models', '?')}")
        print(f"  Votes    : {config.get('votes', '?')}  |  Ensemble: {config.get('ensemble_mode', '?')}")
    print(f"  Total    : {s['total']} questions evaluated")
    print(f"{'='*W}\n")

    # ── 1. Overall metrics ────────────────────────────────────────────────
    print("── 1. OVERALL METRICS")
    if has_baseline:
        print(f"   {'Metric':<14} {'NEW':>10}   {'BASELINE':>10}   {'DELTA':>8}")
        print(f"   {'Accuracy':<14} {s['acc']:>9.1f}%   {b['acc']:>9.1f}%   {delta(s['acc'], b['acc']):>8}")
        print(f"   {'Macro-P':<14} {s['macro_p']:>9.1f}%   {b['macro_p']:>9.1f}%   {delta(s['macro_p'], b['macro_p']):>8}")
        print(f"   {'Macro-R':<14} {s['macro_r']:>9.1f}%   {b['macro_r']:>9.1f}%   {delta(s['macro_r'], b['macro_r']):>8}")
        print(f"   {'Macro-F1':<14} {s['macro_f1']:>9.1f}%   {b['macro_f1']:>9.1f}%   {delta(s['macro_f1'], b['macro_f1']):>8}")
    else:
        print(f"   Accuracy  : {s['correct']}/{s['total']}  =  {s['acc']:.1f}%")
        print(f"   Macro-P   : {s['macro_p']:.1f}%")
        print(f"   Macro-R   : {s['macro_r']:.1f}%")
        print(f"   Macro-F1  : {s['macro_f1']:.1f}%")
    print()

    # ── 2. Per-class precision / recall / F1 ──────────────────────────────
    print("── 2. PER-CLASS METRICS  (Precision / Recall / F1)")
    hdr = f"   {'Class':<26}  {'Support':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    if has_baseline:
        hdr += f"  |  {'Baseline F1':>10}  {'ΔF1':>7}"
    print(hdr)
    print("   " + "─" * (len(hdr) - 3))
    for l in LABELS:
        row = (f"   {l:<26}  {s['support'].get(l, 0):>7}"
               f"  {s['precision'].get(l, 0):>5.1f}%"
               f"  {s['recall'].get(l, 0):>5.1f}%"
               f"  {s['f1'].get(l, 0):>5.1f}%")
        if has_baseline:
            row += (f"  |  {b['f1'].get(l, 0):>9.1f}%"
                    f"  {delta(s['f1'].get(l, 0), b['f1'].get(l, 0)):>7}")
        print(row)
    print()

    # ── 3. Confusion Matrix ───────────────────────────────────────────────
    print("── 3. CONFUSION MATRIX  (rows=Ground Truth, cols=Predicted)")
    col_w = 7
    header_cols = [SHORT[l] for l in LABELS]
    gt_pred = "GT \\ Pred"
    print("   " + f"{gt_pred:<26}" + "".join(f"{c:>{col_w}}" for c in header_cols))
    print("   " + "─" * (26 + col_w * len(LABELS)))
    for gt in LABELS:
        row_str = f"   {gt:<26}"
        for pred in LABELS:
            cnt = s["confusion"].get(gt, {}).get(pred, 0)
            row_str += f"{cnt:>{col_w}}"
        print(row_str)
    print()

    # ── 4. Context source breakdown ───────────────────────────────────────
    print("── 4. CONTEXT SOURCE BREAKDOWN")
    all_srcs = sorted(set(list(s['src_total'].keys()) +
                          (list(b['src_total'].keys()) if has_baseline else [])))
    for src in all_srcs:
        n_acc = s['src_acc'].get(src, 0.0)
        n_cnt = s['src_total'].get(src, 0)
        bar   = "█" * int(n_acc / 5)
        if has_baseline:
            b_acc2 = b['src_acc'].get(src, 0.0)
            b_cnt  = b['src_total'].get(src, 0)
            print(f"   {src:<15}  NEW {n_cnt:>4}q {n_acc:5.1f}%   "
                  f"BASELINE {b_cnt:>4}q {b_acc2:5.1f}%   {delta(n_acc, b_acc2)}  {bar}")
        else:
            print(f"   {src:<15}  {n_cnt:>4} questions  |  acc {n_acc:5.1f}%  {bar}")
    print()

    # ── 5. KG contribution ────────────────────────────────────────────────
    print("── 5. KG CONTRIBUTION")
    if has_baseline:
        print(f"   {'':25}  {'NEW':>10}   {'BASELINE':>10}   {'DELTA':>8}")
        print(f"   {'WITH KG  (count)':<25}  {s['kg_count']:>10}   {b['kg_count']:>10}")
        print(f"   {'WITH KG  (acc)':<25}  {s['kg_acc']:>9.1f}%   {b['kg_acc']:>9.1f}%   {delta(s['kg_acc'], b['kg_acc']):>8}")
        print(f"   {'WITHOUT KG (count)':<25}  {s['nkg_count']:>10}   {b['nkg_count']:>10}")
        print(f"   {'WITHOUT KG (acc)':<25}  {s['nkg_acc']:>9.1f}%   {b['nkg_acc']:>9.1f}%   {delta(s['nkg_acc'], b['nkg_acc']):>8}")
    else:
        print(f"   WITH KG     {s['kg_count']:>4} questions  |  acc {s['kg_acc']:.1f}%")
        print(f"   WITHOUT KG  {s['nkg_count']:>4} questions  |  acc {s['nkg_acc']:.1f}%")
        print(f"   KG lift: {s['kg_acc'] - s['nkg_acc']:+.1f}%")
    print()

    # ── 6. Atomic proposition quality ─────────────────────────────────────
    print("── 6. ATOMIC PROPOSITION QUALITY")
    for pq in ["rich", "minimal", "none"]:
        n_cnt = s['prop_total'].get(pq, 0)
        b_cnt = b['prop_total'].get(pq, 0) if has_baseline else 0
        if not n_cnt and not b_cnt:
            continue
        n_acc = s['prop_acc'].get(pq, 0.0)
        if has_baseline:
            b_acc2 = b['prop_acc'].get(pq, 0.0)
            print(f"   {pq:<10}  NEW {n_cnt:>4}q {n_acc:5.1f}%   "
                  f"BASELINE {b_cnt:>4}q {b_acc2:5.1f}%   {delta(n_acc, b_acc2)}")
        else:
            print(f"   {pq:<10}  {n_cnt:>4} questions  |  acc {n_acc:.1f}%")
    print()

    # ── 7. Vote confidence ────────────────────────────────────────────────
    print("── 7. VOTE CONFIDENCE vs ACCURACY")
    for vt in ["unanimous", "majority", "all_split"]:
        n_cnt = s['vote_total'].get(vt, 0)
        if not n_cnt:
            continue
        n_acc = s['vote_acc'].get(vt, 0.0)
        if has_baseline:
            b_acc2 = b['vote_acc'].get(vt, 0.0)
            b_cnt  = b['vote_total'].get(vt, 0)
            print(f"   {vt:<12}  NEW {n_cnt:>4}q {n_acc:5.1f}%   "
                  f"BASELINE {b_cnt:>4}q {b_acc2:5.1f}%   {delta(n_acc, b_acc2)}")
        else:
            print(f"   {vt:<12}  {n_cnt:>4} questions  |  acc {n_acc:.1f}%")
    print()

    # ── 8. Per-model vote accuracy ────────────────────────────────────────
    print("── 8. PER-MODEL VOTE ACCURACY")
    all_models = sorted(set(list(s['model_acc'].keys()) +
                            (list(b['model_acc'].keys()) if has_baseline else [])))
    if all_models:
        for m in all_models:
            n_acc = s['model_acc'].get(m, 0.0)
            n_cnt = s['model_counts'].get(m, 0)
            if has_baseline:
                b_acc2 = b['model_acc'].get(m, 0.0)
                b_cnt  = b['model_counts'].get(m, 0)
                print(f"   {m:<22}  NEW {n_cnt:>4} votes {n_acc:5.1f}%   "
                      f"BASELINE {b_cnt:>4} votes {b_acc2:5.1f}%   {delta(n_acc, b_acc2)}")
            else:
                print(f"   {m:<22}  {n_cnt} votes  =  {n_acc:.1f}%")
    else:
        print("   (no reasoning trace data)")
    print()

    # ── 9. Voting dynamics ────────────────────────────────────────────────
    print("── 9. VOTING DYNAMICS")
    if has_baseline:
        print(f"   {'Rescued  (final=correct, ≥1 wrong vote)':<42}  "
              f"NEW {s['rescued']:>4}   BASELINE {b['rescued']:>4}   ({s['rescued']-b['rescued']:+d})")
        print(f"   {'Betrayed (final=wrong,   ≥1 right vote)':<42}  "
              f"NEW {s['betrayed']:>4}   BASELINE {b['betrayed']:>4}   ({s['betrayed']-b['betrayed']:+d})")
    else:
        print(f"   Rescued : {s['rescued']}")
        print(f"   Betrayed: {s['betrayed']}")
    print()

    # ── 10. Timing ────────────────────────────────────────────────────────
    times = s['times']
    if times:
        avg = sum(times) / len(times)
        print("── 10. TIMING")
        if has_baseline and b['times']:
            b_avg = sum(b['times']) / len(b['times'])
            print(f"   avg {avg:.1f}s {delta(avg, b_avg)}  |  min {min(times):.1f}s  |  max {max(times):.1f}s")
        else:
            print(f"   avg {avg:.1f}s  |  min {min(times):.1f}s  |  max {max(times):.1f}s")
        slow = sorted(results, key=lambda r: r.get("time_seconds", 0), reverse=True)[:3]
        for r in slow:
            print(f"     {r['time_seconds']:.1f}s  —  {r['question'][:65]}")
    print()

    # ── 11. Failure analysis ──────────────────────────────────────────────
    wrong = s['wrong']
    print(f"── 11. FAILURE ANALYSIS  ({len(wrong)} wrong)")

    # Error pattern: gt → pred breakdown for wrong answers
    err_pattern = Counter(
        (r.get("gt_label", "?"), r.get("pred_label", "?"))
        for r in wrong
    )
    print("   Most common confusion pairs (gt → pred):")
    for (gt_l, pred_l), cnt in err_pattern.most_common(6):
        print(f"     {gt_l:<26} → {pred_l:<26}  {cnt}")

    wrong_src  = Counter(classify_context(r.get("retrieved_context", "")) for r in wrong)
    wrong_vote = Counter(vote_type(r.get("vote_distribution", {})) for r in wrong)
    if has_baseline:
        b_wrong      = b['wrong']
        b_wrong_src  = Counter(classify_context(r.get("retrieved_context", "")) for r in b_wrong)
        b_wrong_vote = Counter(vote_type(r.get("vote_distribution", {})) for r in b_wrong)
        print(f"   Total wrong:  NEW {len(wrong)}   BASELINE {len(b_wrong)}   ({len(wrong)-len(b_wrong):+d})")
        print("   By context source:")
        for src in sorted(set(list(wrong_src.keys()) + list(b_wrong_src.keys()))):
            print(f"     {src:<15}  NEW {wrong_src.get(src,0):>3}   "
                  f"BASELINE {b_wrong_src.get(src,0):>3}   ({wrong_src.get(src,0)-b_wrong_src.get(src,0):+d})")
        print("   By vote confidence:")
        for vt in ["unanimous", "majority", "all_split"]:
            print(f"     {vt:<12}  NEW {wrong_vote.get(vt,0):>3}   "
                  f"BASELINE {b_wrong_vote.get(vt,0):>3}   ({wrong_vote.get(vt,0)-b_wrong_vote.get(vt,0):+d})")
    else:
        print(f"   By context: {dict(wrong_src)}")
        print(f"   By vote   : {dict(wrong_vote)}")

    print("\n   Sample wrong answers:")
    for r in wrong[:3]:
        vd = r.get("vote_distribution", {})
        print(f"   ─ [PMID:{r.get('pmid','?')[:10]}] {r['question'][:60]}")
        print(f"     gt={r.get('gt_label','?')}  pred={r.get('pred_label','?')}  votes={vd}")
        traces = r.get("reasoning_traces", [])
        if traces:
            raw = traces[0].get("step_by_step_thinking", "")
            thinking = (raw if isinstance(raw, str) else json.dumps(raw))[:140].replace("\n", " ")
            print(f"     reasoning: {thinking}...")
        print()

    print(f"{'='*W}\n")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_medrevqa.py results.json [baseline.json]")
        sys.exit(1)
    new_path      = sys.argv[1]
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(new_path, baseline_path)
