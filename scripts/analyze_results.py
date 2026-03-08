"""
ClinProof Results Analyzer
Usage:
  python analyze_results.py results.json                        # single run
  python analyze_results.py results_new.json results_old.json   # compare vs baseline
"""
import json, sys, re
from collections import Counter, defaultdict

def load(path):
    with open(path) as f:
        return json.load(f)

def classify_context(ctx: str):
    has_kg     = "KG:" in ctx or "Atomic Propositions" in ctx or "Entity:" in ctx
    has_pubmed = "PMID" in ctx or "Abstract" in ctx or "PubMed" in ctx or "[Chunk" in ctx
    has_empty  = len(ctx.strip()) < 80
    if has_empty:             return "empty"
    if has_kg and has_pubmed: return "kg+pubmed"
    if has_kg:                return "kg_only"
    if has_pubmed:            return "pubmed_only"
    return "other"

def context_length(ctx: str):
    return len(ctx)

def vote_type(vd: dict):
    total  = sum(vd.values())
    unique = len(vd)
    if unique == 1:      return "unanimous"
    if unique == total:  return "all_split"
    return "majority"

def prop_quality(props: str):
    if not props or props.strip() == "None": return "none"
    lines = [l for l in props.splitlines() if l.strip().startswith("-")]
    if len(lines) == 0:  return "none"
    if len(lines) <= 2:  return "minimal"
    return "rich"

def delta(new, old):
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.1f})"

def pct(n, d):
    return n / d * 100 if d else 0.0

# ─── Per-run stats builder ────────────────────────────────────────────────────

def build_stats(results):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])

    src_correct = defaultdict(int)
    src_total   = defaultdict(int)
    for r in results:
        src = classify_context(r.get("retrieved_context", ""))
        src_total[src]  += 1
        if r["correct"]: src_correct[src] += 1
    src_acc = {s: pct(src_correct[s], src_total[s]) for s in src_total}

    kg_res  = [r for r in results if "KG:" in r.get("retrieved_context","")]
    no_kg   = [r for r in results if "KG:" not in r.get("retrieved_context","")]
    kg_acc  = pct(sum(r["correct"] for r in kg_res),  len(kg_res))
    nkg_acc = pct(sum(r["correct"] for r in no_kg),   len(no_kg))
    avg_ctx_kg  = sum(context_length(r["retrieved_context"]) for r in kg_res)  / max(len(kg_res), 1)
    avg_ctx_nkg = sum(context_length(r["retrieved_context"]) for r in no_kg)   / max(len(no_kg), 1)

    prop_correct = defaultdict(int)
    prop_total   = defaultdict(int)
    for r in results:
        pq = prop_quality(r.get("atomic_propositions",""))
        prop_total[pq]  += 1
        if r["correct"]: prop_correct[pq] += 1
    prop_acc = {pq: pct(prop_correct[pq], prop_total[pq]) for pq in prop_total}

    vote_correct = defaultdict(int)
    vote_total   = defaultdict(int)
    for r in results:
        vt = vote_type(r.get("vote_distribution", {}))
        vote_total[vt]  += 1
        if r["correct"]: vote_correct[vt] += 1
    vote_acc = {vt: pct(vote_correct[vt], vote_total[vt]) for vt in vote_total}

    model_right = defaultdict(int)
    model_seen  = defaultdict(int)
    for r in results:
        gt = r["gt_answer"]
        for t in r.get("reasoning_traces", []):
            m = t["model"]
            model_seen[m]  += 1
            if t["choice"] == gt: model_right[m] += 1
    model_acc    = {m: pct(model_right[m], model_seen[m]) for m in model_seen}
    model_counts = dict(model_seen)

    rescued  = sum(1 for r in results
                   if r["pred_answer"] == r["gt_answer"]
                   and any(t["choice"] != r["gt_answer"] for t in r.get("reasoning_traces",[])))
    betrayed = sum(1 for r in results
                   if r["pred_answer"] != r["gt_answer"]
                   and any(t["choice"] == r["gt_answer"] for t in r.get("reasoning_traces",[])))

    times = [r["time_seconds"] for r in results if "time_seconds" in r]

    return {
        "total": total, "correct": correct, "acc": pct(correct, total),
        "src_acc": src_acc, "src_total": dict(src_total),
        "kg_count": len(kg_res), "kg_acc": kg_acc, "avg_ctx_kg": avg_ctx_kg,
        "nkg_count": len(no_kg), "nkg_acc": nkg_acc, "avg_ctx_nkg": avg_ctx_nkg,
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
    config  = data["config"]
    results = data["results"]
    s       = build_stats(results)

    has_baseline = baseline_path is not None
    if has_baseline:
        bdata = load(baseline_path)
        b     = build_stats(bdata["results"])

    W = 72
    print(f"\n{'='*W}")
    print(f"  ClinProof Results Analysis")
    print(f"  NEW      : {path}")
    if has_baseline:
        print(f"  BASELINE : {baseline_path}")
    print(f"  Models   : {config['models']}")
    print(f"  Votes    : {config['votes']}  |  Ensemble: {config['ensemble_mode']}")
    print(f"  Total    : {s['total']} questions")
    print(f"{'='*W}\n")

    # ── 1. Overall accuracy ───────────────────────────────────────────────
    print(f"── 1. OVERALL ACCURACY")
    if has_baseline:
        print(f"   NEW      {s['correct']}/{s['total']}  =  {s['acc']:.1f}%  {delta(s['acc'], b['acc'])}")
        print(f"   BASELINE {b['correct']}/{b['total']}  =  {b['acc']:.1f}%")
    else:
        print(f"   {s['correct']}/{s['total']}  =  {s['acc']:.1f}%")
    print()

    # ── 2. Context source breakdown ───────────────────────────────────────
    print(f"── 2. CONTEXT SOURCE BREAKDOWN")
    all_srcs = sorted(set(list(s['src_total'].keys()) + (list(b['src_total'].keys()) if has_baseline else [])))
    for src in all_srcs:
        n_acc = s['src_acc'].get(src, 0.0)
        n_cnt = s['src_total'].get(src, 0)
        bar   = "█" * int(n_acc / 5)
        if has_baseline:
            b_acc = b['src_acc'].get(src, 0.0)
            b_cnt = b['src_total'].get(src, 0)
            print(f"   {src:<15}  NEW {n_cnt:>4}q {n_acc:5.1f}%   BASELINE {b_cnt:>4}q {b_acc:5.1f}%   {delta(n_acc,b_acc)}  {bar}")
        else:
            print(f"   {src:<15}  {n_cnt:>4} questions  |  acc {n_acc:5.1f}%  {bar}")
    print()

    # ── 3. KG contribution ────────────────────────────────────────────────
    print(f"── 3. KG CONTRIBUTION")
    if has_baseline:
        print(f"   {'':25}  {'NEW':>10}   {'BASELINE':>10}   {'DELTA':>8}")
        print(f"   {'WITH KG  (count)':<25}  {s['kg_count']:>10}   {b['kg_count']:>10}")
        print(f"   {'WITH KG  (acc)':<25}  {s['kg_acc']:>9.1f}%   {b['kg_acc']:>9.1f}%   {delta(s['kg_acc'],b['kg_acc']):>8}")
        print(f"   {'WITHOUT KG (count)':<25}  {s['nkg_count']:>10}   {b['nkg_count']:>10}")
        print(f"   {'WITHOUT KG (acc)':<25}  {s['nkg_acc']:>9.1f}%   {b['nkg_acc']:>9.1f}%   {delta(s['nkg_acc'],b['nkg_acc']):>8}")
        print(f"   {'KG lift':<25}  {s['kg_acc']-s['nkg_acc']:>+9.1f}%   {b['kg_acc']-b['nkg_acc']:>+9.1f}%   {delta(s['kg_acc']-s['nkg_acc'], b['kg_acc']-b['nkg_acc']):>8}")
        print(f"   {'avg ctx WITH KG':<25}  {s['avg_ctx_kg']:>9.0f}c   {b['avg_ctx_kg']:>9.0f}c")
        print(f"   {'avg ctx WITHOUT KG':<25}  {s['avg_ctx_nkg']:>9.0f}c   {b['avg_ctx_nkg']:>9.0f}c")
    else:
        print(f"   WITH KG     {s['kg_count']:>4} questions  |  acc {s['kg_acc']:.1f}%  |  avg ctx {s['avg_ctx_kg']:.0f} chars")
        print(f"   WITHOUT KG  {s['nkg_count']:>4} questions  |  acc {s['nkg_acc']:.1f}%  |  avg ctx {s['avg_ctx_nkg']:.0f} chars")
        print(f"   KG lift: {s['kg_acc']-s['nkg_acc']:+.1f}%")
    print()

    # ── 4. Atomic proposition quality ─────────────────────────────────────
    print(f"── 4. ATOMIC PROPOSITION QUALITY")
    for pq in ["rich", "minimal", "none"]:
        n_acc = s['prop_acc'].get(pq, 0.0)
        n_cnt = s['prop_total'].get(pq, 0)
        if not n_cnt and (not has_baseline or not b['prop_total'].get(pq, 0)):
            continue
        if has_baseline:
            b_acc = b['prop_acc'].get(pq, 0.0)
            b_cnt = b['prop_total'].get(pq, 0)
            print(f"   {pq:<10}  NEW {n_cnt:>4}q {n_acc:5.1f}%   BASELINE {b_cnt:>4}q {b_acc:5.1f}%   {delta(n_acc,b_acc)}")
        else:
            print(f"   {pq:<10}  {n_cnt:>4} questions  |  acc {n_acc:.1f}%")
    print()

    # ── 5. Vote confidence ────────────────────────────────────────────────
    print(f"── 5. VOTE CONFIDENCE vs ACCURACY")
    for vt in ["unanimous", "majority", "all_split"]:
        n_acc = s['vote_acc'].get(vt, 0.0)
        n_cnt = s['vote_total'].get(vt, 0)
        if not n_cnt: continue
        if has_baseline:
            b_acc = b['vote_acc'].get(vt, 0.0)
            b_cnt = b['vote_total'].get(vt, 0)
            print(f"   {vt:<12}  NEW {n_cnt:>4}q {n_acc:5.1f}%   BASELINE {b_cnt:>4}q {b_acc:5.1f}%   {delta(n_acc,b_acc)}")
        else:
            print(f"   {vt:<12}  {n_cnt:>4} questions  |  acc {n_acc:.1f}%")
    print()

    # ── 6. Per-model vote accuracy ────────────────────────────────────────
    print(f"── 6. PER-MODEL VOTE ACCURACY")
    all_models = sorted(set(list(s['model_acc'].keys()) + (list(b['model_acc'].keys()) if has_baseline else [])))
    for m in all_models:
        n_acc = s['model_acc'].get(m, 0.0)
        n_cnt = s['model_counts'].get(m, 0)
        if has_baseline:
            b_acc = b['model_acc'].get(m, 0.0)
            b_cnt = b['model_counts'].get(m, 0)
            print(f"   {m:<22}  NEW {n_cnt:>4} votes {n_acc:5.1f}%   BASELINE {b_cnt:>4} votes {b_acc:5.1f}%   {delta(n_acc,b_acc)}")
        else:
            print(f"   {m:<22}  {n_cnt} votes  =  {n_acc:.1f}%")
    print()

    # ── 7. Voting dynamics ────────────────────────────────────────────────
    print(f"── 7. VOTING DYNAMICS")
    if has_baseline:
        print(f"   {'Rescued  (final=correct, ≥1 wrong vote)':<42}  NEW {s['rescued']:>4}   BASELINE {b['rescued']:>4}   ({s['rescued']-b['rescued']:+d})")
        print(f"   {'Betrayed (final=wrong,   ≥1 right vote)':<42}  NEW {s['betrayed']:>4}   BASELINE {b['betrayed']:>4}   ({s['betrayed']-b['betrayed']:+d})")
    else:
        print(f"   Rescued : {s['rescued']}")
        print(f"   Betrayed: {s['betrayed']}")
    print()

    # ── 8. Timing ─────────────────────────────────────────────────────────
    times = s['times']
    if times:
        avg = sum(times)/len(times)
        print(f"── 8. TIMING")
        if has_baseline and b['times']:
            b_avg = sum(b['times'])/len(b['times'])
            print(f"   avg {avg:.1f}s {delta(avg,b_avg)}  |  min {min(times):.1f}s  |  max {max(times):.1f}s")
        else:
            print(f"   avg {avg:.1f}s  |  min {min(times):.1f}s  |  max {max(times):.1f}s")
        slow = sorted(results, key=lambda r: r.get("time_seconds",0), reverse=True)[:3]
        for r in slow:
            print(f"     {r['time_seconds']:.1f}s  —  {r['question'][:60]}")
    print()

    # ── 9. Failure analysis ───────────────────────────────────────────────
    wrong = s['wrong']
    print(f"── 9. FAILURE ANALYSIS  ({len(wrong)} wrong)")
    wrong_src  = Counter(classify_context(r.get("retrieved_context","")) for r in wrong)
    wrong_vote = Counter(vote_type(r.get("vote_distribution",{})) for r in wrong)
    if has_baseline:
        b_wrong      = b['wrong']
        b_wrong_src  = Counter(classify_context(r.get("retrieved_context","")) for r in b_wrong)
        b_wrong_vote = Counter(vote_type(r.get("vote_distribution",{})) for r in b_wrong)
        print(f"   Total wrong:  NEW {len(wrong)}   BASELINE {len(b_wrong)}   ({len(wrong)-len(b_wrong):+d})")
        print(f"   By context source:")
        for src in sorted(set(list(wrong_src.keys()) + list(b_wrong_src.keys()))):
            print(f"     {src:<15}  NEW {wrong_src.get(src,0):>3}   BASELINE {b_wrong_src.get(src,0):>3}   ({wrong_src.get(src,0)-b_wrong_src.get(src,0):+d})")
        print(f"   By vote confidence:")
        for vt in ["unanimous","majority","all_split"]:
            print(f"     {vt:<12}  NEW {wrong_vote.get(vt,0):>3}   BASELINE {b_wrong_vote.get(vt,0):>3}   ({wrong_vote.get(vt,0)-b_wrong_vote.get(vt,0):+d})")
    else:
        print(f"   By context: {dict(wrong_src)}")
        print(f"   By vote   : {dict(wrong_vote)}")

    print(f"\n   Sample wrong answers (NEW):")
    for r in wrong[:3]:
        vd = r.get("vote_distribution", {})
        print(f"   ─ [{r['id'][:8]}] {r['question'][:55]}")
        print(f"     gt={r['gt_answer']}  pred={r['pred_answer']}  votes={vd}")
        traces = r.get("reasoning_traces", [])
        if traces:
            raw = traces[0].get("step_by_step_thinking", "")
            thinking = (raw if isinstance(raw, str) else json.dumps(raw))[:120].replace("\n", " ")
            print(f"     reasoning: {thinking}...")
        print()

    print(f"{'='*W}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py new.json [baseline.json]")
        sys.exit(1)
    new_path      = sys.argv[1]
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(new_path, baseline_path)