"""
Inspect results where context source classified as 'other'.
Usage: python inspect_other.py results.json
Outputs: other_results.json  +  a readable summary to stdout
"""
import json, sys, re
from collections import Counter

def classify_context(ctx: str):
    has_kg     = "KG:" in ctx or "Atomic Propositions" in ctx or "Entity:" in ctx
    has_pubmed = "PMID" in ctx or "Abstract" in ctx or "PubMed" in ctx
    has_empty  = len(ctx.strip()) < 80
    if has_empty:             return "empty"
    if has_kg and has_pubmed: return "kg+pubmed"
    if has_kg:                return "kg_only"
    if has_pubmed:            return "pubmed_only"
    return "other"

def sniff_context(ctx: str):
    """Try to figure out what 'other' actually contains."""
    ctx_lower = ctx.lower()
    hints = []
    if "document [" in ctx_lower:           hints.append("has_doc_headers")
    if "title:" in ctx_lower:               hints.append("has_title_field")
    if re.search(r'\bpmid\b', ctx_lower):   hints.append("has_pmid")
    if "abstract" in ctx_lower:             hints.append("has_abstract")
    if "pubmed" in ctx_lower:               hints.append("has_pubmed_mention")
    if "http" in ctx_lower:                 hints.append("has_url")
    if re.search(r'\d{4};\d+', ctx_lower):  hints.append("has_citation_format")
    if "mesh" in ctx_lower:                 hints.append("has_mesh")
    if not hints:                           hints.append("unknown_format")
    return hints

def first_n_chars(ctx, n=300):
    return ctx.strip()[:n].replace("\n", " ↵ ")

def analyze(path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    other   = [r for r in results if classify_context(r.get("retrieved_context","")) == "other"]

    print(f"\n{'='*70}")
    print(f"  'other' context inspector")
    print(f"  File  : {path}")
    print(f"  Total : {len(results)} results  →  {len(other)} are 'other'")
    print(f"{'='*70}\n")

    if not other:
        print("No 'other' results found.")
        return

    # ── What do the contexts actually look like? ──────────────────────────
    hint_counts = Counter()
    for r in other:
        for h in sniff_context(r.get("retrieved_context","")):
            hint_counts[h] += 1

    print("── WHAT IS IN THE 'OTHER' CONTEXTS?")
    for hint, count in hint_counts.most_common():
        print(f"   {hint:<30}  {count}/{len(other)}")
    print()

    # ── Accuracy breakdown ────────────────────────────────────────────────
    correct = [r for r in other if r["correct"]]
    wrong   = [r for r in other if not r["correct"]]
    print(f"── ACCURACY")
    print(f"   correct : {len(correct)}/{len(other)}  =  {len(correct)/len(other)*100:.1f}%")
    print(f"   wrong   : {len(wrong)}/{len(other)}")
    print()

    # ── Context length distribution ───────────────────────────────────────
    lengths = [len(r.get("retrieved_context","")) for r in other]
    print(f"── CONTEXT LENGTH")
    print(f"   min {min(lengths)}  |  avg {sum(lengths)//len(lengths)}  |  max {max(lengths)} chars")
    print()

    # ── Show every result with its context preview ────────────────────────
    print(f"── ALL 'OTHER' RESULTS  ({len(other)} total)\n")
    for i, r in enumerate(other):
        ctx   = r.get("retrieved_context","")
        hints = sniff_context(ctx)
        ok    = "✅" if r["correct"] else "❌"
        vd    = r.get("vote_distribution", {})

        print(f"  [{i+1:>3}] {ok}  id={r['id'][:8]}  gt={r['gt_answer']}  pred={r['pred_answer']}  votes={vd}")
        print(f"        Q: {r['question'][:70]}")
        print(f"        hints : {hints}")
        print(f"        ctx   : {first_n_chars(ctx, 300)}")
        print()

    # ── Save filtered results to JSON ─────────────────────────────────────
    out_path = path.replace(".json", "_other_only.json")
    with open(out_path, "w") as f:
        json.dump({
            "source_file": path,
            "total_other": len(other),
            "accuracy": len(correct) / len(other),
            "hint_counts": dict(hint_counts),
            "results": other
        }, f, indent=2)

    print(f"── SAVED")
    print(f"   {out_path}  ({len(other)} records)\n")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_other.py results.json")
        sys.exit(1)
    analyze(sys.argv[1])