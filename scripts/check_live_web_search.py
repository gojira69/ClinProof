"""
Simple smoke test for the DDGS live web retriever.

Examples:
  python3 scripts/check_live_web_search.py
  python3 scripts/check_live_web_search.py --query "aspirin reduces myocardial infarction risk"
  python3 scripts/check_live_web_search.py --region us-en --k 3
"""
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.live_web_search import LiveWebSearchRetriever


DEFAULT_QUERY = "aspirin reduces myocardial infarction risk"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether DDGS live web search is working"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Medical claim or query to test",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="in-en",
        help="DDGS region code (for example: in-en, us-en)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of live web results to request",
    )
    args = parser.parse_args()

    retriever = LiveWebSearchRetriever()

    print("=" * 70)
    print("Live Web Search Smoke Test")
    print(f"Query  : {args.query}")
    print(f"Region : {args.region}")
    print(f"Top-k  : {args.k}")
    print("=" * 70)

    try:
        docs, scores = retriever.retrieve(
            query=args.query,
            k=args.k,
            region=args.region,
        )
    except Exception as e:
        print(f"[FAIL] Live web search raised an error: {e}")
        return 1

    if not docs:
        print("[FAIL] Live web search returned no results.")
        return 2

    print(f"[OK] Retrieved {len(docs)} live web result(s)\n")

    for idx, (doc, score) in enumerate(zip(docs, scores), start=1):
        print(f"Result {idx}")
        print(f"  title          : {doc.get('title', '')}")
        print(f"  url            : {doc.get('url', '')}")
        print(f"  domain         : {doc.get('domain', '')}")
        print(f"  source         : {doc.get('source', '')}")
        print(f"  published_date : {doc.get('published_date', '')}")
        print(f"  retrieved_at   : {doc.get('retrieved_at', '')}")
        print(f"  score          : {score:.4f}")
        print(f"  snippet        : {doc.get('snippet', '')}")
        print("-" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
