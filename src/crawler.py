"""TimeTracer competitor sentiment crawler — main entrypoint.

Pipeline:
  1. Fetch     — pull relevant threads from each subreddit (cached on disk)
  2. Discover  — Claude extracts competitor product names (needs ANTHROPIC_API_KEY)
  3. Sentiment — distilBERT (local) scores polarity per mention; render Markdown + SQL

Usage:
    python crawler.py                 # full run (all configured subreddits)
    python crawler.py --limit 3       # only first 3 subreddits
    python crawler.py --dry-run       # fetch + discovery only, print competitors + cost estimate
    python crawler.py --no-cache      # bypass on-disk Reddit cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from config import OUTPUT_DIR, SUBREDDITS
from dashboard import build_dashboard
from reddit_fetch import gather_all


def parse_args():
    ap = argparse.ArgumentParser(description="TimeTracer Reddit competitor crawler")
    ap.add_argument("--limit", type=int, default=None, help="Cap subreddits processed")
    ap.add_argument("--dry-run", action="store_true",
                    help="Stop after discovery; print competitors + estimated cost")
    ap.add_argument("--no-cache", action="store_true", help="Bypass on-disk Reddit cache")
    return ap.parse_args()


def run(subreddits, threads_by_sub, all_threads, args) -> int:
    from discovery import discover_competitors
    from llm import COST
    from sentiment import classify_all, write_jsonl

    print()
    print(f"== Stage 2/3: Discovery (extracting competitor names from {len(all_threads)} threads) ==")
    competitors, debug = discover_competitors(all_threads)
    print(f"-> {debug['competitors_after_threshold']} competitors above mention threshold "
          f"(from {debug['total_unique_names_seen']} unique names; "
          f"failed batches: {debug['failed_batches']})")
    print("Top 15:")
    for c in competitors[:15]:
        print(f"  {c['mention_count']:>3}  {c['name']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "competitors.json").write_text(json.dumps(competitors, indent=2))

    if args.dry_run:
        print()
        print(f"[dry-run] discovery cost: {COST.summary()}")
        print("[dry-run] sentiment runs locally with RoBERTa (no LLM cost).")
        return 0

    print()
    print(f"== Stage 3/3: Per-mention sentiment via local RoBERTa ({len(competitors)} competitors) ==")
    mentions = classify_all(all_threads, competitors)
    write_jsonl(mentions)
    print(f"-> wrote {len(mentions)} mentions to {OUTPUT_DIR / 'raw_mentions.jsonl'}")

    dash_path = build_dashboard(
        subreddits_scanned=subreddits,
        threads_by_sub=threads_by_sub,
        mentions=mentions,
        discovery_cost=COST.summary(),
    )
    print(f"-> wrote dashboard to {dash_path}")
    print("   open it in a browser; use its Download PDF / Download SQL buttons to export.")
    print()
    print(f"Discovery LLM totals: {COST.summary()}  |  Sentiment: local RoBERTa (free)")
    return 0


def main() -> int:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY not found.\n"
            "  Discovery (finding competitor names) uses Claude. Sentiment is local (distilBERT).\n"
            "  Add your key to reddit-crawler/.env:\n"
            "    echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env",
            file=sys.stderr,
        )
        return 2

    subreddits = [name for name, _tier in SUBREDDITS]
    if args.limit:
        subreddits = subreddits[: args.limit]

    print(f"== Stage 1/3: Fetching threads from {len(subreddits)} subreddit(s) ==")
    threads_by_sub = gather_all(subreddits, use_cache=not args.no_cache)
    all_threads = [t for ts in threads_by_sub.values() for t in ts]
    print(f"-> total relevant threads: {len(all_threads)}")
    if not all_threads:
        print("No threads found. Exiting.")
        return 1

    return run(subreddits, threads_by_sub, all_threads, args)


if __name__ == "__main__":
    sys.exit(main())
