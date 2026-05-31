"""Pass 2: per-mention sentiment polarity via a local Hugging Face model.

Social-media-tuned RoBERTa with a native negative/neutral/positive head
(see hf_sentiment). No pain-point / praise extraction in this mode — that needed an LLM.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from config import MAX_SENTIMENT_CALLS, OUTPUT_DIR
from hf_sentiment import classify
from reddit_fetch import Thread


def _excerpt_around(text: str, term: str, context: int = 320) -> str:
    """Pull a window of `context` chars around the first match, snapped to whitespace."""
    pat = re.compile(re.escape(term), re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return text[: context * 2]
    start = max(0, m.start() - context)
    end = min(len(text), m.end() + context)
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()


def _find_mentions(threads: list[Thread], competitor: dict) -> list[dict]:
    """Build (thread, excerpt) pairs for every thread mentioning the competitor."""
    name = competitor["name"]
    aliases = {name, competitor["normalized"]}
    if name.lower() == "quickbooks time":
        aliases.add("tsheets")
    pat = re.compile("|".join(re.escape(a) for a in aliases), re.IGNORECASE)

    mentions = []
    for t in threads:
        blob = t.text_blob()
        if not pat.search(blob):
            continue
        excerpt = _excerpt_around(blob, name)
        mentions.append({
            "competitor": name,
            "subreddit": t.subreddit,
            "thread_id": t.id,
            "permalink": t.permalink,
            "upvotes": t.score,
            "date": dt.datetime.fromtimestamp(t.created_utc, tz=dt.timezone.utc).strftime("%Y-%m-%d"),
            "excerpt": excerpt,
        })
    return mentions


def classify_all(threads: list[Thread], competitors: list[dict]) -> list[dict]:
    """Build all mentions, sample if needed, score polarity with distilBERT."""
    candidates: list[dict] = []
    for comp in competitors:
        candidates.extend(_find_mentions(threads, comp))

    print(f"  [sentiment] {len(candidates)} candidate mentions before sampling")
    if len(candidates) > MAX_SENTIMENT_CALLS:
        candidates.sort(key=lambda m: m["upvotes"], reverse=True)
        candidates = candidates[:MAX_SENTIMENT_CALLS]
        print(f"  [sentiment] sampling top {MAX_SENTIMENT_CALLS} by upvotes")

    if not candidates:
        return []

    print(f"  [sentiment] scoring {len(candidates)} mentions with local RoBERTa...")
    scores = classify([m["excerpt"] for m in candidates])

    results: list[dict] = []
    for mention, s in zip(candidates, scores):
        results.append({
            **mention,
            "sentiment": s["sentiment"],
            "score": s["score"],
            "p_pos": s["p_pos"],
        })
    print(f"  [sentiment] done ({len(results)} scored)")
    return results


def write_jsonl(mentions: list[dict], path: Path | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = path or (OUTPUT_DIR / "raw_mentions.jsonl")
    with p.open("w") as f:
        for m in mentions:
            # Drop the long excerpt; keep a short quote instead.
            row = {**m, "quote": m["excerpt"][:280].strip()}
            row.pop("excerpt", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p
