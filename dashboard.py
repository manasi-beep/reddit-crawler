"""Build the self-contained dashboard.html from the analysis results.

Computes aggregates, generates the SQL dump string (reusing sql_export), and
injects everything into templates/dashboard.html — producing one offline HTML file
with Download PDF + Download SQL buttons.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

from config import DISCOVERY_MODEL, OUTPUT_DIR, SENTIMENT_HF_MODEL
from sql_export import build_sql_string

ROOT = Path(__file__).parent
TEMPLATE = ROOT / "templates" / "dashboard.html"
VENDOR_HTML2PDF = ROOT / "vendor" / "html2pdf.bundle.min.js"

# Soft palette for competitor monogram avatars (deterministic by name).
_PALETTE = [
    "#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#14b8a6", "#f97316", "#3b82f6", "#84cc16", "#a855f7",
]


def _color_for(name: str) -> str:
    return _PALETTE[sum(ord(c) for c in name) % len(_PALETTE)]


# Max quotes shown per sentiment bucket per competitor (top by upvotes).
_QUOTES_PER_BUCKET = 8
_QUOTE_MAXLEN = 400


def _quote_buckets(mentions: list[dict]) -> dict[str, list[dict]]:
    """Bucket a competitor's mentions into pos/neu/neg, top by upvotes."""
    buckets: dict[str, list[dict]] = {"positive": [], "neutral": [], "negative": []}
    ordered = sorted(
        mentions,
        key=lambda r: (r.get("upvotes", 0) or 0, r.get("score", 0.0) or 0.0),
        reverse=True,
    )
    for x in ordered:
        b = x.get("sentiment", "neutral")
        if b not in buckets:
            b = "neutral"
        if len(buckets[b]) >= _QUOTES_PER_BUCKET:
            continue
        text = " ".join((x.get("quote") or "").split())[:_QUOTE_MAXLEN]
        buckets[b].append({
            "quote": text,
            "sentiment": x.get("sentiment"),
            "score": round(x.get("score", 0.0) or 0.0, 3),
            "upvotes": x.get("upvotes", 0) or 0,
            "subreddit": x.get("subreddit"),
            "permalink": x.get("permalink"),
            "date": x.get("date"),
        })
    return buckets


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _build_data(subreddits_scanned, threads_by_sub, mentions, discovery_cost) -> dict:
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_comp[m["competitor"]].append(m)

    mentions_by_sub: dict[str, Counter] = defaultdict(Counter)
    for m in mentions:
        mentions_by_sub[m["subreddit"]][m["competitor"]] += 1

    competitors = []
    for comp, ms in by_comp.items():
        avg = sum(x["score"] for x in ms) / max(len(ms), 1)
        pos = sum(1 for x in ms if x["sentiment"] == "positive")
        neg = sum(1 for x in ms if x["sentiment"] == "negative")
        neu = len(ms) - pos - neg
        # top subreddit for this competitor
        sub_counter = Counter(x["subreddit"] for x in ms)
        top_sub = sub_counter.most_common(1)[0][0] if sub_counter else None
        competitors.append({
            "name": comp,
            "mentions": len(ms),
            "avg_score": round(avg, 3),
            "pct_negative": round(100 * neg / max(len(ms), 1), 1),
            "pos": pos, "neu": neu, "neg": neg,
            "top_subreddit": top_sub,
            "color": _color_for(comp),
            "quotes": _quote_buckets(ms),
        })
    competitors.sort(key=lambda c: c["mentions"], reverse=True)

    dist = Counter(m["sentiment"] for m in mentions)
    total = max(sum(dist.values()), 1)
    distribution = {
        "positive": dist.get("positive", 0),
        "neutral": dist.get("neutral", 0),
        "negative": dist.get("negative", 0),
        "pct": {
            "positive": round(100 * dist.get("positive", 0) / total),
            "neutral": round(100 * dist.get("neutral", 0) / total),
            "negative": round(100 * dist.get("negative", 0) / total),
        },
    }

    subreddits = []
    for sub in subreddits_scanned:
        n_threads = len(threads_by_sub.get(sub, []))
        n_mentions = sum(mentions_by_sub[sub].values())
        top = mentions_by_sub[sub].most_common(1)
        subreddits.append({
            "subreddit": sub,
            "threads": n_threads,
            "mentions": n_mentions,
            "top_competitor": top[0][0] if top else None,
        })

    return {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "subreddits_scanned": list(subreddits_scanned),
            "threads_analyzed": sum(len(v) for v in threads_by_sub.values()),
            "total_mentions": len(mentions),
            "discovery_cost": discovery_cost,
            "discovery_model": DISCOVERY_MODEL,
            "sentiment_model": SENTIMENT_HF_MODEL,
        },
        "distribution": distribution,
        "competitors": competitors,
        "subreddits": subreddits,
    }


def build_dashboard(
    *,
    subreddits_scanned: list[str],
    threads_by_sub: dict[str, list],
    mentions: list[dict],
    discovery_cost: str = "n/a",
    output_path: Path | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (OUTPUT_DIR / "dashboard.html")

    data = _build_data(subreddits_scanned, threads_by_sub, mentions, discovery_cost)
    sql = build_sql_string(
        subreddits_scanned=subreddits_scanned,
        threads_by_sub=threads_by_sub,
        mentions=mentions,
    )

    html2pdf_js = VENDOR_HTML2PDF.read_text(encoding="utf-8")
    # Prevent a stray "</script>" in the bundle from closing our inline script tag.
    html2pdf_js = html2pdf_js.replace("</script>", "<\\/script>")

    template = TEMPLATE.read_text(encoding="utf-8")
    html = (
        template
        .replace("%%DATA_B64%%", _b64(json.dumps(data, ensure_ascii=False)))
        .replace("%%SQL_B64%%", _b64(sql))
        .replace("%%HTML2PDF_JS%%", html2pdf_js)
    )
    out.write_text(html, encoding="utf-8")
    return out
