"""Pass 1: discover competitor product names mentioned in threads."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from config import (
    DISCOVERY_BATCH_SIZE,
    DISCOVERY_DENYLIST,
    DISCOVERY_MODEL,
    MIN_MENTIONS_FOR_COMPETITOR,
)
from llm import MAX_WORKERS, call_with_cached_system, extract_json
from reddit_fetch import Thread

DISCOVERY_SYSTEM = """You analyze Reddit threads to find DIRECT COMPETITORS of a time-tracking and hourly-billing product for freelancers/contractors.

WHAT COUNTS (extract these):
- Dedicated time trackers (e.g. Harvest, Toggl, Clockify, RescueTime, Timely, Hubstaff, TimeCamp, Everhour, My Hours, Jibble, Timeular).
- Time-tracking + invoicing tools (e.g. FreshBooks, QuickBooks Time, Bonsai, Paymo, HoneyBook).
- Invoicing / billing software (e.g. Wave, Xero, Invoice Ninja, Zoho Invoice, QuickBooks).
- DIY tracking/billing baselines: ALWAYS extract Excel, Google Sheets, Notion, and "pen and paper" / spreadsheets whenever a freelancer/contractor mentions using them to track time, log hours, manage invoices, or handle their freelance billing. These are important "I just use a spreadsheet instead of paying for a tool" competitors — do not skip them.

WHAT TO EXCLUDE (do NOT extract — these are NOT competitors):
- Freelance MARKETPLACES: Upwork, Fiverr, Freelancer.com, Toptal, Guru, PeoplePerHour, Contra.
- PAYMENT PROCESSORS / money transfer: Stripe, PayPal, Wise, Payoneer, Venmo, Square, Revolut, Mercury, Gumroad.
- Design / content / comms / storage / generic productivity: Canva, Loom, Figma, Photoshop, Google Docs, Dropbox, Slack, Zoom, Calendly, Trello, Asana, Discord, ChatGPT, GitHub.
- Generic categories ("time tracker", "the app"), company names that aren't products, person names.

The product MUST be something used to TRACK TIME or CREATE INVOICES / BILL CLIENTS. If a tool is only about finding work, moving money, designing, or communicating, do not extract it.

For each qualifying product, capture a short evidence quote (~10-30 words) from the source text.
Use the canonical product name (e.g. "Toggl Track" → "Toggl"; keep "QuickBooks Time" as-is).

OUTPUT FORMAT: Return ONLY a JSON array. No prose. Each item:
{"product": "<name>", "thread_id": "<id>", "evidence": "<short quote>"}

If no qualifying products are mentioned, return []."""


def _thread_excerpt(t: Thread, max_chars: int = 2000) -> str:
    parts = [f"TITLE: {t.title}"]
    if t.selftext:
        parts.append(f"BODY: {t.selftext}")
    for i, c in enumerate(t.comments[:5]):
        parts.append(f"COMMENT_{i}: {c}")
    blob = "\n".join(parts)
    return blob[:max_chars]


def _build_user_message(batch: list[Thread]) -> str:
    sections = []
    for t in batch:
        sections.append(f"=== thread_id: {t.id} (r/{t.subreddit}) ===\n{_thread_excerpt(t)}")
    return (
        "Extract product names mentioned in these Reddit threads. "
        "Return a JSON array as specified.\n\n" + "\n\n".join(sections)
    )


def _normalize(name: str) -> str:
    s = name.strip().lower()
    aliases = {
        "toggl track": "toggl",
        "toggle": "toggl",
        "togglr": "toggl",
        "harvest app": "harvest",
        "clockify time tracker": "clockify",
        "quickbooks time": "quickbooks time",
        "qb time": "quickbooks time",
        "tsheets": "quickbooks time",
        "freshbooks classic": "freshbooks",
        "google sheet": "google sheets",
        "sheets": "google sheets",
        "ms excel": "excel",
        "microsoft excel": "excel",
        "notion app": "notion",
    }
    return aliases.get(s, s)


def _display_name(normalized: str, raw_seen: list[str]) -> str:
    """Pick the most common original casing as the display name."""
    if not raw_seen:
        return normalized.title()
    counts = Counter(raw_seen)
    return counts.most_common(1)[0][0]


def _process_batch(batch: list[Thread]) -> list[dict] | None:
    """One discovery LLM call. Returns parsed items, or None on failure."""
    user_msg = _build_user_message(batch)
    try:
        raw = call_with_cached_system(
            model=DISCOVERY_MODEL,
            system=DISCOVERY_SYSTEM,
            user=user_msg,
            max_tokens=2048,
        )
    except Exception as e:
        print(f"    [warn] discovery LLM call failed: {e}")
        return None
    data = extract_json(raw)
    if not isinstance(data, list):
        print(f"    [warn] could not parse JSON from discovery response")
        return None
    return data


def discover_competitors(threads: list[Thread]) -> tuple[list[dict], dict]:
    """Run discovery across all threads. Returns (competitor_list, debug_info)."""
    mentions_by_norm: dict[str, list[dict]] = defaultdict(list)
    raw_names_by_norm: dict[str, list[str]] = defaultdict(list)
    failed_batches = 0

    batches = [threads[i : i + DISCOVERY_BATCH_SIZE]
               for i in range(0, len(threads), DISCOVERY_BATCH_SIZE)]
    print(f"  [discovery] {len(batches)} batches, fanning out across {MAX_WORKERS} workers")

    # Fan out batches concurrently, then aggregate single-threaded.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        batch_results = list(pool.map(_process_batch, batches))

    denied = 0
    for data in batch_results:
        if data is None:
            failed_batches += 1
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            name = (item.get("product") or "").strip()
            if not name or len(name) > 60:
                continue
            norm = _normalize(name)
            # Drop non-competitors (marketplaces, payment processors, unrelated tools)
            if norm in DISCOVERY_DENYLIST:
                denied += 1
                continue
            mentions_by_norm[norm].append({
                "thread_id": item.get("thread_id"),
                "evidence": item.get("evidence", ""),
                "raw_name": name,
            })
            raw_names_by_norm[norm].append(name)
    if denied:
        print(f"  [discovery] dropped {denied} denylisted mention(s) (marketplaces/payments/unrelated)")

    competitors = []
    for norm, mentions in mentions_by_norm.items():
        if len(mentions) < MIN_MENTIONS_FOR_COMPETITOR:
            continue
        competitors.append({
            "name": _display_name(norm, raw_names_by_norm[norm]),
            "normalized": norm,
            "mention_count": len(mentions),
            "thread_ids": list({m["thread_id"] for m in mentions if m.get("thread_id")}),
        })
    competitors.sort(key=lambda c: c["mention_count"], reverse=True)

    debug = {
        "total_unique_names_seen": len(mentions_by_norm),
        "competitors_after_threshold": len(competitors),
        "failed_batches": failed_batches,
    }
    return competitors, debug
