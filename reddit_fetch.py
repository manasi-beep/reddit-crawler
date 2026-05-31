"""Fetch Reddit threads via public JSON endpoints with on-disk caching."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import (
    CACHE_DIR,
    MAX_COMMENTS_PER_THREAD,
    MAX_THREADS_PER_SUBREDDIT,
    RELEVANCE_KEYWORDS,
    REQUEST_DELAY_SECONDS,
    SEARCH_QUERIES,
    USER_AGENT,
)


@dataclass
class Thread:
    id: str
    subreddit: str
    title: str
    selftext: str
    permalink: str
    score: int
    created_utc: float
    comments: list[str] = field(default_factory=list)

    def text_blob(self) -> str:
        return "\n".join([self.title, self.selftext, *self.comments])


class RedditFetchError(Exception):
    pass


_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _cache_path(url: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".json")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(RedditFetchError),
    reraise=True,
)
def _http_get(url: str) -> dict:
    time.sleep(REQUEST_DELAY_SECONDS)
    try:
        resp = _session.get(url, timeout=20)
    except requests.RequestException as e:
        raise RedditFetchError(f"network error: {e}") from e
    if resp.status_code == 429:
        raise RedditFetchError("429 rate limited")
    if resp.status_code >= 500:
        raise RedditFetchError(f"{resp.status_code} server error")
    if resp.status_code != 200:
        raise RedditFetchError(f"{resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as e:
        raise RedditFetchError(f"invalid json: {e}") from e


def _fetch_cached(url: str, use_cache: bool = True) -> dict:
    path = _cache_path(url)
    if use_cache and path.exists():
        with path.open() as f:
            return json.load(f)
    data = _http_get(url)
    with path.open("w") as f:
        json.dump(data, f)
    return data


def _search_subreddit(subreddit: str, query: str, use_cache: bool = True) -> list[dict]:
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={requests.utils.quote(query)}&restrict_sr=1&sort=relevance&t=year&limit=100"
    )
    data = _fetch_cached(url, use_cache=use_cache)
    children = (data.get("data") or {}).get("children") or []
    return [c["data"] for c in children if c.get("kind") == "t3"]


def _fetch_thread(subreddit: str, thread_id: str, use_cache: bool = True) -> list[str]:
    url = (
        f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}.json"
        f"?limit={MAX_COMMENTS_PER_THREAD}&sort=top"
    )
    try:
        data = _fetch_cached(url, use_cache=use_cache)
    except RedditFetchError:
        return []
    if not isinstance(data, list) or len(data) < 2:
        return []
    comments_listing = data[1].get("data", {}).get("children", [])
    out: list[str] = []
    for c in comments_listing:
        if c.get("kind") != "t1":
            continue
        body = (c.get("data") or {}).get("body")
        if body and body not in {"[deleted]", "[removed]"}:
            out.append(body.strip())
        if len(out) >= MAX_COMMENTS_PER_THREAD:
            break
    return out


_relevance_re = re.compile("|".join(re.escape(k) for k in RELEVANCE_KEYWORDS), re.IGNORECASE)


def is_relevant(text: str) -> bool:
    return bool(_relevance_re.search(text or ""))


def gather_threads(subreddit: str, use_cache: bool = True) -> list[Thread]:
    """Search a subreddit with all queries, dedupe, expand comments, filter for relevance."""
    seen: dict[str, dict] = {}
    for q in SEARCH_QUERIES:
        try:
            for post in _search_subreddit(subreddit, q, use_cache=use_cache):
                pid = post.get("id")
                if pid and pid not in seen:
                    seen[pid] = post
        except RedditFetchError as e:
            print(f"  [warn] search failed for r/{subreddit} q={q!r}: {e}")

    # Sort by score desc, take top N candidates. Threads matched at least one of our
    # search queries so they're already topical — fetch all of them up to the cap.
    candidates = sorted(seen.values(), key=lambda p: p.get("score", 0), reverse=True)
    candidates = candidates[: MAX_THREADS_PER_SUBREDDIT * 2]

    threads: list[Thread] = []
    for post in candidates:
        title = post.get("title", "") or ""
        selftext = post.get("selftext", "") or ""

        comments = _fetch_thread(subreddit, post["id"], use_cache=use_cache)
        full_text = title + " " + selftext + " " + " ".join(comments)
        if not is_relevant(full_text):
            continue

        threads.append(
            Thread(
                id=post["id"],
                subreddit=subreddit,
                title=title,
                selftext=selftext,
                permalink="https://www.reddit.com" + post.get("permalink", ""),
                score=int(post.get("score", 0)),
                created_utc=float(post.get("created_utc", 0)),
                comments=comments,
            )
        )
        if len(threads) >= MAX_THREADS_PER_SUBREDDIT:
            break

    return threads


def gather_all(subreddits: Iterable[str], use_cache: bool = True) -> dict[str, list[Thread]]:
    out: dict[str, list[Thread]] = {}
    for sub in subreddits:
        print(f"[fetch] r/{sub} ...")
        threads = gather_threads(sub, use_cache=use_cache)
        print(f"  -> {len(threads)} relevant threads")
        out[sub] = threads
    return out
