"""Build a SQLite/Postgres-compatible .sql dump (schema + data) as a string.

The dashboard embeds this string and offers it as a download; nothing is written
to disk by the pipeline.

Tables:
  competitors        — one row per competitor, with mention count + avg sentiment
  mentions           — one row per (competitor, thread) mention, with polarity
  subreddit_coverage — per-subreddit thread/mention counts

Once downloaded, load with:  sqlite3 timetracer.db < competitor_sentiment.sql
or in Supabase:  paste into the SQL Editor and run.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _q(value) -> str:
    """Quote a value as a SQL literal (SQLite/Postgres compatible)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


SCHEMA = """-- TimeTracer competitor sentiment — schema
DROP TABLE IF EXISTS competitors;
DROP TABLE IF EXISTS mentions;
DROP TABLE IF EXISTS subreddit_coverage;

CREATE TABLE competitors (
    name            TEXT PRIMARY KEY,
    mention_count   INTEGER NOT NULL,
    avg_sentiment   REAL,
    pct_negative    REAL
);

CREATE TABLE mentions (
    id              INTEGER PRIMARY KEY,
    competitor      TEXT NOT NULL,
    subreddit       TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    permalink       TEXT,
    upvotes         INTEGER,
    posted_date     TEXT,           -- YYYY-MM-DD
    sentiment       TEXT,           -- positive | neutral | negative
    score           REAL,           -- -1.0 .. +1.0  (2*p_pos - 1)
    p_pos           REAL,           -- distilBERT POSITIVE-class probability
    quote           TEXT
);

CREATE TABLE subreddit_coverage (
    subreddit       TEXT PRIMARY KEY,
    threads         INTEGER,
    mentions        INTEGER,
    top_competitor  TEXT
);
"""


def build_sql_string(
    *,
    subreddits_scanned: list[str],
    threads_by_sub: dict[str, list],
    mentions: list[dict],
) -> str:
    """Build the full SQL dump as a string (schema + data + example queries).

    No file is written — callers embed this (e.g. in the dashboard for download).
    """
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_comp[m["competitor"]].append(m)

    lines: list[str] = [SCHEMA, "", "BEGIN TRANSACTION;", ""]

    # competitors
    lines.append("-- competitors")
    for comp, ms in sorted(by_comp.items(), key=lambda kv: len(kv[1]), reverse=True):
        avg = sum(m["score"] for m in ms) / max(len(ms), 1)
        neg_pct = 100 * sum(1 for m in ms if m["sentiment"] == "negative") / max(len(ms), 1)
        lines.append(
            "INSERT INTO competitors (name, mention_count, avg_sentiment, pct_negative) VALUES ("
            f"{_q(comp)}, {len(ms)}, {round(avg, 4)}, {round(neg_pct, 2)});"
        )
    lines.append("")

    # mentions
    lines.append("-- mentions")
    for mid, m in enumerate(mentions, 1):
        quote = m.get("quote") or m.get("excerpt") or ""
        lines.append(
            "INSERT INTO mentions (id, competitor, subreddit, thread_id, permalink, "
            "upvotes, posted_date, sentiment, score, p_pos, quote) VALUES ("
            f"{mid}, {_q(m['competitor'])}, {_q(m['subreddit'])}, {_q(m['thread_id'])}, "
            f"{_q(m.get('permalink'))}, {_q(m.get('upvotes'))}, {_q(m.get('date'))}, "
            f"{_q(m.get('sentiment'))}, {_q(m.get('score'))}, {_q(m.get('p_pos'))}, {_q(quote)});"
        )
    lines.append("")

    # subreddit_coverage
    lines.append("-- subreddit_coverage")
    mentions_by_sub: dict[str, Counter] = defaultdict(Counter)
    for m in mentions:
        mentions_by_sub[m["subreddit"]][m["competitor"]] += 1
    for sub in subreddits_scanned:
        n_threads = len(threads_by_sub.get(sub, []))
        n_mentions = sum(mentions_by_sub[sub].values())
        top = mentions_by_sub[sub].most_common(1)
        top_comp = top[0][0] if top else None
        lines.append(
            "INSERT INTO subreddit_coverage (subreddit, threads, mentions, top_competitor) VALUES ("
            f"{_q(sub)}, {n_threads}, {n_mentions}, {_q(top_comp)});"
        )
    lines.append("")

    lines.append("COMMIT;")
    lines.append("")
    lines.append(_EXAMPLE_QUERIES)

    return "\n".join(lines)


_EXAMPLE_QUERIES = """-- ============================================================
-- Example queries
-- ============================================================
-- Competitors ranked by how negative the sentiment is:
--   SELECT name, mention_count, avg_sentiment, pct_negative
--   FROM competitors ORDER BY avg_sentiment ASC;
--
-- All negative quotes for a given competitor (TimeTracer opportunities):
--   SELECT subreddit, upvotes, quote, permalink FROM mentions
--   WHERE competitor = 'Harvest' AND sentiment = 'negative'
--   ORDER BY upvotes DESC;
--
-- Sentiment mix per competitor:
--   SELECT competitor, sentiment, COUNT(*) AS n FROM mentions
--   GROUP BY competitor, sentiment ORDER BY competitor, sentiment;
--
-- Which subreddit yields the most competitor chatter:
--   SELECT subreddit, mentions, top_competitor FROM subreddit_coverage
--   ORDER BY mentions DESC;
"""
