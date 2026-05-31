# PRD — Switch sentiment scoring to a local model

> **Revision note:** We first tried distilBERT SST-2 (binary). On real Reddit excerpts it
> labelled ~75% of mentions negative and rated every competitor negative (including the
> best-liked tools) — SST-2 is sentence/movie-review domain and overconfident. We switched
> to **`cardiffnlp/twitter-roberta-base-sentiment-latest`** (social-media-tuned RoBERTa, native
> 3-way head), which fits Reddit text far better. This doc reflects the final design.

## Problem / motivation
The crawler currently classifies each competitor mention with Claude Haiku, which also
extracts pain-point/praise phrases used to synthesize themes. That works but (a) costs money
per run and (b) requires network + API key for the bulk of the calls. We want sentiment
polarity computed **locally and free** with a purpose-built model.

## Decision
Replace the per-mention LLM sentiment call with **`cardiffnlp/twitter-roberta-base-sentiment-latest`**
running locally via `transformers` + `torch`.

Accepted trade-off (chosen by user, "Full local" option): the local model only produces polarity,
so we **drop** pain-point/praise extraction and the cross-competitor theme synthesis. The
report becomes polarity + quote buckets.

## Scope

### Pipeline after change
1. **Fetch** — Reddit JSON (cached). *unchanged*
2. **Discovery** — Claude Sonnet extracts competitor names. *unchanged* (still needs `ANTHROPIC_API_KEY`).
3. **Sentiment** — local RoBERTa scores polarity for every (competitor, thread) mention. *new*
4. **Render** — Markdown report + SQL dump. *no theme sections*

### Sentiment model details
- Native **3-way head** (negative / neutral / positive). Per mention:
  - `sentiment` = argmax class label
  - `p_pos` = probability of the positive class
  - `score` = `p_pos - p_neg` → signed polarity in [-1, +1]
- Input: the ~640-char excerpt around the mention. `truncation=True, max_length=512`.
- Runs on CPU/MPS; batched. ~124 mentions → a few seconds. First run downloads model weights.

### Output changes
- `raw_mentions.jsonl`: keeps competitor/subreddit/thread/permalink/upvotes/date/quote/
  sentiment/score; **removes** `pain_points`, `praise`, `is_actual_user`; adds `p_pos`.
- **Markdown report**: Executive summary (top competitors by volume + avg score + %neg,
  sentiment distribution), per-competitor quote buckets (negative / positive / neutral,
  top by upvotes), subreddit coverage, methodology. **No** pain/praise theme tables, **no**
  cross-competitor themes.
- **SQL**: keep `competitors`, `mentions`, `subreddit_coverage`. **Drop**
  `pain_points`, `praise`, `competitor_themes`, `cross_themes`.

### Files touched
- `requirements.txt` — add `transformers`, `torch`.
- `distilbert_sentiment.py` — NEW: lazy-loaded pipeline + `classify(texts)`.
- `sentiment.py` — build mentions, call distilBERT instead of Claude; drop pain/praise.
- `synthesis.py` — DELETE (themes gone).
- `crawler.py` — 3-stage flow, drop synthesis + theme args.
- `render.py` — `build_report` no longer takes synth/cross_*; drop theme sections.
- `sql_export.py` — drop theme/phrase tables and their args.

## Non-goals
- No change to discovery (stays Claude). No fully-offline mode (could add seed-list discovery later).
- No neutral-class fine-tuning; neutral is a probability-band heuristic.

## Verification
1. `python crawler.py --limit 7` runs end-to-end; sentiment stage logs distilBERT progress.
2. `raw_mentions.jsonl` rows have `sentiment` ∈ {positive,neutral,negative} and `score` ∈ [-1,1].
3. Sentiment distribution is not 100% one class (sanity).
4. `sqlite3 v.db < competitor_sentiment.sql` loads with 3 tables, sensible row counts.
5. Spot-check 3 quotes: negative-labelled excerpts read negative.
