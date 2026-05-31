"""Static configuration for the TimeTracer Reddit crawler."""

from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"

USER_AGENT = "timetracer-research/0.1 (competitive sentiment research)"

SUBREDDITS = [
    # Tier 1 — hourly billers
    ("freelance", 1),
    ("forhire", 1),
    ("freelanceWritersForum", 1),
    ("Upwork", 1),
    ("freelancers", 1),
    ("digitalnomad", 1),
    ("consulting", 1),
    # Tier 2 — tech contractors
    ("webdev", 2),
    ("programming", 2),
    ("cscareerquestions", 2),
    ("ExperiencedDevs", 2),
    ("devops", 2),
    ("SideProject", 2),
    ("indiehackers", 2),
]

SEARCH_QUERIES = [
    "time tracking",
    "time tracker",
    "hourly billing",
    "invoicing",
    "timesheet",
    "Harvest Toggl Clockify",
    "freelance billing app",
    "best time tracking app",
]

# Per-subreddit ceilings
MAX_THREADS_PER_SUBREDDIT = 50
MAX_COMMENTS_PER_THREAD = 30

# Relevance filter — at least one of these must appear in the thread
RELEVANCE_KEYWORDS = [
    "time track",
    "time-track",
    "timetrack",
    "timesheet",
    "time sheet",
    "invoic",
    "billing",
    "hourly rate",
    "bill by the hour",
    "bill hourly",
    "freelance tool",
    "harvest",
    "toggl",
    "clockify",
    "freshbooks",
    "quickbooks",
]

# LLM configuration (discovery only; sentiment is local distilBERT)
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Discovery still uses an LLM to extract competitor names from threads.
DISCOVERY_MODEL = SONNET_MODEL

# Local sentiment model (Hugging Face). Social-media-tuned RoBERTa with a native
# 3-way head (negative / neutral / positive) — a much better fit for Reddit text
# than the binary SST-2 distilBERT. Label = argmax; signed score = p_pos - p_neg.
SENTIMENT_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Cost guardrails
MAX_SENTIMENT_CALLS = 1000
MAX_SYNTHESIS_CALLS = 30
DISCOVERY_BATCH_SIZE = 10
MIN_MENTIONS_FOR_COMPETITOR = 3
MIN_MENTIONS_FOR_SYNTHESIS = 5

# Discovery denylist: normalized (lowercased) product names that are NOT
# time-tracking / billing / invoicing competitors and should be dropped even
# if the LLM surfaces them. Covers marketplaces, payment processors, and
# unrelated tools (design, comms, storage, generic productivity).
DISCOVERY_DENYLIST = {
    # freelance marketplaces
    "upwork", "fiverr", "freelancer", "freelancer.com", "toptal", "guru",
    "guru.com", "peopleperhour", "contra", "99designs", "dribbble", "behance",
    "people per hour", "flexjobs", "we work remotely",
    # payment processors / money movement
    "stripe", "paypal", "wise", "transferwise", "payoneer", "venmo", "revolut",
    "mercury", "square", "cash app", "cashapp", "zelle", "remitly", "skrill",
    "gumroad", "lemonsqueezy", "lemon squeezy", "ach", "swift",
    # design / content / comms / storage / generic productivity
    "canva", "loom", "figma", "photoshop", "adobe", "google docs", "google doc",
    "google drive", "gdrive", "dropbox", "slack", "zoom", "calendly", "trello",
    "gmail", "google calendar", "microsoft word", "word", "powerpoint",
    "google meet", "discord", "chatgpt", "claude", "github", "vscode",
    "linkedin", "obsidian", "evernote", "todoist", "airtable",
}

# Reddit politeness
REQUEST_DELAY_SECONDS = 1.5
