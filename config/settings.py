"""Shared runtime settings for SigDriftr."""

import json
import os


def _parse_scheduled_topics(default: list[str]) -> list[str]:
    raw = os.environ.get("SCHEDULED_TOPICS")
    if not raw:
        return list(default)

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return list(default)

    if not isinstance(decoded, list):
        return list(default)

    topics = [str(item).strip() for item in decoded if str(item).strip()]
    return topics or list(default)

MIN_ARTICLES_FOR_BASELINE = 30
BASELINE_EMA_ALPHA = 0.8

CONFIDENCE_COVERAGE_WEIGHT = 0.5
CONFIDENCE_BASELINE_WEIGHT = 0.5
CONFIDENCE_ARTICLE_DENOMINATOR = 50

MIN_BRIEF_CONFIDENCE = 0.4
HIGH_BRIEF_CONFIDENCE = 0.7

BANDIT_ALPHA = 0.85
BANDIT_TIME_BUCKET_HOURS = 6
BANDIT_MAX_FEEDS_PER_CRAWL = 6

CRAWL_FETCH_CONCURRENCY = 4
CRAWL_FEED_TIMEOUT_SECONDS = 15

EMBED_MODEL_NAME = os.environ.get(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()
EMBED_BATCH_SIZE = max(1, int(os.environ.get("EMBED_BATCH_SIZE", "32")))
EMBED_LIMIT_DEFAULT = max(1, int(os.environ.get("EMBED_LIMIT_DEFAULT", "200")))

CLUSTER_SIGNAL_PROVIDER = (
    os.environ.get("CLUSTER_SIGNAL_PROVIDER", "ollama").strip().lower() or "ollama"
)
CLUSTER_SIGNAL_MODEL = (
    os.environ.get("CLUSTER_SIGNAL_MODEL", "qwen2.5:7b-instruct").strip()
    or "qwen2.5:7b-instruct"
)
CLUSTER_SIGNAL_OLLAMA_URL = (
    os.environ.get("CLUSTER_SIGNAL_OLLAMA_URL", "http://localhost:11434/api/generate")
    .strip()
    or "http://localhost:11434/api/generate"
)
CLUSTER_SIGNAL_TIMEOUT_SECONDS = max(
    5,
    int(os.environ.get("CLUSTER_SIGNAL_TIMEOUT_SECONDS", "90")),
)
CLUSTER_SIGNAL_MIN_EXEMPLARS = max(
    1,
    int(os.environ.get("CLUSTER_SIGNAL_MIN_EXEMPLARS", "3")),
)
CLUSTER_SIGNAL_MAX_EXEMPLARS = max(
    CLUSTER_SIGNAL_MIN_EXEMPLARS,
    int(os.environ.get("CLUSTER_SIGNAL_MAX_EXEMPLARS", "5")),
)
CLUSTER_SIGNAL_BODY_CHAR_LIMIT = max(
    200,
    int(os.environ.get("CLUSTER_SIGNAL_BODY_CHAR_LIMIT", "900")),
)

SCHEDULED_TOPICS = _parse_scheduled_topics(["inflace", "energie", "zdravi"])
PIPELINE_INTERVAL_MINUTES = max(
    1,
    int(os.environ.get("PIPELINE_INTERVAL_MINUTES", "30")),
)
