import hashlib
from datetime import datetime, timezone
from functools import lru_cache

import feedparser

from config.feeds import FEEDS
from db.init import get_conn

SEMANTIC_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def _get_embedder():
    """Lazy-load the multilingual sentence embedder once."""
    try:
        from sentence_transformers import SentenceTransformer, util as st_util

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model, st_util
    except Exception:
        return None, None


def _is_relevant(topic: str, title: str, summary: str) -> bool:
    """Two-pass relevance check: keyword pre-filter + semantic similarity."""
    combined = f"{title} {summary}".lower()
    topic_lower = topic.lower()

    if not topic_lower:
        return True

    if topic_lower in combined:
        return True

    model, st_util = _get_embedder()
    if model is None:
        return False

    topic_emb = model.encode(topic, convert_to_tensor=True)
    article_emb = model.encode(combined[:512], convert_to_tensor=True)
    score = float(st_util.cos_sim(topic_emb, article_emb)[0][0])
    return score >= SEMANTIC_THRESHOLD


def crawl(topic: str) -> int:
    conn = get_conn()
    inserted = 0

    for feed in FEEDS:
        parsed = feedparser.parse(feed["rss_url"])

        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not _is_relevant(topic, title, summary):
                continue

            url = entry.get("link", "")
            if not url:
                continue

            article_id = hashlib.sha256(url.encode()).hexdigest()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    article_id,
                    feed["outlet"],
                    title,
                    summary,
                    url,
                    topic,
                    entry.get("published", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted += cursor.rowcount

    return inserted
