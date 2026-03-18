import asyncio
import hashlib
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache

try:
    import feedparser as _feedparser
except ModuleNotFoundError:
    class _FeedparserProxy:
        @staticmethod
        def parse(*args, **kwargs):
            raise ModuleNotFoundError("feedparser")

    feedparser = _FeedparserProxy()
else:
    feedparser = _feedparser

from config.settings import CRAWL_FETCH_CONCURRENCY, CRAWL_FEED_TIMEOUT_SECONDS
from db.init import get_conn
from ingestion.bandit import record_crawl_miss, select_feeds

SEMANTIC_THRESHOLD = 0.55
REQUEST_HEADERS = {"User-Agent": "SigDriftr/1.0"}


@lru_cache(maxsize=1)
def _get_embedder():
    """Lazy-load the multilingual sentence embedder once."""
    try:
        from sentence_transformers import SentenceTransformer, util as st_util

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model, st_util
    except Exception:
        return None, None


@lru_cache(maxsize=128)
def _get_topic_embedding(topic: str):
    model, _ = _get_embedder()
    if model is None:
        return None
    return model.encode(topic, convert_to_tensor=True)


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

    topic_emb = _get_topic_embedding(topic_lower)
    if topic_emb is None:
        return False
    article_emb = model.encode(combined[:512], convert_to_tensor=True)
    score = float(st_util.cos_sim(topic_emb, article_emb)[0][0])
    return score >= SEMANTIC_THRESHOLD


def _fetch_feed_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=REQUEST_HEADERS, method="GET")
    with urllib.request.urlopen(request, timeout=CRAWL_FEED_TIMEOUT_SECONDS) as response:
        return response.read()


async def _fetch_feed(feed: dict, semaphore: asyncio.Semaphore) -> tuple[dict, object | None]:
    async with semaphore:
        try:
            payload = await asyncio.to_thread(_fetch_feed_bytes, feed["rss_url"])
            parsed = await asyncio.to_thread(feedparser.parse, payload)
            return feed, parsed
        except Exception:
            return feed, None


async def _crawl_async(topic: str) -> int:
    conn = get_conn()
    inserted = 0
    now = datetime.now(timezone.utc)
    selected_feeds = select_feeds(topic, now=now)
    semaphore = asyncio.Semaphore(CRAWL_FETCH_CONCURRENCY)
    parsed_feeds = await asyncio.gather(
        *[_fetch_feed(feed, semaphore) for feed in selected_feeds]
    )

    for feed, parsed in parsed_feeds:
        matched_entries = 0
        entries = list(getattr(parsed, "entries", []) or [])

        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not _is_relevant(topic, title, summary):
                continue

            matched_entries += 1
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
                    now.isoformat(),
                ),
            )
            inserted += cursor.rowcount

        if matched_entries == 0:
            record_crawl_miss(feed["outlet"], topic, when=now, feed=feed)

    conn.commit()
    return inserted


def crawl(topic: str) -> int:
    return asyncio.run(_crawl_async(topic))
