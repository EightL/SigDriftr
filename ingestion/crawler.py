import asyncio
import hashlib
import re
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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

from config.feeds import get_enabled_feeds
from config.settings import CRAWL_FETCH_CONCURRENCY, CRAWL_FEED_TIMEOUT_SECONDS
from db.init import get_conn
from ingestion.bandit import record_crawl_miss, select_feeds

SEMANTIC_THRESHOLD = 0.55
REQUEST_HEADERS = {"User-Agent": "SigDriftr/1.0"}
MAX_EMBED_TEXT_LENGTH = 1200
MAX_BODY_CHARS = 6000


@lru_cache(maxsize=1)
def _get_fulltext_extractor():
    try:
        import trafilatura as _trafilatura
    except Exception:
        return None
    return _trafilatura


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


def _clean_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    return compact[:MAX_BODY_CHARS]


def _extract_fallback_text(html: str, fallback: str) -> str:
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    if not paragraphs:
        return _clean_text(fallback)
    combined = " ".join(
        re.sub(r"<[^>]+>", " ", paragraph)
        for paragraph in paragraphs[:20]
    )
    combined = re.sub(r"&[a-z#0-9]+;", " ", combined, flags=re.IGNORECASE)
    return _clean_text(combined or fallback)


def _normalize_published_at(entry: dict) -> str | None:
    raw_value = str(entry.get("published") or entry.get("updated") or "").strip()
    parsed = None

    if raw_value:
        try:
            parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw_value)
            except (TypeError, ValueError, IndexError):
                parsed = None

    if parsed is None:
        parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed_struct is not None:
            try:
                parsed = datetime(*parsed_struct[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError, IndexError):
                parsed = None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _text_for_similarity(title: str, summary: str, body: str = "") -> str:
    return _clean_text(" ".join(part for part in [title, summary, body] if part))


def _relevance_score(topic: str, title: str, summary: str, body: str = "") -> float:
    """Score relevance using direct match first and multilingual embeddings second."""
    combined = _text_for_similarity(title, summary, body).lower()
    topic_lower = topic.lower()

    if not topic_lower:
        return 1.0

    title_lower = (title or "").lower()
    if topic_lower in title_lower:
        return 1.0
    if topic_lower in combined:
        return 0.95

    model, st_util = _get_embedder()
    if model is None:
        return 0.0

    topic_emb = _get_topic_embedding(topic_lower)
    if topic_emb is None:
        return 0.0
    article_emb = model.encode(combined[:MAX_EMBED_TEXT_LENGTH], convert_to_tensor=True)
    score = float(st_util.cos_sim(topic_emb, article_emb)[0][0])
    return round(max(0.0, min(1.0, score)), 4)


def _is_relevant(score: float) -> bool:
    return score >= SEMANTIC_THRESHOLD


def _fetch_url_bytes(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers=REQUEST_HEADERS, method="GET")
    with urllib.request.urlopen(request, timeout=CRAWL_FEED_TIMEOUT_SECONDS) as response:
        return response.read(), response.geturl()


def _fetch_feed_bytes(url: str) -> bytes:
    payload, _ = _fetch_url_bytes(url)
    return payload


def _fetch_article_body(url: str, fallback_summary: str) -> tuple[str, str]:
    try:
        payload, resolved_url = _fetch_url_bytes(url)
    except Exception:
        return _clean_text(fallback_summary), url

    html = payload.decode("utf-8", errors="replace")
    extractor = _get_fulltext_extractor()
    if extractor is not None:
        try:
            extracted = extractor.extract(
                html,
                include_comments=False,
                include_tables=False,
                output_format="txt",
            )
        except Exception:
            extracted = None
        if extracted:
            return _clean_text(extracted), resolved_url

    return _extract_fallback_text(html, fallback_summary), resolved_url


async def _fetch_feed(feed: dict, semaphore: asyncio.Semaphore) -> tuple[dict, object | None]:
    async with semaphore:
        try:
            payload = await asyncio.to_thread(_fetch_feed_bytes, feed["rss_url"])
            parsed = await asyncio.to_thread(feedparser.parse, payload)
            return feed, parsed
        except Exception:
            return feed, None


async def _fetch_article_details(
    feed: dict,
    entry: dict,
    topic: str,
    initial_score: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    title = entry.get("title", "")
    summary = entry.get("summary", "")
    url = entry.get("link", "")
    if not url:
        return None

    async with semaphore:
        body, canonical_url = await asyncio.to_thread(_fetch_article_body, url, summary)

    final_score = _relevance_score(topic, title, summary, body)
    if topic and not _is_relevant(final_score):
        return None

    return {
        "feed": feed,
        "title": title,
        "summary": summary,
        "body": body,
        "url": url,
        "canonical_url": canonical_url or url,
        "published_at": _normalize_published_at(entry),
        "relevance_score": max(initial_score, final_score),
    }


def _upsert_article_topic(
    conn,
    article_id: str,
    topic: str,
    relevance_score: float,
    matched_at: str,
) -> None:
    if not topic:
        return
    conn.execute(
        """
        INSERT INTO article_topics (article_id, topic, relevance_score, matched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(article_id, topic) DO UPDATE SET
            relevance_score = MAX(article_topics.relevance_score, excluded.relevance_score),
            matched_at = excluded.matched_at
        """,
        (article_id, topic, round(relevance_score, 4), matched_at),
    )


async def _crawl_async(topic: str, country: str = "", source: str = "") -> int:
    conn = get_conn()
    inserted = 0
    now = datetime.now(timezone.utc)
    available_feeds = get_enabled_feeds(country=country, source=source)
    selected_feeds = select_feeds(topic, now=now, feeds=available_feeds)
    semaphore = asyncio.Semaphore(CRAWL_FETCH_CONCURRENCY)
    parsed_feeds = await asyncio.gather(
        *[_fetch_feed(feed, semaphore) for feed in selected_feeds]
    )

    for feed, parsed in parsed_feeds:
        matched_entries = 0
        entries = list(getattr(parsed, "entries", []) or [])
        candidates: list[tuple[dict, float]] = []

        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            initial_score = _relevance_score(topic, title, summary)
            if topic and not _is_relevant(initial_score):
                continue
            candidates.append((entry, initial_score))

        enriched_entries = await asyncio.gather(
            *[
                _fetch_article_details(feed, entry, topic, score, semaphore)
                for entry, score in candidates
            ]
        )
        for item in enriched_entries:
            if item is None:
                continue
            matched_entries += 1
            canonical_url = item["canonical_url"] or item["url"]
            article_id = hashlib.sha256(canonical_url.encode()).hexdigest()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO articles
                (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id,
                    feed["outlet"],
                    item["title"],
                    item["summary"],
                    item["body"],
                    item["url"],
                    canonical_url,
                    topic,
                    feed.get("country", "CZ"),
                    feed.get("language", "cs"),
                    item["published_at"],
                    now.isoformat(),
                ),
            )
            if cursor.rowcount:
                inserted += 1
            else:
                conn.execute(
                    """
                    UPDATE articles
                    SET outlet = ?,
                        title = ?,
                        summary = ?,
                        body = CASE WHEN COALESCE(?, '') = '' THEN body ELSE ? END,
                        canonical_url = COALESCE(?, canonical_url),
                        country = ?,
                        language = ?,
                        published_at = COALESCE(?, published_at),
                        fetched_at = ?
                    WHERE id = ?
                    """,
                    (
                        feed["outlet"],
                        item["title"],
                        item["summary"],
                        item["body"],
                        item["body"],
                        canonical_url,
                        feed.get("country", "CZ"),
                        feed.get("language", "cs"),
                        item["published_at"],
                        now.isoformat(),
                        article_id,
                    ),
                )

            _upsert_article_topic(
                conn,
                article_id,
                topic,
                item["relevance_score"],
                now.isoformat(),
            )

        if matched_entries == 0:
            record_crawl_miss(feed["outlet"], topic, when=now, feed=feed)

    conn.commit()
    return inserted


def crawl(topic: str, country: str = "", source: str = "") -> int:
    """Run the async crawler from sync code paths.

    This wrapper assumes there is no active event loop. Async callers should
    await `_crawl_async()` directly to avoid the usual `asyncio.run()` error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_crawl_async(topic, country=country, source=source))
    raise RuntimeError(
        "crawl() is a synchronous wrapper and cannot run inside an active event loop; "
        "await _crawl_async(topic) instead; optional country/source filters may also be passed."
    )
