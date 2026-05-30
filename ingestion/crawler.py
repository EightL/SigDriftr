import asyncio
import hashlib
import re
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
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
from config.settings import (
    BANDIT_REWARD_MODE,
    COLLECTION_MODE,
    CRAWL_FETCH_CONCURRENCY,
    CRAWL_FEED_TIMEOUT_SECONDS,
)
from config.source_panels import select_fixed_panel_feeds
from db.init import get_conn
from db.topic_resolver import TopicResolution, resolve_topic
from ingestion.bandit import record_crawl_miss, record_yield_reward, select_feeds

SEMANTIC_THRESHOLD = 0.55
REQUEST_HEADERS = {"User-Agent": "SigDriftr/1.0"}
MAX_EMBED_TEXT_LENGTH = 1200
MAX_BODY_CHARS = 6000
COLLECTION_MODES = {"bandit", "all", "fixed_panel"}
REWARD_MODES = {"yield", "signal"}


@dataclass
class FeedCrawlStats:
    outlet: str
    country: str = ""
    language: str | None = None
    selected: bool = True
    fetch_success: bool = False
    entries_seen: int = 0
    candidates: int = 0
    accepted: int = 0
    inserted: int = 0
    duplicates: int = 0
    relevance_total: float = 0.0
    reward: float = 0.0
    error_message: str | None = None

    @property
    def avg_relevance_score(self) -> float:
        if self.accepted <= 0:
            return 0.0
        return round(self.relevance_total / self.accepted, 4)


@dataclass
class CrawlReport:
    run_id: str
    topic: str
    canonical_topic_id: str
    country: str
    source: str
    collection_mode: str
    reward_mode: str
    eligible_feeds: list[str]
    selected_feeds: list[str]
    inserted: int = 0
    accepted: int = 0
    duplicates: int = 0
    started_at: str = ""
    completed_at: str = ""
    duration_s: float = 0.0
    feed_stats: list[FeedCrawlStats] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "canonical_topic_id": self.canonical_topic_id,
            "country": self.country,
            "source": self.source,
            "collection_mode": self.collection_mode,
            "reward_mode": self.reward_mode,
            "eligible_feeds": self.eligible_feeds,
            "selected_feeds": self.selected_feeds,
            "inserted": self.inserted,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "feed_stats": [
                {
                    "outlet": item.outlet,
                    "country": item.country,
                    "language": item.language,
                    "selected": item.selected,
                    "fetch_success": item.fetch_success,
                    "entries_seen": item.entries_seen,
                    "candidates": item.candidates,
                    "accepted": item.accepted,
                    "inserted": item.inserted,
                    "duplicates": item.duplicates,
                    "avg_relevance_score": item.avg_relevance_score,
                    "reward": item.reward,
                    "error_message": item.error_message,
                }
                for item in self.feed_stats
            ],
        }


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


def _topic_match_terms(resolution: TopicResolution) -> list[str]:
    terms = [
        resolution.requested_topic,
        resolution.canonical_topic_id,
        *resolution.aliases,
    ]
    return list(dict.fromkeys(term for term in terms if term))


def _best_relevance_score(
    topic_terms: list[str],
    title: str,
    summary: str,
    body: str = "",
) -> float:
    if not topic_terms:
        return 1.0
    return max(_relevance_score(term, title, summary, body) for term in topic_terms)


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


def _normalize_collection_mode(collection_mode: str | None) -> str:
    normalized = (collection_mode or COLLECTION_MODE).strip().lower() or "bandit"
    if normalized not in COLLECTION_MODES:
        raise ValueError(
            f"Unsupported collection_mode {collection_mode!r}; expected one of {sorted(COLLECTION_MODES)}."
        )
    return normalized


def _normalize_reward_mode(reward_mode: str | None) -> str:
    normalized = (reward_mode or BANDIT_REWARD_MODE).strip().lower() or "yield"
    if normalized not in REWARD_MODES:
        raise ValueError(
            f"Unsupported reward_mode {reward_mode!r}; expected one of {sorted(REWARD_MODES)}."
        )
    return normalized


def _select_collection_feeds(
    topic: str,
    *,
    country: str,
    source: str,
    now: datetime,
    collection_mode: str,
) -> tuple[list[dict], list[dict]]:
    available_feeds = get_enabled_feeds(country=country, source=source)
    if collection_mode == "all":
        return available_feeds, list(available_feeds)
    if collection_mode == "fixed_panel":
        return available_feeds, select_fixed_panel_feeds(
            available_feeds,
            topic=topic,
            country=country,
        )
    return available_feeds, select_feeds(topic, now=now, feeds=available_feeds)


def _persist_crawl_report(conn, report: CrawlReport) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO collection_runs
        (run_id, topic, canonical_topic_id, country, source, collection_mode, reward_mode,
         eligible_feeds, selected_feeds, inserted, accepted, duplicates,
         started_at, completed_at, duration_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report.run_id,
            report.topic,
            report.canonical_topic_id,
            report.country,
            report.source,
            report.collection_mode,
            report.reward_mode,
            len(report.eligible_feeds),
            len(report.selected_feeds),
            report.inserted,
            report.accepted,
            report.duplicates,
            report.started_at,
            report.completed_at,
            report.duration_s,
        ),
    )
    conn.execute(
        "DELETE FROM collection_feed_stats WHERE run_id = ?",
        (report.run_id,),
    )
    conn.executemany(
        """
        INSERT INTO collection_feed_stats
        (run_id, outlet, country, language, selected, fetch_success, entries_seen,
         candidates, accepted, inserted, duplicates, avg_relevance_score, reward,
         error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                report.run_id,
                item.outlet,
                item.country,
                item.language,
                int(item.selected),
                int(item.fetch_success),
                item.entries_seen,
                item.candidates,
                item.accepted,
                item.inserted,
                item.duplicates,
                item.avg_relevance_score,
                item.reward,
                item.error_message,
            )
            for item in report.feed_stats
        ],
    )


async def _fetch_article_details(
    feed: dict,
    entry: dict,
    topic_terms: list[str],
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

    final_score = _best_relevance_score(topic_terms, title, summary, body)
    if topic_terms and not _is_relevant(final_score):
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
    canonical_topic_id: str,
    relevance_score: float,
    matched_at: str,
) -> None:
    if not topic:
        return
    conn.execute(
        """
        INSERT INTO article_topics
        (article_id, topic, raw_topic, canonical_topic_id, relevance_score, matched_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id, topic) DO UPDATE SET
            raw_topic = excluded.raw_topic,
            canonical_topic_id = excluded.canonical_topic_id,
            relevance_score = MAX(article_topics.relevance_score, excluded.relevance_score),
            matched_at = excluded.matched_at
        """,
        (
            article_id,
            topic,
            topic,
            canonical_topic_id,
            round(relevance_score, 4),
            matched_at,
        ),
    )


async def _crawl_async_report(
    topic: str,
    country: str = "",
    source: str = "",
    *,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> CrawlReport:
    conn = get_conn()
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    resolution = resolve_topic(topic)
    canonical_topic_id = resolution.canonical_topic_id
    topic_terms = _topic_match_terms(resolution)
    normalized_mode = _normalize_collection_mode(collection_mode)
    normalized_reward_mode = _normalize_reward_mode(reward_mode)
    available_feeds, selected_feeds = _select_collection_feeds(
        canonical_topic_id or topic,
        country=country,
        source=source,
        now=now,
        collection_mode=normalized_mode,
    )
    report = CrawlReport(
        run_id=f"collect-{uuid.uuid4().hex}",
        topic=topic,
        canonical_topic_id=canonical_topic_id,
        country=(country or "").strip().upper(),
        source=(source or "").strip().lower(),
        collection_mode=normalized_mode,
        reward_mode=normalized_reward_mode,
        eligible_feeds=[str(feed.get("outlet", "")) for feed in available_feeds],
        selected_feeds=[str(feed.get("outlet", "")) for feed in selected_feeds],
        started_at=now.isoformat(),
    )
    semaphore = asyncio.Semaphore(CRAWL_FETCH_CONCURRENCY)
    parsed_feeds = await asyncio.gather(
        *[_fetch_feed(feed, semaphore) for feed in selected_feeds]
    )

    for feed, parsed in parsed_feeds:
        stats = FeedCrawlStats(
            outlet=str(feed.get("outlet", "")),
            country=str(feed.get("country", "")),
            language=feed.get("language"),
        )
        report.feed_stats.append(stats)
        if parsed is None:
            stats.error_message = "feed_fetch_failed"
            if normalized_reward_mode == "yield":
                stats.reward = record_yield_reward(
                    stats.outlet,
                    canonical_topic_id or topic,
                    accepted_count=0,
                    fetch_success=False,
                    when=now,
                    feed=feed,
                )
            else:
                stats.reward = record_crawl_miss(
                    stats.outlet,
                    canonical_topic_id or topic,
                    when=now,
                    feed=feed,
                )
            continue

        stats.fetch_success = True
        entries = list(getattr(parsed, "entries", []) or [])
        stats.entries_seen = len(entries)
        candidates: list[tuple[dict, float]] = []

        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            initial_score = _best_relevance_score(topic_terms, title, summary)
            if topic_terms and not _is_relevant(initial_score):
                continue
            candidates.append((entry, initial_score))
        stats.candidates = len(candidates)

        enriched_entries = await asyncio.gather(
            *[
                _fetch_article_details(feed, entry, topic_terms, score, semaphore)
                for entry, score in candidates
            ]
        )
        for item in enriched_entries:
            if item is None:
                continue
            stats.accepted += 1
            stats.relevance_total += float(item["relevance_score"] or 0.0)
            canonical_url = item["canonical_url"] or item["url"]
            article_id = hashlib.sha256(canonical_url.encode()).hexdigest()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO articles
                (id, outlet, title, summary, body, url, canonical_url, topic,
                 canonical_topic_id, country, language, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    canonical_topic_id,
                    feed.get("country", "CZ"),
                    feed.get("language", "cs"),
                    item["published_at"],
                    now.isoformat(),
                ),
            )
            if cursor.rowcount:
                report.inserted += 1
                stats.inserted += 1
            else:
                stats.duplicates += 1
                conn.execute(
                    """
                    UPDATE articles
                    SET outlet = ?,
                        title = ?,
                        summary = ?,
                        body = CASE WHEN COALESCE(?, '') = '' THEN body ELSE ? END,
                        canonical_url = COALESCE(?, canonical_url),
                        canonical_topic_id = COALESCE(?, canonical_topic_id),
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
                        canonical_topic_id,
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
                canonical_topic_id,
                item["relevance_score"],
                now.isoformat(),
            )

        report.accepted += stats.accepted
        report.duplicates += stats.duplicates
        if normalized_reward_mode == "yield":
            stats.reward = record_yield_reward(
                stats.outlet,
                canonical_topic_id or topic,
                accepted_count=stats.accepted,
                avg_relevance_score=stats.avg_relevance_score,
                duplicate_count=stats.duplicates,
                fetch_success=stats.fetch_success,
                when=now,
                feed=feed,
            )
        elif stats.accepted == 0:
            stats.reward = record_crawl_miss(
                stats.outlet,
                canonical_topic_id or topic,
                when=now,
                feed=feed,
            )

    report.completed_at = datetime.now(timezone.utc).isoformat()
    report.duration_s = round(time.perf_counter() - started, 4)
    _persist_crawl_report(conn, report)
    conn.commit()
    return report


async def _crawl_async(
    topic: str,
    country: str = "",
    source: str = "",
    *,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> int:
    report = await _crawl_async_report(
        topic,
        country=country,
        source=source,
        collection_mode=collection_mode,
        reward_mode=reward_mode,
    )
    return report.inserted


def crawl(
    topic: str,
    country: str = "",
    source: str = "",
    *,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> int:
    """Run the async crawler from sync code paths.

    This wrapper assumes there is no active event loop. Async callers should
    await `_crawl_async()` directly to avoid the usual `asyncio.run()` error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _crawl_async(
                topic,
                country=country,
                source=source,
                collection_mode=collection_mode,
                reward_mode=reward_mode,
            )
        )
    raise RuntimeError(
        "crawl() is a synchronous wrapper and cannot run inside an active event loop; "
        "await _crawl_async(topic) instead; optional country/source filters may also be passed."
    )
