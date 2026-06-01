#!/usr/bin/env python3
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db.init
import pytest

from db_helpers import ORIGINAL_DB_PATH, cleanup_temp_db, setup_temp_db

pytest.importorskip("pydantic")


def _single_entry_feed(topic_text: str = "Inflace a ekonomika meni trh") -> SimpleNamespace:
    return SimpleNamespace(
        entries=[
            {
                "title": topic_text,
                "summary": "Shrnuti k tematu inflace a ekonomika.",
                "link": "https://example.test/articles/shared-story",
                "published": "2026-03-18T10:00:00+00:00",
            }
        ]
    )


def test_db_migration_adds_stage_one_columns_and_topic_links() -> None:
    temp_dir = tempfile.TemporaryDirectory()
    try:
        db_path = Path(temp_dir.name) / "sigdriftr.db"
        legacy = sqlite3.connect(str(db_path))
        legacy.execute(
            """
            CREATE TABLE articles (
                id TEXT PRIMARY KEY,
                outlet TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT UNIQUE NOT NULL,
                topic TEXT,
                published_at TEXT,
                fetched_at TEXT NOT NULL
            )
            """
        )
        legacy.execute(
            """
            CREATE TABLE baselines (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                segment TEXT NOT NULL,
                concern_level REAL,
                purchase_intent REAL,
                avoidance_signals REAL,
                dominant_frame TEXT,
                seeded INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        legacy.execute(
            """
            INSERT INTO articles
            (id, outlet, title, summary, url, topic, published_at, fetched_at)
            VALUES ('legacy-1', 'irozhlas', 'Legacy title', 'Legacy summary',
                    'https://example.test/legacy-1', 'inflace',
                    '2026-03-18T10:00:00+00:00', '2026-03-18T10:05:00+00:00')
            """
        )
        legacy.commit()
        legacy.close()

        db.init.DB_PATH = db_path
        if hasattr(db.init._local, "conn"):
            db.init._local.conn.close()
            delattr(db.init._local, "conn")

        conn = db.init.get_conn()
        article_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()
        }
        topic_rows = conn.execute(
            "SELECT article_id, topic FROM article_topics ORDER BY article_id, topic"
        ).fetchall()
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        if hasattr(db.init._local, "conn"):
            db.init._local.conn.close()
            delattr(db.init._local, "conn")
        db.init.DB_PATH = ORIGINAL_DB_PATH
        temp_dir.cleanup()

    assert {"body", "country", "language", "canonical_url"}.issubset(article_columns)
    assert topic_rows == [("legacy-1", "inflace")]
    assert {"collection_runs", "collection_feed_stats"}.issubset(table_names)


def test_crawl_filters_feeds_by_country_and_source() -> None:
    from ingestion.crawler import crawl

    temp_dir = setup_temp_db()
    captured: dict[str, object] = {}
    try:
        def fake_select_feeds(topic: str, now=None, k: int = 4, feeds=None):
            captured["feeds"] = feeds
            return []

        with patch("ingestion.crawler.select_feeds", side_effect=fake_select_feeds):
            inserted = crawl("inflace", country="DE", source="spiegel")
    finally:
        cleanup_temp_db(temp_dir)

    assert inserted == 0
    assert captured["feeds"] == [
        {
            "outlet": "spiegel",
            "rss_url": "https://www.spiegel.de/schlagzeilen/index.rss",
            "affinity_tag": "mainstream",
            "country": "DE",
            "language": "de",
            "enabled": True,
        }
    ]


def test_crawl_fixed_panel_mode_skips_bandit_selection() -> None:
    from ingestion.crawler import crawl

    temp_dir = setup_temp_db()
    try:
        with patch("ingestion.crawler.select_feeds") as mock_select, patch(
            "ingestion.crawler._fetch_feed_bytes",
            side_effect=RuntimeError("network blocked in unit test"),
        ):
            inserted = crawl("inflace", country="DE", collection_mode="fixed_panel")

        rows = db.init.get_conn().execute(
            """
            SELECT outlet, fetch_success, reward
            FROM collection_feed_stats
            ORDER BY outlet
            """
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert inserted == 0
    mock_select.assert_not_called()
    assert [row[0] for row in rows] == ["spiegel", "tagesschau"]
    assert all(row[1] == 0 for row in rows)
    assert all(row[2] == 0.0 for row in rows)


def test_same_article_can_map_to_multiple_topics_without_duplicate_article_rows() -> None:
    from ingestion.crawler import crawl

    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "spiegel",
            "rss_url": "https://example.test/rss",
            "affinity_tag": "mainstream",
            "country": "DE",
            "language": "de",
            "enabled": True,
        }
        with patch("ingestion.crawler.select_feeds", return_value=[feed]), patch(
            "ingestion.crawler._fetch_feed_bytes",
            return_value=b"<rss />",
        ), patch(
            "ingestion.crawler.feedparser.parse",
            return_value=_single_entry_feed(),
        ), patch(
            "ingestion.crawler._fetch_article_body",
            return_value=("Full article body for testing.", "https://example.test/articles/shared-story"),
        ):
            crawl("inflace")
            crawl("ekonomika")

        conn = db.init.get_conn()
        article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        topic_links = conn.execute(
            "SELECT topic FROM article_topics ORDER BY topic"
        ).fetchall()
        run_stats = conn.execute(
            """
            SELECT inserted, accepted, duplicates
            FROM collection_runs
            ORDER BY completed_at
            """
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert article_count == 1
    assert topic_links == [("ekonomika",), ("inflace",)]
    assert run_stats == [(1, 1, 0), (0, 1, 1)]


def test_alias_topics_share_canonical_article_links_without_duplicate_rows() -> None:
    from ingestion.crawler import crawl

    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "spiegel",
            "rss_url": "https://example.test/rss",
            "affinity_tag": "mainstream",
            "country": "DE",
            "language": "de",
            "enabled": True,
        }
        with patch("ingestion.crawler.select_feeds", return_value=[feed]), patch(
            "ingestion.crawler._fetch_feed_bytes",
            return_value=b"<rss />",
        ), patch(
            "ingestion.crawler.feedparser.parse",
            return_value=_single_entry_feed("Energie zdrazuje domacnostem"),
        ), patch(
            "ingestion.crawler._fetch_article_body",
            return_value=("Full article body about energy prices.", "https://example.test/articles/energy-story"),
        ):
            crawl("energie")
            crawl("energy")

        conn = db.init.get_conn()
        article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        topic_links = conn.execute(
            """
            SELECT topic, raw_topic, canonical_topic_id
            FROM article_topics
            ORDER BY topic
            """
        ).fetchall()
        run_scopes = conn.execute(
            """
            SELECT topic, canonical_topic_id, inserted, accepted, duplicates
            FROM collection_runs
            ORDER BY completed_at
            """
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert article_count == 1
    assert topic_links == [
        ("energie", "energie", "energy"),
        ("energy", "energy", "energy"),
    ]
    assert run_scopes == [
        ("energie", "energy", 1, 1, 0),
        ("energy", "energy", 0, 1, 1),
    ]


def test_crawl_stores_fallback_summary_as_body_when_full_text_is_unavailable() -> None:
    from ingestion.crawler import crawl

    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "bbc_world",
            "rss_url": "https://example.test/rss",
            "affinity_tag": "mainstream",
            "country": "GLOBAL",
            "language": "en",
            "enabled": True,
        }
        summary = "Fallback RSS summary is still stored as article body."
        parsed_feed = SimpleNamespace(
            entries=[
                {
                    "title": "Inflace moves global markets",
                    "summary": summary,
                    "link": "https://example.test/articles/fallback-story",
                    "published": "2026-03-18T10:00:00+00:00",
                }
            ]
        )
        with patch("ingestion.crawler.select_feeds", return_value=[feed]), patch(
            "ingestion.crawler._fetch_feed_bytes",
            return_value=b"<rss />",
        ), patch(
            "ingestion.crawler.feedparser.parse",
            return_value=parsed_feed,
        ), patch(
            "ingestion.crawler._fetch_article_body",
            return_value=(summary, "https://example.test/articles/fallback-story"),
        ):
            crawl("inflace", country="GLOBAL", source="bbc_world")

        body = db.init.get_conn().execute(
            "SELECT body, country FROM articles WHERE outlet = 'bbc_world'"
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert body == (summary, "GLOBAL")


def test_digest_route_honors_filters_and_returns_citation_articles() -> None:
    from api.routes import summaries as summaries_route

    temp_dir = setup_temp_db()
    try:
        conn = db.init.get_conn()
        conn.execute(
            """
            INSERT INTO articles
            (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
            VALUES
            ('de-1', 'spiegel', 'German inflation story', 'DE summary', 'DE body',
             'https://example.test/de-1', 'https://example.test/de-1', 'inflace', 'DE', 'de',
             '2026-03-18T10:00:00+00:00', '2026-03-18T10:05:00+00:00'),
            ('cz-1', 'irozhlas', 'Czech inflation story', 'CZ summary', 'CZ body',
             'https://example.test/cz-1', 'https://example.test/cz-1', 'inflace', 'CZ', 'cs',
             '2026-03-18T10:00:00+00:00', '2026-03-18T10:05:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO article_topics(article_id, topic, relevance_score, matched_at)
            VALUES
            ('de-1', 'inflace', 0.91, '2026-03-18T10:05:00+00:00'),
            ('cz-1', 'inflace', 0.82, '2026-03-18T10:05:00+00:00')
            """
        )
        conn.commit()

        with patch(
            "brief.digest._call_ollama_json",
            return_value={
                "summary_headline": "Inflation pressure remains visible in Germany",
                "summary_text": "German coverage stays focused on inflation and business effects.",
                "key_points": [
                    "Spiegel provides the highest-ranked matching article.",
                    "The digest stays limited to German coverage.",
                    "The response includes article citations separately.",
                ],
            },
        ):
            result = summaries_route.get_summaries(
                topic="inflace",
                country="DE",
                source="spiegel",
                limit=5,
            )
    finally:
        cleanup_temp_db(temp_dir)

    assert result.topic == "inflace"
    assert result.requested_topic == "inflace"
    assert result.canonical_topic_id == "inflation"
    assert result.canonical_display_name == "Inflation"
    assert result.country == "DE"
    assert result.source == "spiegel"
    assert result.article_count == 1
    assert result.sources_used == ["spiegel"]
    assert result.summary_headline == "Inflation pressure remains visible in Germany"
    assert len(result.key_points) == 3
    assert [article.outlet for article in result.articles] == ["spiegel"]
    assert [article.country for article in result.articles] == ["DE"]
