#!/usr/bin/env python3
import asyncio
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import db.init
import pytest

from ingestion.bandit import (
    get_bandit_snapshot,
    record_signal_reward,
    record_yield_reward,
    reward_from_signals,
    reward_from_yield,
    select_feeds,
    warm_start_from_history,
)
from ingestion.crawler import crawl


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = db.init.Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    temp_dir.cleanup()


def insert_article(
    article_id: str,
    topic: str,
    outlet: str,
    title: str,
    summary: str = "Summary",
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, outlet, title, summary, f"https://example.test/{article_id}", topic),
    )
    conn.commit()


def test_reward_from_signals_uses_domain_weights() -> None:
    reward = reward_from_signals(
        "politika",
        {
            "domain": "civic",
            "concern_level": 0.8,
            "purchase_intent": 1.0,
            "avoidance_signals": 0.5,
        },
    )

    assert reward == 0.68


def test_reward_from_yield_uses_non_llm_collection_features() -> None:
    reward = reward_from_yield(
        accepted_count=4,
        avg_relevance_score=0.8,
        duplicate_count=1,
        fetch_success=True,
    )

    assert reward == 0.52
    assert reward_from_yield(
        accepted_count=4,
        avg_relevance_score=0.8,
        fetch_success=False,
    ) == 0.0


def test_select_feeds_shifts_toward_high_reward_sources() -> None:
    temp_dir = setup_temp_db()
    try:
        feeds = [
            {"outlet": "alpha", "rss_url": "https://alpha.test/rss", "affinity_tag": "mainstream"},
            {"outlet": "beta", "rss_url": "https://beta.test/rss", "affinity_tag": "b2b"},
            {"outlet": "gamma", "rss_url": "https://gamma.test/rss", "affinity_tag": "family"},
        ]
        when = datetime(2026, 3, 18, 9, 0, tzinfo=timezone.utc)

        for _ in range(3):
            record_signal_reward(
                "alpha",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.9,
                    "purchase_intent": 0.9,
                    "avoidance_signals": 0.7,
                },
                when=when,
                feed=feeds[0],
            )
            record_signal_reward(
                "beta",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.2,
                    "purchase_intent": 0.1,
                    "avoidance_signals": 0.1,
                },
                when=when,
                feed=feeds[1],
            )
            record_signal_reward(
                "gamma",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.1,
                    "purchase_intent": 0.0,
                    "avoidance_signals": 0.0,
                },
                when=when,
                feed=feeds[2],
            )

        selected = select_feeds("inflace", now=when, k=2, feeds=feeds)
    finally:
        cleanup_temp_db(temp_dir)

    assert [feed["outlet"] for feed in selected] == ["alpha", "beta"]
    assert selected[0]["outlet"] == "alpha"


def test_select_feeds_orders_visited_arms_by_descending_score() -> None:
    temp_dir = setup_temp_db()
    try:
        feeds = [
            {"outlet": "alpha", "rss_url": "https://alpha.test/rss", "affinity_tag": "mainstream"},
            {"outlet": "beta", "rss_url": "https://beta.test/rss", "affinity_tag": "b2b"},
            {"outlet": "gamma", "rss_url": "https://gamma.test/rss", "affinity_tag": "family"},
        ]
        when = datetime(2026, 3, 18, 9, 0, tzinfo=timezone.utc)

        for _ in range(3):
            record_signal_reward(
                "alpha",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.9,
                    "purchase_intent": 0.9,
                    "avoidance_signals": 0.7,
                },
                when=when,
                feed=feeds[0],
            )
            record_signal_reward(
                "beta",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.1,
                    "purchase_intent": 0.0,
                    "avoidance_signals": 0.0,
                },
                when=when,
                feed=feeds[1],
            )
            record_signal_reward(
                "gamma",
                "inflace",
                {
                    "domain": "commerce",
                    "concern_level": 0.2,
                    "purchase_intent": 0.1,
                    "avoidance_signals": 0.1,
                },
                when=when,
                feed=feeds[2],
            )

        selected = select_feeds("inflace", now=when, k=2, feeds=feeds)
    finally:
        cleanup_temp_db(temp_dir)

    assert [feed["outlet"] for feed in selected] == ["alpha", "gamma"]


def test_record_signal_reward_accepts_rfc2822_timestamp() -> None:
    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "alpha",
            "rss_url": "https://alpha.test/rss",
            "affinity_tag": "mainstream",
        }

        reward = record_signal_reward(
            "alpha",
            "inflace",
            {
                "domain": "commerce",
                "concern_level": 0.9,
                "purchase_intent": 0.5,
                "avoidance_signals": 0.2,
            },
            when="Wed, 18 Mar 2026 14:36:08 +0100",
            feed=feed,
        )
        snapshot = get_bandit_snapshot("alpha")
    finally:
        cleanup_temp_db(temp_dir)

    assert reward == 0.6
    assert snapshot["pulls"] == 1
    assert snapshot["total_reward"] == 0.6


def test_run_extraction_updates_bandit_using_signal_rewards() -> None:
    pytest.importorskip("tenacity")
    from extraction.extractor import run_extraction

    temp_dir = setup_temp_db()
    try:
        feeds = [
            {"outlet": "irozhlas", "rss_url": "https://www.irozhlas.cz/rss/irozhlas", "affinity_tag": "mainstream"},
            {"outlet": "idnes", "rss_url": "https://servis.idnes.cz/rss.aspx", "affinity_tag": "mainstream"},
        ]
        for index in range(3):
            insert_article(
                article_id=f"irozhlas-{index}",
                topic="inflace",
                outlet="irozhlas",
                title=f"irozhlas-{index}",
            )
            insert_article(
                article_id=f"idnes-{index}",
                topic="inflace",
                outlet="idnes",
                title=f"idnes-{index}",
            )

        def fake_extract_signals(
            title: str,
            summary: str,
            affinity_tag: str = "mainstream",
            topic: str = "",
        ) -> dict:
            if title.startswith("irozhlas"):
                concern_level = 0.6
                avoidance_signals = 0.2
            elif title.endswith("-0"):
                concern_level = 0.06
                avoidance_signals = 0.0
            else:
                concern_level = 0.0
                avoidance_signals = 0.0

            return {
                "concern_level": concern_level,
                "purchase_intent": 0.0,
                "avoidance_signals": avoidance_signals,
                "dominant_frame": "fear",
                "seg_young_urban": 0.25,
                "seg_family": 0.25,
                "seg_senior": 0.25,
                "seg_b2b": 0.25,
                "domain": "commerce",
                "irrelevant_fields": [],
            }

        with patch(
            "extraction.extractor.extract_signals",
            side_effect=fake_extract_signals,
        ), patch("extraction.extractor.extract_entities", return_value=[]):
            processed = run_extraction("inflace", record_bandit_reward=True)

        irozhlas_snapshot = get_bandit_snapshot("irozhlas")
        idnes_snapshot = get_bandit_snapshot("idnes")
        selected = select_feeds(
            "inflace",
            now=datetime.now(timezone.utc),
            k=2,
            feeds=feeds,
        )
    finally:
        cleanup_temp_db(temp_dir)

    assert processed == 6
    assert irozhlas_snapshot["pulls"] == 3
    assert idnes_snapshot["pulls"] == 3
    assert irozhlas_snapshot["total_reward"] == 0.84
    assert idnes_snapshot["total_reward"] == 0.024
    assert [feed["outlet"] for feed in selected] == ["irozhlas", "idnes"]


def test_run_extraction_does_not_update_bandit_by_default() -> None:
    pytest.importorskip("tenacity")
    from extraction.extractor import run_extraction

    temp_dir = setup_temp_db()
    try:
        insert_article(
            article_id="irozhlas-default",
            topic="inflace",
            outlet="irozhlas",
            title="irozhlas-default",
        )

        with patch(
            "extraction.extractor.extract_signals",
            return_value={
                "concern_level": 0.9,
                "purchase_intent": 0.5,
                "avoidance_signals": 0.2,
                "dominant_frame": "fear",
                "seg_young_urban": 0.25,
                "seg_family": 0.25,
                "seg_senior": 0.25,
                "seg_b2b": 0.25,
                "domain": "commerce",
            },
        ), patch("extraction.extractor.extract_entities", return_value=[]):
            processed = run_extraction("inflace")

        snapshot = get_bandit_snapshot("irozhlas")
    finally:
        cleanup_temp_db(temp_dir)

    assert processed == 1
    assert snapshot["pulls"] == 0


def test_record_yield_reward_updates_bandit_without_signals() -> None:
    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "alpha",
            "rss_url": "https://alpha.test/rss",
            "affinity_tag": "mainstream",
        }
        reward = record_yield_reward(
            "alpha",
            "inflace",
            accepted_count=8,
            avg_relevance_score=0.9,
            duplicate_count=0,
            feed=feed,
        )
        snapshot = get_bandit_snapshot("alpha")
    finally:
        cleanup_temp_db(temp_dir)

    assert reward == 0.97
    assert snapshot["pulls"] == 1
    assert snapshot["total_reward"] == 0.97


def test_warm_start_from_history_defaults_to_yield_rewards() -> None:
    temp_dir = setup_temp_db()
    try:
        conn = db.init.get_conn()
        conn.execute(
            """
            INSERT INTO collection_runs
            (run_id, topic, canonical_topic_id, country, source, collection_mode,
             reward_mode, eligible_feeds, selected_feeds, inserted, accepted,
             duplicates, started_at, completed_at, duration_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-yield-1",
                "inflace",
                "inflation",
                "CZ",
                "",
                "bandit",
                "yield",
                1,
                1,
                4,
                4,
                1,
                "2026-05-30T10:00:00+00:00",
                "2026-05-30T10:01:00+00:00",
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO collection_feed_stats
            (run_id, outlet, country, language, selected, fetch_success,
             entries_seen, candidates, accepted, inserted, duplicates,
             avg_relevance_score, reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-yield-1",
                "irozhlas",
                "CZ",
                "cs",
                1,
                1,
                8,
                4,
                4,
                4,
                1,
                0.8,
                0.0,
            ),
        )
        conn.commit()

        updated = warm_start_from_history("inflace")
        snapshot = get_bandit_snapshot("irozhlas")
    finally:
        cleanup_temp_db(temp_dir)

    assert updated == 1
    assert snapshot["pulls"] == 1
    assert snapshot["total_reward"] == reward_from_yield(
        accepted_count=4,
        avg_relevance_score=0.8,
        duplicate_count=1,
    )


def test_crawl_records_zero_reward_for_selected_feed_with_no_relevant_matches() -> None:
    temp_dir = setup_temp_db()
    try:
        feed = {"outlet": "irozhlas", "rss_url": "https://example.test/rss", "affinity_tag": "mainstream"}
        parsed = SimpleNamespace(
            entries=[
                {
                    "title": "Unrelated article",
                    "summary": "No topic hit here.",
                    "link": "https://example.test/a1",
                }
            ]
        )

        with patch("ingestion.crawler.select_feeds", return_value=[feed]), patch(
            "ingestion.crawler._fetch_feed_bytes",
            return_value=b"<rss />",
        ), patch("ingestion.crawler.feedparser.parse", return_value=parsed), patch(
            "ingestion.crawler._is_relevant",
            return_value=False,
        ):
            inserted = crawl("inflace")

        snapshot = get_bandit_snapshot("irozhlas")
    finally:
        cleanup_temp_db(temp_dir)

    assert inserted == 0
    assert snapshot["pulls"] == 1
    assert snapshot["total_reward"] == 0.0


def test_crawl_raises_clear_error_inside_active_event_loop() -> None:
    async def run_inside_loop() -> None:
        try:
            crawl("inflace")
        except RuntimeError as exc:
            assert "await _crawl_async(topic) instead" in str(exc)
        else:
            raise AssertionError("Expected crawl() to reject active event loops.")

    asyncio.run(run_inside_loop())
