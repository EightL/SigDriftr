#!/usr/bin/env python3
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import db.init

from ingestion.bandit import (
    get_bandit_snapshot,
    record_signal_reward,
    reward_from_signals,
    select_feeds,
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
