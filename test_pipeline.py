#!/usr/bin/env python3
import asyncio
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import db.init

from api import pipeline as pipeline_helpers
from api import scheduler as pipeline_scheduler


ORIGINAL_DB_PATH = db.init.DB_PATH


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()


def test_run_collection_cycle_chains_extract_and_rewards() -> None:
    stub_generator = types.ModuleType("brief.generator")
    stub_generator.clear_brief_cache = Mock()
    stub_extractor = types.ModuleType("extraction.extractor")
    stub_extractor.run_extraction = Mock(return_value=2)

    with patch(
        "api.pipeline._crawl_async",
        new=AsyncMock(return_value=2),
    ) as mock_crawl, patch(
        "api.pipeline.record_recent_signal_rewards",
        return_value=2,
    ) as mock_rewards, patch.dict(
        "sys.modules",
        {
            "brief.generator": stub_generator,
            "extraction.extractor": stub_extractor,
        },
    ):
        result = asyncio.run(pipeline_helpers.run_collection_cycle("inflace"))

    assert result == {
        "inserted": 2,
        "extracted": 2,
        "rewards_recorded": 2,
        "topic": "inflace",
    }
    mock_crawl.assert_awaited_once_with("inflace")
    stub_extractor.run_extraction.assert_called_once_with(
        "inflace",
        record_bandit_reward=False,
    )
    mock_rewards.assert_called_once()
    stub_generator.clear_brief_cache.assert_called_once_with()


def test_run_collection_cycle_skips_extract_when_no_articles_are_inserted() -> None:
    with patch(
        "api.pipeline._crawl_async",
        new=AsyncMock(return_value=0),
    ) as mock_crawl, patch(
        "api.pipeline.record_recent_signal_rewards"
    ) as mock_rewards:
        result = asyncio.run(pipeline_helpers.run_collection_cycle("inflace"))

    assert result == {
        "inserted": 0,
        "extracted": 0,
        "rewards_recorded": 0,
        "topic": "inflace",
    }
    mock_crawl.assert_awaited_once_with("inflace")
    mock_rewards.assert_not_called()


def test_record_recent_signal_rewards_only_replays_fresh_articles() -> None:
    temp_dir = setup_temp_db()
    try:
        conn = db.init.get_conn()
        rows = [
            (
                "old-1",
                "https://example.test/old-1",
                "2026-03-18T11:59:00+00:00",
                "2026-03-18T11:59:30+00:00",
                json.dumps(
                    {
                        "domain": "commerce",
                        "concern_level": 0.9,
                        "purchase_intent": 0.2,
                        "avoidance_signals": 0.4,
                    }
                ),
            ),
            (
                "new-1",
                "https://example.test/new-1",
                "2026-03-18T12:00:00+00:00",
                "2026-03-18T12:00:30+00:00",
                json.dumps(
                    {
                        "domain": "commerce",
                        "concern_level": 0.7,
                        "purchase_intent": 0.1,
                        "avoidance_signals": 0.3,
                    }
                ),
            ),
        ]
        for article_id, url, fetched_at, extracted_at, raw_json in rows:
            conn.execute(
                """
                INSERT INTO articles
                (id, outlet, title, summary, url, topic, published_at, fetched_at)
                VALUES (?, 'irozhlas', 'Title', 'Summary', ?, 'inflace',
                        '2026-03-18T10:00:00+00:00', ?)
                """,
                (article_id, url, fetched_at),
            )
            conn.execute(
                """
                INSERT INTO signals
                (article_id, concern_level, purchase_intent, avoidance_signals,
                 dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
                 raw_json, extracted_at)
                VALUES (?, 0.5, 0.1, 0.2, 'fear', 0.25, 0.25, 0.25, 0.25, ?, ?)
                """,
                (article_id, raw_json, extracted_at),
            )
        conn.commit()

        recorded_calls: list[dict[str, object]] = []

        def fake_record_signal_reward(
            outlet: str,
            topic: str,
            signals: dict,
            when: str | None = None,
            feed=None,
        ) -> float:
            recorded_calls.append(
                {
                    "outlet": outlet,
                    "topic": topic,
                    "signals": signals,
                    "when": when,
                }
            )
            return 0.0

        with patch(
            "api.pipeline.record_signal_reward",
            side_effect=fake_record_signal_reward,
        ):
            recorded = pipeline_helpers.record_recent_signal_rewards(
                "inflace",
                "2026-03-18T12:00:00+00:00",
            )
    finally:
        cleanup_temp_db(temp_dir)

    assert recorded == 1
    assert recorded_calls == [
        {
            "outlet": "irozhlas",
            "topic": "inflace",
            "signals": {
                "domain": "commerce",
                "concern_level": 0.7,
                "purchase_intent": 0.1,
                "avoidance_signals": 0.3,
            },
            "when": "2026-03-18T10:00:00+00:00",
        }
    ]


def test_scheduler_starts_one_job_per_topic() -> None:
    class FakeScheduler:
        def __init__(self, daemon: bool = False):
            self.daemon = daemon
            self.jobs: list[dict[str, object]] = []
            self.started = False
            self.shutdown_wait = None

        def add_job(self, func, trigger, minutes, args, id, max_instances):
            self.jobs.append(
                {
                    "func": func,
                    "trigger": trigger,
                    "minutes": minutes,
                    "args": args,
                    "id": id,
                    "max_instances": max_instances,
                }
            )

        def start(self):
            self.started = True

        def shutdown(self, wait: bool = False):
            self.shutdown_wait = wait

    pipeline_scheduler.stop_scheduler()
    with patch.object(
        pipeline_scheduler,
        "BackgroundScheduler",
        new=FakeScheduler,
    ), patch.object(
        pipeline_scheduler,
        "SCHEDULED_TOPICS",
        ["inflace", "energie"],
    ), patch.object(
        pipeline_scheduler,
        "PIPELINE_INTERVAL_MINUTES",
        15,
    ):
        pipeline_scheduler.start_scheduler()
        scheduler = pipeline_scheduler._scheduler
        pipeline_scheduler.stop_scheduler()

    assert scheduler is not None
    assert scheduler.daemon is True
    assert scheduler.started is True
    assert [job["id"] for job in scheduler.jobs] == [
        "pipeline_inflace",
        "pipeline_energie",
    ]
    assert all(job["minutes"] == 15 for job in scheduler.jobs)
    assert scheduler.shutdown_wait is False


def test_scheduler_refreshes_profiles_after_successful_collection_cycle() -> None:
    with patch(
        "api.scheduler.run_collection_cycle_sync",
        return_value={
            "inserted": 1,
            "extracted": 1,
            "rewards_recorded": 1,
            "topic": "inflace",
        },
    ) as mock_run, patch(
        "api.scheduler.compute_segment_profiles",
    ) as mock_profiles:
        pipeline_scheduler._run_topic_pipeline("inflace")

    mock_run.assert_called_once_with("inflace")
    mock_profiles.assert_called_once_with("inflace", learn_baseline=True)
