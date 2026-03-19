#!/usr/bin/env python3
import asyncio
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import db.init
from brief.generator import OLLAMA_MODEL, clear_brief_cache, generate_brief
from delta.mapper import compute_segment_profiles
from delta.seeder import seed_baselines

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
    clear_brief_cache()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    clear_brief_cache()
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()


def insert_article_with_signal(
    article_id: str,
    topic: str,
    concern: float,
    purchase: float,
    avoidance: float,
    frame: str,
    seg_young_urban: float,
    seg_family: float,
    seg_senior: float,
    seg_b2b: float,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, 'unit-test', ?, 'Summary', ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, article_id, f"https://example.test/{article_id}", topic),
    )
    raw_json = {
        "concern_level": concern,
        "purchase_intent": purchase,
        "avoidance_signals": avoidance,
        "dominant_frame": frame,
        "seg_young_urban": seg_young_urban,
        "seg_family": seg_family,
        "seg_senior": seg_senior,
        "seg_b2b": seg_b2b,
    }
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-03-17T00:00:00+00:00')
        """,
        (
            article_id,
            concern,
            purchase,
            avoidance,
            frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            json.dumps(raw_json),
        ),
    )
    conn.commit()


def build_brief_payload(topic: str, segment: str, alert_level: str) -> dict[str, object]:
    return {
        "topic": topic,
        "headline": f"{segment.replace('_', ' ').title()} segment shows the clearest movement",
        "narrative": "Structured summary for trust-state smoke testing.",
        "most_affected_segment": segment,
        "drift_type": "concern_spike",
        "alert_level": alert_level,
        "hypotheses": [
            {
                "segment": segment,
                "hypothesis": "The leading segment will show measurable follow-up behavior change.",
                "signal_basis": "concern_level +0.12",
                "suggested_question": "How much has recent coverage changed your likely behavior?",
            },
            {
                "segment": "family",
                "hypothesis": "Family households will show directional movement if coverage persists.",
                "signal_basis": "avoidance_signals +0.05",
                "suggested_question": "How much has recent coverage changed household decisions?",
            },
            {
                "segment": "b2b",
                "hypothesis": "Business audiences will remain directionally responsive to repeated coverage.",
                "signal_basis": "purchase_intent -0.03",
                "suggested_question": "How likely are you to change a planned decision because of this topic?",
            },
        ],
        "generated_at": "ignored",
        "model_used": OLLAMA_MODEL,
    }


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
        "country": "",
        "source": "",
    }
    mock_crawl.assert_awaited_once_with("inflace", country="", source="")
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
        "country": "",
        "source": "",
    }
    mock_crawl.assert_awaited_once_with("inflace", country="", source="")
    mock_rewards.assert_not_called()


def test_trust_state_smoke_path_covers_cold_warming_and_ready() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["warming-topic", "ready-topic"])
        insert_article_with_signal(
            article_id="warming-1",
            topic="warming-topic",
            concern=0.62,
            purchase=0.18,
            avoidance=0.28,
            frame="fear",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        for index in range(60):
            insert_article_with_signal(
                article_id=f"ready-{index}",
                topic="ready-topic",
                concern=0.74,
                purchase=0.16,
                avoidance=0.35,
                frame="fear",
                seg_young_urban=0.0,
                seg_family=0.0,
                seg_senior=1.0,
                seg_b2b=0.0,
            )
        compute_segment_profiles("ready-topic", learn_baseline=True)

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                build_brief_payload("warming-topic", "young_urban", "mild"),
                build_brief_payload("ready-topic", "senior", "strong"),
            ],
        ) as mock_call:
            cold_brief = generate_brief("cold-topic")
            warming_brief = generate_brief("warming-topic")
            ready_brief = generate_brief("ready-topic")
    finally:
        cleanup_temp_db(temp_dir)

    assert cold_brief.status == "insufficient_data"
    assert cold_brief.alert_level == "none"
    assert warming_brief.status == "warming"
    assert ready_brief.status == "ready"
    assert mock_call.call_count == 2


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
