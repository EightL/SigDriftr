#!/usr/bin/env python3
import asyncio
import json
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import db.init
from brief.generator import OLLAMA_MODEL, generate_brief
from brief.models import ResearchBrief
from db_helpers import cleanup_temp_db as cleanup_temp_db_base
from db_helpers import setup_temp_db as setup_temp_db_base
from delta.mapper import compute_segment_profiles
from delta.seeder import seed_baselines

from api import pipeline as pipeline_helpers
from api import scheduler as pipeline_scheduler


RECENT_TS = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def setup_temp_db():
    return setup_temp_db_base(clear_brief_cache=True)


def cleanup_temp_db(temp_dir) -> None:
    cleanup_temp_db_base(temp_dir, clear_brief_cache=True)


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
        VALUES (?, 'unit-test', ?, 'Summary', ?, ?, ?, ?)
        """,
        (
            article_id,
            article_id,
            f"https://example.test/{article_id}",
            topic,
            RECENT_TS,
            RECENT_TS,
        ),
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            RECENT_TS,
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


def build_analyst_payload() -> dict[str, object]:
    return {
        "facts": ["The leading segment shows the clearest movement in the current window."],
        "numeric_changes": ["concern_level +0.12"],
        "cited_clusters": ["track-1"],
        "cited_articles": ["article-1"],
        "evidence_gaps": [],
    }


def build_explainer_payload(segment: str) -> dict[str, object]:
    return {
        "what_changed": f"{segment.replace('_', ' ').title()} is seeing the strongest recent shift.",
        "for_whom": f"{segment.replace('_', ' ').title()} is the primary affected segment.",
        "uncertainty_and_caveats": [],
    }


def build_crawl_report(inserted: int) -> types.SimpleNamespace:
    feed_stats = [
        {
            "outlet": "irozhlas",
            "country": "CZ",
            "language": "cs",
            "selected": True,
            "fetch_success": True,
            "entries_seen": inserted,
            "candidates": inserted,
            "accepted": inserted,
            "inserted": inserted,
            "duplicates": 0,
            "avg_relevance_score": 0.9 if inserted else 0.0,
            "reward": 0.0,
            "error_message": None,
        }
    ]
    return types.SimpleNamespace(
        run_id="collect-test",
        topic="inflace",
        country="",
        source="",
        collection_mode="bandit",
        reward_mode="yield",
        eligible_feeds=["irozhlas"],
        selected_feeds=["irozhlas"],
        inserted=inserted,
        accepted=inserted,
        duplicates=0,
        to_dict=lambda: {"feed_stats": feed_stats},
    )


def test_run_collection_cycle_chains_extract_and_rewards() -> None:
    stub_generator = types.ModuleType("brief.generator")
    stub_generator.clear_brief_cache = Mock()
    stub_extractor = types.ModuleType("extraction.extractor")
    stub_extractor.run_extraction = Mock(return_value=2)

    with patch(
        "api.pipeline._crawl_async_report",
        new=AsyncMock(return_value=build_crawl_report(2)),
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
        "rewards_recorded": 0,
        "topic": "inflace",
        "canonical_topic_id": "inflation",
        "country": "",
        "source": "",
        "collection_mode": "bandit",
        "reward_mode": "yield",
        "run_id": "collect-test",
        "eligible_feeds": ["irozhlas"],
        "selected_feeds": ["irozhlas"],
        "accepted": 2,
        "duplicates": 0,
        "feed_stats": build_crawl_report(2).to_dict()["feed_stats"],
    }
    mock_crawl.assert_awaited_once_with(
        "inflace",
        country="",
        source="",
        collection_mode="bandit",
        reward_mode="yield",
    )
    stub_extractor.run_extraction.assert_called_once_with(
        "inflace",
        record_bandit_reward=False,
    )
    mock_rewards.assert_not_called()
    stub_generator.clear_brief_cache.assert_called_once_with()


def test_run_collection_cycle_skips_extract_when_no_articles_are_inserted() -> None:
    with patch(
        "api.pipeline._crawl_async_report",
        new=AsyncMock(return_value=build_crawl_report(0)),
    ) as mock_crawl, patch(
        "api.pipeline.record_recent_signal_rewards"
    ) as mock_rewards:
        result = asyncio.run(pipeline_helpers.run_collection_cycle("inflace"))

    assert result == {
        "inserted": 0,
        "extracted": 0,
        "rewards_recorded": 0,
        "topic": "inflace",
        "canonical_topic_id": "inflation",
        "country": "",
        "source": "",
        "collection_mode": "bandit",
        "reward_mode": "yield",
        "run_id": "collect-test",
        "eligible_feeds": ["irozhlas"],
        "selected_feeds": ["irozhlas"],
        "accepted": 0,
        "duplicates": 0,
        "feed_stats": build_crawl_report(0).to_dict()["feed_stats"],
    }
    mock_crawl.assert_awaited_once_with(
        "inflace",
        country="",
        source="",
        collection_mode="bandit",
        reward_mode="yield",
    )
    mock_rewards.assert_not_called()


def test_run_full_pipeline_includes_canonical_topic_scope() -> None:
    temp_dir = setup_temp_db()
    brief = ResearchBrief(
        **{
            **build_brief_payload("inflace", "family", "mild"),
            "status": "warming",
        }
    )
    try:
        with patch(
            "api.pipeline.run_collection_cycle",
            new=AsyncMock(
                return_value={
                    "inserted": 0,
                    "extracted": 0,
                    "rewards_recorded": 0,
                    "topic": "inflace",
                    "canonical_topic_id": "inflation",
                    "country": "",
                    "source": "",
                    "collection_mode": "bandit",
                    "reward_mode": "yield",
                    "run_id": "collect-test",
                    "eligible_feeds": [],
                    "selected_feeds": [],
                    "accepted": 0,
                    "duplicates": 0,
                    "feed_stats": [],
                },
            ),
        ), patch(
            "extraction.embedding_service.embed_pending_articles",
            return_value={
                "model_name": "test-model",
                "embedding_dim": 3,
                "selected": 0,
                "embedded": 0,
                "already_current": 0,
                "retried_failed": 0,
                "stale_reembedded": 0,
                "failed": 0,
                "duration_s": 0.0,
            },
        ), patch(
            "clustering.clustering_service.run_clustering",
            return_value={
                "run_id": None,
                "topic": "inflace",
                "country": "",
                "source": "",
                "language": None,
                "window_start": RECENT_TS,
                "window_end": RECENT_TS,
                "status": "skipped_small_sample",
                "n_articles": 0,
                "n_clusters": 0,
                "n_noise": 0,
                "model_name": "test-model",
                "model_version": None,
                "umap_n_components": 0,
                "umap_n_neighbors": 0,
                "hdbscan_min_cluster_size": 3,
                "hdbscan_min_samples": 0,
                "duration_s": 0.0,
            },
        ), patch(
            "api.pipeline.generate_brief_cached",
            return_value=brief,
        ), patch(
            "api.pipeline.get_scope_counts",
            return_value={"article_count": 0, "signal_count": 0, "embedding_count": 0},
        ):
            result = asyncio.run(pipeline_helpers.run_full_pipeline("inflace"))
    finally:
        cleanup_temp_db(temp_dir)

    assert result["scope"]["canonical_topic_id"] == "inflation"
    assert result["scope"]["canonical_display_name"] == "Inflation"
    assert result["brief_status"] == "warming"


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
                build_analyst_payload(),
                build_explainer_payload("young_urban"),
                build_brief_payload("warming-topic", "young_urban", "mild"),
                build_analyst_payload(),
                build_explainer_payload("senior"),
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
    assert mock_call.call_count == 6


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
        conn.execute(
            """
            INSERT INTO article_topics
            (article_id, topic, raw_topic, canonical_topic_id, relevance_score, matched_at)
            VALUES ('new-1', 'inflation', 'inflation', 'inflation', 1.0,
                    '2026-03-18T12:01:00+00:00')
            """
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
            "topic": "inflation",
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
