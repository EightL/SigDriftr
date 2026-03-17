#!/usr/bin/env python3
import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import db.init
import pytest

pytest.importorskip("pydantic")

from brief.generator import (
    BRIEF_CACHE_TTL,
    OLLAMA_MODEL,
    clear_brief_cache,
    generate_brief,
    generate_brief_cached,
)
from brief.models import ResearchBrief
from delta.seeder import seed_baselines


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = db.init.Path(temp_dir.name) / "sigdriftr.db"
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
    temp_dir.cleanup()


def sample_brief(topic: str = "inflace", *, headline: str | None = None) -> ResearchBrief:
    return ResearchBrief(
        topic=topic,
        headline=headline or "Seniors show rising concern in inflation coverage",
        narrative="Inflation stories are amplifying concern among seniors, while other audiences remain less affected.",
        most_affected_segment="senior",
        drift_type="concern_spike",
        alert_level="mild",
        hypotheses=[
            {
                "segment": "senior",
                "hypothesis": "Seniors will reduce discretionary spending after repeated inflation coverage.",
                "signal_basis": "concern_level +0.12",
                "suggested_question": "Inflation news makes me less willing to spend on non-essentials.",
            },
            {
                "segment": "family",
                "hypothesis": "Families will report more cautious budgeting after inflation news.",
                "signal_basis": "avoidance_signals +0.07",
                "suggested_question": "Inflation coverage makes me more careful with household spending.",
            },
            {
                "segment": "young_urban",
                "hypothesis": "Young urban adults will show weaker concern shifts than seniors.",
                "signal_basis": "purchase_intent -0.03",
                "suggested_question": "Inflation coverage changes how likely I am to make planned purchases.",
            },
        ],
        generated_at="2026-03-17T00:00:00+00:00",
        model_used=OLLAMA_MODEL,
    )


def insert_article_with_signal(
    article_id: str,
    topic: str,
    title: str,
    summary: str,
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
        VALUES (?, 'unit-test', ?, ?, ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, title, summary, f"https://example.test/{article_id}", topic),
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


def test_generate_brief_returns_valid_structured_brief() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="senior-1",
            topic="inflace",
            title="Inflation fears hit pensioners",
            summary="Older households are cutting spending as prices keep rising.",
            concern=0.92,
            purchase=0.08,
            avoidance=0.61,
            frame="fear",
            seg_young_urban=0.05,
            seg_family=0.10,
            seg_senior=0.85,
            seg_b2b=0.00,
        )
        insert_article_with_signal(
            article_id="family-1",
            topic="inflace",
            title="Families delay big purchases",
            summary="Parents are postponing appliance upgrades and travel plans.",
            concern=0.74,
            purchase=0.18,
            avoidance=0.49,
            frame="fear",
            seg_young_urban=0.10,
            seg_family=0.75,
            seg_senior=0.15,
            seg_b2b=0.00,
        )

        response_data = {
            "topic": "inflace",
            "headline": "Seniors show rising concern as inflation coverage intensifies",
            "narrative": "Inflation coverage is skewing more fear-based for seniors, with concern and avoidance both climbing. Family households also show caution, but the strongest reaction is still concentrated among older audiences.",
            "most_affected_segment": "senior",
            "drift_type": "concern_spike",
            "alert_level": "strong",
            "hypotheses": [
                {
                    "segment": "senior",
                    "hypothesis": "Seniors exposed to fear-framed inflation coverage will report lower willingness to make discretionary purchases over the next month.",
                    "signal_basis": "concern_level +0.18",
                    "suggested_question": "I expect to postpone non-essential purchases because of inflation news.",
                },
                {
                    "segment": "family",
                    "hypothesis": "Family households will report increased budgeting behavior after repeated inflation stories tied to household essentials.",
                    "signal_basis": "avoidance_signals +0.11",
                    "suggested_question": "Inflation coverage has made me more cautious about household spending.",
                },
                {
                    "segment": "young_urban",
                    "hypothesis": "Young urban adults will maintain attention to inflation news but show a smaller reduction in purchase intent than seniors.",
                    "signal_basis": "purchase_intent -0.05",
                    "suggested_question": "Inflation news has changed my willingness to make planned purchases.",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch("brief.generator._call_ollama_json", return_value=response_data) as mock_call:
            brief = generate_brief("inflace")

        assert isinstance(brief, ResearchBrief)
        assert brief.topic == "inflace"
        assert brief.most_affected_segment == "senior"
        assert brief.drift_type == "concern_spike"
        assert brief.alert_level == "strong"
        assert brief.model_used == OLLAMA_MODEL
        assert len(brief.hypotheses) == 3
        assert brief.hypotheses[0].segment == "senior"
        mock_call.assert_called_once()
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_cached_reuses_recent_result() -> None:
    temp_dir = setup_temp_db()
    try:
        expected = sample_brief()
        with patch("brief.generator.generate_brief", return_value=expected) as mock_generate:
            first = generate_brief_cached("inflace")
            second = generate_brief_cached("inflace")

        assert first == expected
        assert second == expected
        mock_generate.assert_called_once_with("inflace")
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_cached_refreshes_stale_result() -> None:
    temp_dir = setup_temp_db()
    try:
        import brief.generator as brief_generator

        stale = sample_brief(headline="Stale headline")
        fresh = sample_brief(headline="Fresh headline")

        with brief_generator._cache_lock:
            brief_generator._brief_cache["inflace"] = (
                stale,
                datetime.now(timezone.utc) - BRIEF_CACHE_TTL - timedelta(seconds=1),
            )

        with patch("brief.generator.generate_brief", return_value=fresh) as mock_generate:
            result = generate_brief_cached("inflace")

        assert result == fresh
        mock_generate.assert_called_once_with("inflace")
    finally:
        cleanup_temp_db(temp_dir)


def test_brief_route_returns_cached_generator_result() -> None:
    pytest.importorskip("fastapi")
    from api.routes import brief as brief_route

    expected = sample_brief()

    with patch("api.routes.brief.generate_brief_cached", return_value=expected):
        result = brief_route.get_brief("inflace")

    assert result == expected


def test_health_route_reports_db_and_ollama_status() -> None:
    pytest.importorskip("fastapi")
    from api.routes import health as health_route

    temp_dir = setup_temp_db()
    try:
        insert_article_with_signal(
            article_id="health-1",
            topic="inflace",
            title="Health route article",
            summary="Used to confirm row counts.",
            concern=0.4,
            purchase=0.3,
            avoidance=0.2,
            frame="neutral",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"models":[{"name":"qwen2.5:7b-instruct"}]}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = health_route.health()

        assert result["status"] == "ok"
        assert result["db"] == "ok"
        assert result["articles"] == 1
        assert result["signals"] == 1
        assert result["ollama"] == "ok"
        assert result["ollama_models"] == ["qwen2.5:7b-instruct"]
    finally:
        cleanup_temp_db(temp_dir)


def test_main_registers_brief_and_health_routes() -> None:
    pytest.importorskip("fastapi")
    import main

    route_methods = {route.path: route.methods for route in main.app.routes}
    assert "GET" in route_methods["/brief/{topic}"]
    assert "/health" in route_methods
    assert "/brief/{topic}/status" not in route_methods
    assert "/brief/{topic}/result" not in route_methods
