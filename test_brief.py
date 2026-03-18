#!/usr/bin/env python3
import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import db.init
import pytest

pydantic = pytest.importorskip("pydantic")
ValidationError = pydantic.ValidationError

from brief.generator import (
    BRIEF_CACHE_TTL,
    OLLAMA_MODEL,
    _call_ollama_json,
    clear_brief_cache,
    generate_brief,
    generate_brief_cached,
)
from brief.models import ResearchBrief
from brief.prompt import BRIEF_TEMPLATE, LOW_CONFIDENCE_WARNING
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


def test_brief_prompt_requires_english_output() -> None:
    assert BRIEF_TEMPLATE.startswith(
        "IMPORTANT: Your entire response MUST be in English."
    )


def test_call_ollama_json_uses_larger_output_budget() -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"response":"{\\"headline\\": \\"ok\\"}"}'

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _call_ollama_json("prompt")

    assert result == {"headline": "ok"}
    assert captured["payload"]["options"]["num_predict"] == 1024


def test_generate_brief_normalizes_segment_aliases() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="alias-1",
            topic="inflace",
            title="Alias test article",
            summary="Used to verify segment normalization.",
            concern=0.8,
            purchase=0.2,
            avoidance=0.3,
            frame="fear",
            seg_young_urban=0.2,
            seg_family=0.2,
            seg_senior=0.2,
            seg_b2b=0.4,
        )

        response_data = {
            "topic": "inflace",
            "headline": "B2B segment reacts most strongly to inflation",
            "narrative": "Business decision-makers show the strongest shift in purchase intent and avoidance behavior.",
            "most_affected_segment": "business_decision-makers",
            "drift_type": "frame_shift",
            "alert_level": "strong",
            "hypotheses": [
                {
                    "segment": "business decision-makers",
                    "hypothesis": "B2B buyers will defer major commitments.",
                    "signal_basis": "purchase_intent -0.36",
                    "suggested_question": "How likely are you to delay a major business purchase?",
                },
                {
                    "segment": "young urban adults",
                    "hypothesis": "Young urban adults will reduce optional spending.",
                    "signal_basis": "avoidance_signals +0.15",
                    "suggested_question": "How likely are you to avoid discretionary purchases?",
                },
                {
                    "segment": "seniors",
                    "hypothesis": "Seniors will remain cautious but stable.",
                    "signal_basis": "avoidance_signals -0.09",
                    "suggested_question": "How much have your spending habits changed?",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch("brief.generator._call_ollama_json", return_value=response_data):
            brief = generate_brief("inflace")

        assert brief.most_affected_segment == "b2b"
        assert [item.segment for item in brief.hypotheses] == [
            "b2b",
            "young_urban",
            "senior",
        ]
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_normalizes_composite_drift_type() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="drift-type-1",
            topic="inflace",
            title="Drift type alias article",
            summary="Used to verify drift type normalization.",
            concern=0.7,
            purchase=0.1,
            avoidance=0.4,
            frame="conflict",
            seg_young_urban=0.4,
            seg_family=0.2,
            seg_senior=0.2,
            seg_b2b=0.2,
        )

        response_data = {
            "topic": "inflace",
            "headline": "B2B demand weakens under conflicting inflation signals",
            "narrative": "The model returned a composite drift label that should be normalized.",
            "most_affected_segment": "b2b",
            "drift_type": "purchase_surge | frame_shift",
            "alert_level": "strong",
            "hypotheses": [
                {
                    "segment": "b2b",
                    "hypothesis": "B2B buyers will delay purchases.",
                    "signal_basis": "purchase_intent -0.36",
                    "suggested_question": "How likely are you to postpone a major business purchase?",
                },
                {
                    "segment": "family",
                    "hypothesis": "Families will stay cautious.",
                    "signal_basis": "avoidance_signals +0.03",
                    "suggested_question": "How likely are you to cut household spending?",
                },
                {
                    "segment": "young_urban",
                    "hypothesis": "Young urban consumers will reduce optional spending.",
                    "signal_basis": "avoidance_signals +0.15",
                    "suggested_question": "How likely are you to avoid discretionary purchases?",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch("brief.generator._call_ollama_json", return_value=response_data):
            brief = generate_brief("inflace")

        assert brief.drift_type == "purchase_surge"
    finally:
        cleanup_temp_db(temp_dir)


def test_research_brief_rejects_noncanonical_segment_keys() -> None:
    with pytest.raises(ValidationError):
        ResearchBrief(
            topic="inflace",
            headline="Invalid segment example",
            narrative="This should fail validation.",
            most_affected_segment="business_decision-makers",
            drift_type="mixed",
            alert_level="mild",
            hypotheses=[
                {
                    "segment": "young_urban",
                    "hypothesis": "Hypothesis one.",
                    "signal_basis": "concern_level +0.10",
                    "suggested_question": "Question one?",
                },
                {
                    "segment": "family",
                    "hypothesis": "Hypothesis two.",
                    "signal_basis": "concern_level +0.05",
                    "suggested_question": "Question two?",
                },
                {
                    "segment": "senior citizens",
                    "hypothesis": "Hypothesis three.",
                    "signal_basis": "concern_level +0.02",
                    "suggested_question": "Question three?",
                },
            ],
            generated_at="2026-03-17T00:00:00+00:00",
            model_used=OLLAMA_MODEL,
        )


def test_generate_brief_returns_fallback_when_validation_still_fails() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="fallback-1",
            topic="inflace",
            title="Fallback article",
            summary="Used to verify degraded fallback behavior.",
            concern=0.9,
            purchase=0.1,
            avoidance=0.6,
            frame="fear",
            seg_young_urban=0.1,
            seg_family=0.2,
            seg_senior=0.6,
            seg_b2b=0.1,
        )

        response_data = {
            "topic": "inflace",
            "headline": "Invalid alert level example",
            "narrative": "This payload should fail strict validation and trigger fallback.",
            "most_affected_segment": "business_decision-makers",
            "drift_type": "unmapped_drift_type",
            "alert_level": "very_strong",
            "hypotheses": [
                {
                    "segment": "business decision-makers",
                    "hypothesis": "Invalid hypothesis one.",
                    "signal_basis": "bad",
                    "suggested_question": "Q1?",
                },
                {
                    "segment": "young urban adults",
                    "hypothesis": "Invalid hypothesis two.",
                    "signal_basis": "bad",
                    "suggested_question": "Q2?",
                },
                {
                    "segment": "seniors",
                    "hypothesis": "Invalid hypothesis three.",
                    "signal_basis": "bad",
                    "suggested_question": "Q3?",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch("brief.generator._call_ollama_json", return_value=response_data):
            brief = generate_brief("inflace")

        assert "degraded" not in brief.headline.lower()
        assert brief.headline.startswith(("Young Urban", "Family", "Senior", "B2B"))
        assert brief.alert_level in ("none", "mild", "strong")
        assert brief.most_affected_segment in ("young_urban", "family", "senior", "b2b")
        assert len(brief.hypotheses) == 3
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_returns_fallback_when_ollama_json_is_invalid() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="invalid-json-1",
            topic="inflace",
            title="Invalid JSON article",
            summary="Used to verify malformed Ollama JSON fallback behavior.",
            concern=0.8,
            purchase=0.2,
            avoidance=0.5,
            frame="fear",
            seg_young_urban=0.2,
            seg_family=0.2,
            seg_senior=0.5,
            seg_b2b=0.1,
        )

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=RuntimeError("Ollama returned invalid JSON in JSON mode."),
        ):
            brief = generate_brief("inflace")

        assert "degraded" not in brief.headline.lower()
        assert brief.headline.startswith(("Young Urban", "Family", "Senior", "B2B"))
        assert brief.topic == "inflace"
        assert brief.alert_level in ("none", "mild", "strong")
        assert len(brief.hypotheses) == 3
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_prepends_low_confidence_warning() -> None:
    temp_dir = setup_temp_db()
    try:
        response_data = {
            "topic": "inflace",
            "headline": "Early signal only",
            "narrative": "Signals are still emerging across segments.",
            "most_affected_segment": "young_urban",
            "drift_type": "mixed",
            "alert_level": "none",
            "hypotheses": [
                {
                    "segment": "young_urban",
                    "hypothesis": "Young urban adults may show early movement.",
                    "signal_basis": "concern_level +0.02",
                    "suggested_question": "Has recent coverage changed your behavior?",
                },
                {
                    "segment": "family",
                    "hypothesis": "Families may show early movement.",
                    "signal_basis": "concern_level +0.01",
                    "suggested_question": "Has recent coverage changed your household behavior?",
                },
                {
                    "segment": "senior",
                    "hypothesis": "Seniors may show early movement.",
                    "signal_basis": "concern_level +0.01",
                    "suggested_question": "Has recent coverage changed your spending habits?",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch(
            "brief.generator.compute_drift",
            return_value=[
                {
                    "segment": "young_urban",
                    "drift_magnitude": 0.02,
                    "article_count": 5,
                    "confidence": 0.2,
                    "baseline_is_learned": False,
                    "baseline_sample_count": 0,
                },
                {
                    "segment": "family",
                    "drift_magnitude": 0.01,
                    "article_count": 4,
                    "confidence": 0.2,
                    "baseline_is_learned": False,
                    "baseline_sample_count": 0,
                },
                {
                    "segment": "senior",
                    "drift_magnitude": 0.01,
                    "article_count": 3,
                    "confidence": 0.2,
                    "baseline_is_learned": False,
                    "baseline_sample_count": 0,
                },
                {
                    "segment": "b2b",
                    "drift_magnitude": 0.0,
                    "article_count": 2,
                    "confidence": 0.2,
                    "baseline_is_learned": False,
                    "baseline_sample_count": 0,
                },
            ],
        ), patch("brief.generator._call_ollama_json", return_value=response_data):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert LOW_CONFIDENCE_WARNING in brief.narrative


def test_generate_brief_adds_high_confidence_qualifier() -> None:
    temp_dir = setup_temp_db()
    try:
        response_data = {
            "topic": "inflace",
            "headline": "Seniors react strongly",
            "narrative": "Coverage is driving a clear shift among seniors.",
            "most_affected_segment": "senior",
            "drift_type": "concern_spike",
            "alert_level": "strong",
            "hypotheses": [
                {
                    "segment": "senior",
                    "hypothesis": "Seniors will cut discretionary spending.",
                    "signal_basis": "concern_level +0.12",
                    "suggested_question": "How likely are you to reduce spending?",
                },
                {
                    "segment": "family",
                    "hypothesis": "Families will become more cautious.",
                    "signal_basis": "avoidance_signals +0.07",
                    "suggested_question": "How likely are you to postpone purchases?",
                },
                {
                    "segment": "young_urban",
                    "hypothesis": "Young urban adults will react less strongly.",
                    "signal_basis": "purchase_intent -0.02",
                    "suggested_question": "How likely are you to change planned purchases?",
                },
            ],
            "generated_at": "ignored",
            "model_used": "ignored",
        }

        with patch(
            "brief.generator.compute_drift",
            return_value=[
                {
                    "segment": "senior",
                    "drift_magnitude": 0.5,
                    "article_count": 60,
                    "confidence": 0.9,
                    "baseline_is_learned": True,
                    "baseline_sample_count": 20,
                },
                {
                    "segment": "young_urban",
                    "drift_magnitude": 0.1,
                    "article_count": 55,
                    "confidence": 0.9,
                    "baseline_is_learned": True,
                    "baseline_sample_count": 20,
                },
                {
                    "segment": "family",
                    "drift_magnitude": 0.08,
                    "article_count": 52,
                    "confidence": 0.9,
                    "baseline_is_learned": True,
                    "baseline_sample_count": 20,
                },
                {
                    "segment": "b2b",
                    "drift_magnitude": 0.05,
                    "article_count": 51,
                    "confidence": 0.9,
                    "baseline_is_learned": True,
                    "baseline_sample_count": 20,
                },
            ],
        ), patch("brief.generator._call_ollama_json", return_value=response_data):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert "high confidence" in brief.narrative.lower()


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
