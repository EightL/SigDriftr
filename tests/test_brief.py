#!/usr/bin/env python3
import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import db.init
import pytest

pydantic = pytest.importorskip("pydantic")
ValidationError = pydantic.ValidationError

from brief.generator import (
    BRIEF_CACHE_TTL,
    OLLAMA_MODEL,
    _select_cluster_observations,
    _call_ollama_json,
    clear_brief_cache,
    generate_brief,
    generate_brief_cached,
)
from brief.models import ResearchBrief
from brief.prompt import BRIEF_TEMPLATE, LOW_CONFIDENCE_WARNING, build_context_block
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
        status="ready",
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


def sample_cluster_drift_payload(
    *,
    topic: str = "inflace",
    country: str = "",
    source: str = "",
    language: str | None = "cs",
    top_direction: str = "rising",
) -> dict[str, object]:
    return {
        "topic": topic,
        "country": country,
        "source": source,
        "language": language,
        "run_id": "run-cluster-1",
        "computed_at": "2026-03-19T09:30:00+00:00",
        "segments": [
            {
                "segment": "senior",
                "topic": topic,
                "article_count": 12,
                "has_data": True,
                "current": {
                    "concern_level": 0.78,
                    "purchase_intent": 0.14,
                    "avoidance_signals": 0.52,
                },
                "baseline": {
                    "concern_level": 0.58,
                    "purchase_intent": 0.22,
                    "avoidance_signals": 0.36,
                },
                "deltas": {
                    "concern_level": 0.20,
                    "purchase_intent": -0.08,
                    "avoidance_signals": 0.16,
                },
                "drift_magnitude": 0.41,
                "frame_shift": True,
                "alert_level": "strong",
                "dominant_frame": "fear",
                "baseline_frame": "neutral",
                "confidence": 0.82,
                "baseline_is_learned": True,
                "baseline_sample_count": 35,
                "baseline_age_days": 2,
                "status": "ready",
                "domain": "commerce",
                "relevant_fields": [
                    "concern_level",
                    "purchase_intent",
                    "avoidance_signals",
                ],
                "direction": top_direction,
                "centroid_shift": 0.19,
                "new_cluster_weight": 0.15,
                "tracked_cluster_count": 2,
                "matched_cluster_count": 1,
                "new_cluster_count": 1,
                "missing_cluster_count": 0,
            },
            {
                "segment": "family",
                "topic": topic,
                "article_count": 9,
                "has_data": True,
                "current": {
                    "concern_level": 0.63,
                    "purchase_intent": 0.19,
                    "avoidance_signals": 0.37,
                },
                "baseline": {
                    "concern_level": 0.54,
                    "purchase_intent": 0.21,
                    "avoidance_signals": 0.31,
                },
                "deltas": {
                    "concern_level": 0.09,
                    "purchase_intent": -0.02,
                    "avoidance_signals": 0.06,
                },
                "drift_magnitude": 0.18,
                "frame_shift": False,
                "alert_level": "mild",
                "dominant_frame": "fear",
                "baseline_frame": "fear",
                "confidence": 0.71,
                "baseline_is_learned": True,
                "baseline_sample_count": 29,
                "baseline_age_days": 2,
                "status": "ready",
                "domain": "commerce",
                "relevant_fields": [
                    "concern_level",
                    "purchase_intent",
                    "avoidance_signals",
                ],
                "direction": "rising",
                "centroid_shift": 0.11,
                "new_cluster_weight": 0.08,
                "tracked_cluster_count": 1,
                "matched_cluster_count": 1,
                "new_cluster_count": 0,
                "missing_cluster_count": 0,
            },
            {
                "segment": "young_urban",
                "topic": topic,
                "article_count": 6,
                "has_data": True,
                "current": {
                    "concern_level": 0.47,
                    "purchase_intent": 0.34,
                    "avoidance_signals": 0.21,
                },
                "baseline": {
                    "concern_level": 0.44,
                    "purchase_intent": 0.37,
                    "avoidance_signals": 0.20,
                },
                "deltas": {
                    "concern_level": 0.03,
                    "purchase_intent": -0.03,
                    "avoidance_signals": 0.01,
                },
                "drift_magnitude": 0.06,
                "frame_shift": False,
                "alert_level": "none",
                "dominant_frame": "neutral",
                "baseline_frame": "neutral",
                "confidence": 0.62,
                "baseline_is_learned": True,
                "baseline_sample_count": 24,
                "baseline_age_days": 3,
                "status": "ready",
                "domain": "commerce",
                "relevant_fields": [
                    "concern_level",
                    "purchase_intent",
                    "avoidance_signals",
                ],
                "direction": "stable",
                "centroid_shift": 0.04,
                "new_cluster_weight": 0.0,
                "tracked_cluster_count": 1,
                "matched_cluster_count": 1,
                "new_cluster_count": 0,
                "missing_cluster_count": 0,
            },
            {
                "segment": "b2b",
                "topic": topic,
                "article_count": 3,
                "has_data": True,
                "current": {
                    "concern_level": 0.41,
                    "purchase_intent": 0.29,
                    "avoidance_signals": 0.24,
                },
                "baseline": {
                    "concern_level": 0.40,
                    "purchase_intent": 0.31,
                    "avoidance_signals": 0.22,
                },
                "deltas": {
                    "concern_level": 0.01,
                    "purchase_intent": -0.02,
                    "avoidance_signals": 0.02,
                },
                "drift_magnitude": 0.04,
                "frame_shift": False,
                "alert_level": "none",
                "dominant_frame": "neutral",
                "baseline_frame": "neutral",
                "confidence": 0.56,
                "baseline_is_learned": True,
                "baseline_sample_count": 18,
                "baseline_age_days": 4,
                "status": "warming",
                "domain": "commerce",
                "relevant_fields": [
                    "concern_level",
                    "purchase_intent",
                    "avoidance_signals",
                ],
                "direction": "stable",
                "centroid_shift": 0.03,
                "new_cluster_weight": 0.0,
                "tracked_cluster_count": 1,
                "matched_cluster_count": 1,
                "new_cluster_count": 0,
                "missing_cluster_count": 0,
            },
        ],
        "clusters": [
            {
                "track_id": "track-1",
                "cluster_id": 101,
                "cluster_label": 0,
                "topic_label": "Pension cost pressure",
                "baseline_topic_label": "Retirement budgeting",
                "match_type": "matched",
                "direction": "rising",
                "centroid_distance": 0.14,
                "segment_vector_distance": 0.09,
                "signal_drift": 0.26,
                "drift_magnitude": 0.31,
                "alert_level": "strong",
                "confidence": 0.81,
                "member_count": 6,
                "mean_membership_strength": 0.94,
                "current": {
                    "concern_level": 0.78,
                    "purchase_intent": 0.14,
                    "avoidance_signals": 0.52,
                },
                "baseline": {
                    "concern_level": 0.58,
                    "purchase_intent": 0.22,
                    "avoidance_signals": 0.36,
                },
                "deltas": {
                    "concern_level": 0.20,
                    "purchase_intent": -0.08,
                    "avoidance_signals": 0.16,
                },
                "dominant_frame": "fear",
                "baseline_frame": "neutral",
                "frame_shift": True,
            },
            {
                "track_id": "track-2",
                "cluster_id": 102,
                "cluster_label": 1,
                "topic_label": "Household restraint",
                "baseline_topic_label": "Household restraint",
                "match_type": "new",
                "direction": "emerging",
                "centroid_distance": 0.0,
                "segment_vector_distance": 0.0,
                "signal_drift": 0.0,
                "drift_magnitude": 0.0,
                "alert_level": "none",
                "confidence": 0.67,
                "member_count": 4,
                "mean_membership_strength": 0.91,
                "current": {
                    "concern_level": 0.64,
                    "purchase_intent": 0.18,
                    "avoidance_signals": 0.35,
                },
                "baseline": None,
                "deltas": {
                    "concern_level": 0.0,
                    "purchase_intent": 0.0,
                    "avoidance_signals": 0.0,
                },
                "dominant_frame": "fear",
                "baseline_frame": None,
                "frame_shift": False,
            },
            {
                "track_id": "track-3",
                "cluster_id": 103,
                "cluster_label": 2,
                "topic_label": "Urban cost fatigue",
                "baseline_topic_label": "Urban cost fatigue",
                "match_type": "matched",
                "direction": "rising",
                "centroid_distance": 0.08,
                "segment_vector_distance": 0.06,
                "signal_drift": 0.18,
                "drift_magnitude": 0.22,
                "alert_level": "mild",
                "confidence": 0.7,
                "member_count": 5,
                "mean_membership_strength": 0.89,
                "current": {
                    "concern_level": 0.52,
                    "purchase_intent": 0.29,
                    "avoidance_signals": 0.23,
                },
                "baseline": {
                    "concern_level": 0.44,
                    "purchase_intent": 0.35,
                    "avoidance_signals": 0.19,
                },
                "deltas": {
                    "concern_level": 0.08,
                    "purchase_intent": -0.06,
                    "avoidance_signals": 0.04,
                },
                "dominant_frame": "conflict",
                "baseline_frame": "neutral",
                "frame_shift": True,
            },
            {
                "track_id": "track-4",
                "cluster_id": None,
                "cluster_label": None,
                "topic_label": "Discount rebound",
                "baseline_topic_label": "Discount rebound",
                "match_type": "missing",
                "direction": "fading",
                "centroid_distance": 0.0,
                "segment_vector_distance": 0.0,
                "signal_drift": 0.0,
                "drift_magnitude": 0.0,
                "alert_level": "none",
                "confidence": 0.6,
                "member_count": 0,
                "mean_membership_strength": 0.88,
                "current": {
                    "concern_level": 0.0,
                    "purchase_intent": 0.0,
                    "avoidance_signals": 0.0,
                },
                "baseline": {
                    "concern_level": 0.49,
                    "purchase_intent": 0.42,
                    "avoidance_signals": 0.17,
                },
                "deltas": {
                    "concern_level": 0.0,
                    "purchase_intent": 0.0,
                    "avoidance_signals": 0.0,
                },
                "dominant_frame": "neutral",
                "baseline_frame": "opportunity",
                "frame_shift": False,
            },
        ],
    }


def sample_analyst_output() -> dict[str, object]:
    return {
        "facts": [
            "Senior coverage is carrying the largest calibrated drift in the current window.",
            "Fear-framed reporting is contributing to higher concern and avoidance among seniors.",
        ],
        "numeric_changes": [
            "senior concern_level +0.20",
            "senior avoidance_signals +0.16",
        ],
        "cited_clusters": ["track-1", "track-3"],
        "cited_articles": ["senior-1", "family-1"],
        "evidence_gaps": ["Cluster-level exemplars are limited for some secondary segments."],
    }


def sample_explainer_output() -> dict[str, object]:
    return {
        "what_changed": "Inflation coverage is concentrating the strongest movement in senior-focused narratives, with fear and avoidance rising the fastest there.",
        "for_whom": "Seniors are the clearest leading segment, while families show a milder secondary shift and other segments remain comparatively stable.",
        "uncertainty_and_caveats": [
            "Some secondary clusters have smaller article counts than the leading senior cluster."
        ],
    }


def sample_writer_output(
    *,
    topic: str = "inflace",
    segment: str = "senior",
    alert_level: str = "strong",
) -> dict[str, object]:
    return {
        "topic": topic,
        "headline": "Seniors show rising concern as inflation coverage intensifies",
        "narrative": "Inflation coverage is skewing more fear-based for seniors, with concern and avoidance both climbing. Family households also show caution, but the strongest reaction is still concentrated among older audiences.",
        "most_affected_segment": segment,
        "drift_type": "concern_spike",
        "alert_level": alert_level,
        "hypotheses": [
            {
                "segment": segment,
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

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                sample_writer_output(),
            ],
        ) as mock_call:
            brief = generate_brief("inflace")

        assert isinstance(brief, ResearchBrief)
        assert brief.topic == "inflace"
        assert brief.status == "warming"
        assert brief.most_affected_segment == "senior"
        assert brief.drift_type == "concern_spike"
        assert brief.alert_level == "strong"
        assert brief.model_used == OLLAMA_MODEL
        assert brief.generation_mode == "hierarchical_legacy"
        assert brief.confidence_context is not None
        assert set(brief.confidence_context.segment_confidence.keys()) == {
            "young_urban",
            "family",
            "senior",
            "b2b",
        }
        assert len(brief.hypotheses) == 3
        assert brief.hypotheses[0].segment == "senior"
        assert mock_call.call_count == 3
    finally:
        cleanup_temp_db(temp_dir)


def test_brief_prompt_requires_english_output() -> None:
    assert BRIEF_TEMPLATE.startswith(
        "IMPORTANT: Your entire response MUST be in English."
    )


def test_build_context_block_omits_irrelevant_domain_signals() -> None:
    context = build_context_block(
        [
            {
                "segment": "senior",
                "deltas": {
                    "concern_level": 0.15,
                    "purchase_intent": 0.8,
                    "avoidance_signals": 0.2,
                },
                "dominant_frame": "fear",
                "frame_shift": False,
                "alert_level": "mild",
                "article_count": 12,
                "confidence": 0.7,
                "baseline_is_learned": True,
                "baseline_sample_count": 15,
                "baseline_age_days": 2,
                "domain": "civic",
                "relevant_fields": ["concern_level", "avoidance_signals"],
            }
        ],
        [],
    )

    assert "Topic domain: civic" in context
    assert "Relevant signals: concern_level, avoidance_signals" in context
    assert "purchase_intent delta" not in context
    assert "concern_level delta: +0.150" in context
    assert "avoidance_signals delta: +0.200" in context


def test_generate_brief_returns_insufficient_data_without_calling_llm() -> None:
    temp_dir = setup_temp_db()
    try:
        with patch(
            "brief.generator._call_ollama_json",
            side_effect=AssertionError("LLM should not run for a cold-start brief."),
        ) as mock_call:
            brief = generate_brief("cold-topic")
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.status == "insufficient_data"
    assert brief.alert_level == "none"
    assert brief.drift_type == "stable"
    assert brief.confidence_context is not None
    mock_call.assert_not_called()


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

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                response_data,
            ],
        ):
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

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                response_data,
            ],
        ):
            brief = generate_brief("inflace")

        assert brief.drift_type == "purchase_surge"
    finally:
        cleanup_temp_db(temp_dir)


def test_research_brief_rejects_noncanonical_segment_keys() -> None:
    with pytest.raises(ValidationError):
        ResearchBrief(
            topic="inflace",
            status="warming",
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

        with patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                response_data,
            ],
        ):
            brief = generate_brief("inflace")

        assert "degraded" not in brief.headline.lower()
        assert brief.status == "warming"
        assert brief.generation_mode == "fallback"
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
        assert brief.status == "warming"
        assert brief.generation_mode == "fallback"
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
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                response_data,
            ],
        ):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.status == "warming"
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
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                response_data,
            ],
        ):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.status == "ready"
    assert "high confidence" in brief.narrative.lower()


def test_generate_brief_prefers_cluster_snapshot_and_adds_calibration_metadata() -> None:
    temp_dir = setup_temp_db()
    try:
        cluster_payload = sample_cluster_drift_payload(
            country="DE",
            source="spiegel",
            language="de",
        )
        with patch(
            "brief.generator.get_latest_cluster_drift",
            return_value=cluster_payload,
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                sample_writer_output(),
            ],
        ) as mock_call:
            brief = generate_brief(
                "inflace",
                country="DE",
                source="spiegel",
                language="de",
            )
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.generation_mode == "hierarchical_cluster"
    assert brief.source_scope is not None
    assert brief.source_scope.country == "DE"
    assert brief.source_scope.source == "spiegel"
    assert brief.source_scope.language == "de"
    assert brief.calibration_weights is not None
    assert brief.calibration_weights.source_mode == "cluster_drift"
    assert set(brief.calibration_weights.segment_priority.keys()) == {
        "young_urban",
        "family",
        "senior",
        "b2b",
    }
    assert brief.calibration_weights.top_cluster_priorities
    assert mock_call.call_count == 3


def test_generate_brief_run_id_wins_over_scope_lookup() -> None:
    temp_dir = setup_temp_db()
    try:
        cluster_payload = sample_cluster_drift_payload(topic="inflace")
        with patch(
            "brief.generator.get_cluster_drift",
            return_value=cluster_payload,
        ) as mock_run, patch(
            "brief.generator.get_latest_cluster_drift",
            side_effect=AssertionError("Scope lookup should not run when run_id is provided."),
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                sample_writer_output(),
            ],
        ):
            brief = generate_brief(
                "wrong-topic",
                country="DE",
                source="spiegel",
                language="de",
                run_id="run-cluster-1",
                require_cluster=True,
            )
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.topic == "inflace"
    mock_run.assert_called_once_with("run-cluster-1")


def test_generate_brief_writer_failure_uses_latest_successful_artifact_for_fallback() -> None:
    temp_dir = setup_temp_db()
    try:
        cluster_payload = sample_cluster_drift_payload()
        with patch(
            "brief.generator.get_latest_cluster_drift",
            return_value=cluster_payload,
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                sample_explainer_output(),
                {"headline": "broken"},
            ],
        ):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.generation_mode == "fallback"
    assert "seniors are the clearest leading segment" in brief.narrative.lower()


def test_generate_brief_explainer_failure_uses_analyst_artifact_for_fallback() -> None:
    temp_dir = setup_temp_db()
    try:
        cluster_payload = sample_cluster_drift_payload()
        with patch(
            "brief.generator.get_latest_cluster_drift",
            return_value=cluster_payload,
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                sample_analyst_output(),
                RuntimeError("explainer failed"),
            ],
        ):
            brief = generate_brief("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert brief.generation_mode == "fallback"
    assert "senior coverage is carrying the largest calibrated drift" in brief.narrative.lower()


def test_cluster_observation_selection_only_includes_missing_for_fading_story() -> None:
    payload = sample_cluster_drift_payload(top_direction="rising")
    selected = _select_cluster_observations(payload["segments"], payload["clusters"])
    assert all(item["match_type"] != "missing" for item in selected)

    fading_payload = sample_cluster_drift_payload(top_direction="fading")
    selected_fading = _select_cluster_observations(
        fading_payload["segments"],
        fading_payload["clusters"],
    )
    assert any(item["match_type"] == "missing" for item in selected_fading)
    assert len(selected_fading) <= 4


def test_generate_brief_cached_reuses_recent_result() -> None:
    temp_dir = setup_temp_db()
    try:
        expected = sample_brief()
        resolution = Mock(
            display_topic="inflace",
            country="",
            source="",
            language=None,
            source_mode="legacy_drift",
        )
        with patch("brief.generator._resolve_source_reference", return_value=resolution), patch(
            "brief.generator._build_bundle",
            return_value=Mock(),
        ), patch(
            "brief.generator._generate_hierarchical_brief",
            return_value=expected,
        ) as mock_generate:
            first = generate_brief_cached("inflace")
            second = generate_brief_cached("inflace")

        assert first == expected
        assert second == expected
        mock_generate.assert_called_once()
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_cached_refreshes_stale_result() -> None:
    temp_dir = setup_temp_db()
    try:
        import brief.generator as brief_generator

        stale = sample_brief(headline="Stale headline")
        fresh = sample_brief(headline="Fresh headline")

        with brief_generator._cache_lock:
            brief_generator._brief_cache[("inflace", "", "", None, "legacy_drift", None)] = (
                stale,
                datetime.now(timezone.utc) - BRIEF_CACHE_TTL - timedelta(seconds=1),
            )

        resolution = Mock(
            display_topic="inflace",
            country="",
            source="",
            language=None,
            source_mode="legacy_drift",
        )
        with patch("brief.generator._resolve_source_reference", return_value=resolution), patch(
            "brief.generator._build_bundle",
            return_value=Mock(),
        ), patch(
            "brief.generator._generate_hierarchical_brief",
            return_value=fresh,
        ) as mock_generate:
            result = generate_brief_cached("inflace")

        assert result == fresh
        mock_generate.assert_called_once()
    finally:
        cleanup_temp_db(temp_dir)


def test_generate_brief_cached_isolated_by_scope() -> None:
    temp_dir = setup_temp_db()
    try:
        def resolution_for(country: str) -> Mock:
            return Mock(
                display_topic="inflace",
                country=country,
                source="",
                language=None,
                source_mode="legacy_drift",
            )

        first = sample_brief(headline="Germany brief")
        second = sample_brief(headline="France brief")
        with patch(
            "brief.generator._resolve_source_reference",
            side_effect=[resolution_for("DE"), resolution_for("FR"), resolution_for("DE")],
        ), patch(
            "brief.generator._build_bundle",
            return_value=Mock(),
        ), patch(
            "brief.generator._generate_hierarchical_brief",
            side_effect=[first, second],
        ) as mock_generate:
            de_first = generate_brief_cached("inflace", country="DE")
            fr_first = generate_brief_cached("inflace", country="FR")
            de_second = generate_brief_cached("inflace", country="DE")
    finally:
        cleanup_temp_db(temp_dir)

    assert de_first.headline == "Germany brief"
    assert fr_first.headline == "France brief"
    assert de_second.headline == "Germany brief"
    assert mock_generate.call_count == 2


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

    route_methods = {
        route.path: route.methods
        for route in main.app.routes
        if hasattr(route, "methods")
    }
    assert "GET" in route_methods["/brief/{topic}"]
    assert "/health" in route_methods
    assert "POST" in route_methods["/pipeline/run"]
    assert "POST" in route_methods["/pipeline/brief/hierarchical"]
    assert "/brief/{topic}/status" not in route_methods
    assert "/brief/{topic}/result" not in route_methods
