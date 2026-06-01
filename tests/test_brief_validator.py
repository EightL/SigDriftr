from __future__ import annotations

from types import SimpleNamespace

import pytest

pydantic = pytest.importorskip("pydantic")

from brief.models import ResearchBrief
from brief.validator import validate_brief, validation_issues_as_dicts


def make_brief(**updates: object) -> ResearchBrief:
    payload = {
        "topic": "inflace",
        "status": "ready",
        "headline": "Seniors show rising concern in inflation coverage",
        "narrative": "Coverage is shifting most clearly for seniors.",
        "most_affected_segment": "senior",
        "drift_type": "concern_spike",
        "alert_level": "strong",
        "hypotheses": [
            {
                "segment": "senior",
                "hypothesis": "Seniors will report higher concern.",
                "signal_basis": "concern_level +0.20",
                "suggested_question": "Inflation coverage makes me more concerned.",
            },
            {
                "segment": "family",
                "hypothesis": "Families will report mild caution.",
                "signal_basis": "avoidance_signals +0.06",
                "suggested_question": "Inflation coverage makes me more cautious.",
            },
            {
                "segment": "young_urban",
                "hypothesis": "Young urban adults will remain comparatively stable.",
                "signal_basis": "purchase_intent -0.03",
                "suggested_question": "Inflation coverage changes my planned purchases.",
            },
        ],
        "generated_at": "2026-03-17T00:00:00+00:00",
        "model_used": "qwen2.5:7b-instruct",
    }
    payload.update(updates)
    return ResearchBrief(**payload)


def make_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        status="ready",
        segment_rollups=[
            {
                "segment": "senior",
                "article_count": 12,
                "has_data": True,
                "alert_level": "strong",
                "drift_magnitude": 0.41,
                "confidence": 0.82,
                "baseline_is_learned": True,
                "deltas": {
                    "concern_level": 0.20,
                    "purchase_intent": -0.08,
                    "avoidance_signals": 0.16,
                },
            },
            {
                "segment": "family",
                "article_count": 9,
                "has_data": True,
                "alert_level": "mild",
                "drift_magnitude": 0.18,
                "confidence": 0.71,
                "baseline_is_learned": True,
                "deltas": {
                    "concern_level": 0.09,
                    "purchase_intent": -0.02,
                    "avoidance_signals": 0.06,
                },
            },
            {
                "segment": "young_urban",
                "article_count": 6,
                "has_data": True,
                "alert_level": "none",
                "drift_magnitude": 0.06,
                "confidence": 0.62,
                "baseline_is_learned": True,
                "deltas": {
                    "concern_level": 0.03,
                    "purchase_intent": -0.03,
                    "avoidance_signals": 0.01,
                },
            },
        ],
        selected_observations=[
            {
                "track_id": "track-1",
                "observation_id": "track-1",
                "exemplar_article_ids": ["article-1"],
                "exemplar_articles": [
                    {
                        "article_id": "article-2",
                        "title": "Evidence article",
                    }
                ],
            }
        ],
    )


def test_validate_brief_accepts_grounded_support() -> None:
    issues = validate_brief(
        make_brief(),
        bundle=make_bundle(),
        cited_track_ids=["track-1"],
        cited_article_ids=["article-1", "article-2"],
    )

    assert issues == []


def test_validate_brief_reports_unsupported_ids_and_claim_mismatch() -> None:
    brief = make_brief(
        most_affected_segment="family",
        alert_level="mild",
        hypotheses=[
            {
                "segment": "senior",
                "hypothesis": "Seniors will report higher concern.",
                "signal_basis": "concern_level +0.90",
                "suggested_question": "Inflation coverage makes me more concerned.",
            },
            {
                "segment": "family",
                "hypothesis": "Families will report mild caution.",
                "signal_basis": "avoidance_signals +0.06",
                "suggested_question": "Inflation coverage makes me more cautious.",
            },
            {
                "segment": "young_urban",
                "hypothesis": "Young urban adults will remain comparatively stable.",
                "signal_basis": "purchase_intent -0.03",
                "suggested_question": "Inflation coverage changes my planned purchases.",
            },
        ],
    )

    issues = validate_brief(
        brief,
        bundle=make_bundle(),
        cited_track_ids=["missing-track"],
        cited_article_ids=["missing-article"],
    )
    codes = {issue.code for issue in issues}

    assert {
        "cited_track_not_selected",
        "cited_article_not_selected",
        "most_affected_segment_not_top_ranked",
        "alert_level_differs_from_top_segment",
        "hypothesis_delta_mismatch",
    }.issubset(codes)
    assert validation_issues_as_dicts(issues)[0]["severity"] == "warning"


def test_validate_brief_reports_ready_without_ready_segment() -> None:
    bundle = make_bundle()
    for segment in bundle.segment_rollups:
        segment["baseline_is_learned"] = False
        segment["confidence"] = 0.2

    issues = validate_brief(make_brief(), bundle=bundle)

    assert "ready_without_ready_segment" in {issue.code for issue in issues}
