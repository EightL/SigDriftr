from __future__ import annotations

import json
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")

from evaluation.briefs import evaluate_records, load_golden_records, load_prediction_records


def brief_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "topic": "inflace",
        "status": "ready",
        "headline": "Seniors show rising concern in inflation coverage",
        "narrative": "High signal readiness finding. Coverage is shifting most clearly for seniors.",
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
        "confidence_context": {
            "segment_confidence": {"senior": 0.82},
            "baseline_is_learned": {"senior": True},
            "baseline_sample_count": {"senior": 35},
        },
        "generation_mode": "hierarchical_cluster",
        "generated_at": "2026-03-17T00:00:00+00:00",
        "model_used": "qwen2.5:7b-instruct",
    }
    payload.update(updates)
    return payload


def golden_record() -> dict[str, object]:
    return {
        "id": "brief-1",
        "input": {"topic": "inflace", "country": "CZ", "language": "cs"},
        "gold": {
            "brief": {
                "status": "ready",
                "most_affected_segment": "senior",
                "drift_type": "concern_spike",
                "alert_level": "strong",
            },
            "support_expectations": {
                "required_track_ids": ["track-1"],
                "forbidden_track_ids": ["track-9"],
                "expected_hypothesis_segments": ["senior", "family"],
            },
        },
    }


def test_evaluate_records_scores_fields_and_support() -> None:
    report = evaluate_records(
        [
            {
                **golden_record(),
                "prediction": {
                    "brief": brief_payload(),
                    "support": {
                        "cited_track_ids": ["track-1"],
                        "validation_issues": [],
                    },
                },
            }
        ]
    )

    assert report["valid_prediction_count"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert report["fields"]["alert_level"]["accuracy"] == 1.0
    assert report["support_failure_count"] == 0
    assert report["hypothesis_failure_count"] == 0
    assert report["fallback_rate"] == 0.0


def test_evaluate_records_reports_invalid_and_support_failures() -> None:
    record = golden_record()
    predictions = {
        "brief-1": {
            "brief": brief_payload(
                most_affected_segment="family",
                generation_mode="fallback",
                hypotheses=[
                    {
                        "segment": "senior",
                        "hypothesis": "Seniors will report higher concern.",
                        "signal_basis": "concern_level +0.20",
                        "suggested_question": "Inflation coverage makes me more concerned.",
                    },
                    {
                        "segment": "b2b",
                        "hypothesis": "Businesses will report mild caution.",
                        "signal_basis": "avoidance_signals +0.06",
                        "suggested_question": "Inflation coverage makes planning harder.",
                    },
                    {
                        "segment": "young_urban",
                        "hypothesis": "Young urban adults will remain comparatively stable.",
                        "signal_basis": "purchase_intent -0.03",
                        "suggested_question": "Inflation coverage changes my planned purchases.",
                    },
                ],
            ),
            "support": {
                "cited_track_ids": ["track-9"],
                "validation_issues": [{"code": "example"}],
            },
        }
    }

    report = evaluate_records([record], predictions)

    assert report["valid_prediction_count"] == 1
    assert report["fields"]["most_affected_segment"]["accuracy"] == 0.0
    assert report["fallback_rate"] == 1.0
    assert report["support_failure_count"] == 2
    assert report["hypothesis_failure_count"] == 1
    assert report["validation_issue_total"] == 1


def test_load_brief_eval_jsonl_and_prediction_map(tmp_path: Path) -> None:
    goldens_path = tmp_path / "goldens.jsonl"
    predictions_path = tmp_path / "predictions.json"
    goldens_path.write_text(json.dumps(golden_record()) + "\n", encoding="utf-8")
    predictions_path.write_text(
        json.dumps({"brief-1": {"brief": brief_payload()}}),
        encoding="utf-8",
    )

    assert load_golden_records(goldens_path)[0]["id"] == "brief-1"
    assert "brief-1" in load_prediction_records(predictions_path)
