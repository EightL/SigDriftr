from __future__ import annotations

from pathlib import Path

from evaluation.weak_gold import (
    DEFAULT_BASELINE_PATH,
    evaluate,
    labels_from_record,
    load_baseline,
    load_markdown_table_records,
    load_predictions,
)


def test_labels_from_record_maps_raw_signal_scores() -> None:
    labels = labels_from_record(
        {
            "article_id": "a1",
            "concern_level": 0.7,
            "purchase_intent": 0.3,
            "avoidance_signals": 0.1,
            "dominant_frame": "risk",
            "seg_young_urban": 0.19,
            "seg_family": 0.2,
            "seg_senior": 0.46,
            "seg_b2b": 0.45,
        }
    )

    assert labels == {
        "concern_bucket": "high",
        "purchase_relevance": "1",
        "avoidance_relevance": "0",
        "dominant_frame": "fear",
        "seg_young_urban_relevance": "0",
        "seg_family_relevance": "1",
        "seg_senior_relevance": "2",
        "seg_b2b_relevance": "2",
    }


def test_labels_from_record_prefers_explicit_relevance_and_topic_score() -> None:
    labels = labels_from_record(
        {
            "article_id": "a1",
            "relevance_score": 0.72,
            "seg_young_urban": 0.8,
            "seg_young_urban_relevance": 0.1,
            "raw_json": {
                "seg_family_relevance": 0.7,
            },
        }
    )

    assert labels["topic_relevance"] == "1"
    assert labels["seg_young_urban_relevance"] == "0"
    assert labels["seg_family_relevance"] == "2"


def test_evaluate_skips_blank_baseline_and_tracks_missing_predictions() -> None:
    baseline = [
        {
            "article_id": "a1",
            "canonical_topic_id": "inflace",
            "source": "idnes",
            "labels": {
                "topic_relevance": "2",
                "dominant_frame": "fear",
                "concern_bucket": "medium",
            },
        },
        {
            "article_id": "a2",
            "canonical_topic_id": "inflace",
            "source": "idnes",
            "labels": {
                "dominant_frame": "neutral",
            },
        },
    ]
    predictions = {
        "a1": {
            "dominant_frame": "fear",
            "concern_bucket": "low",
        },
        "extra": {
            "dominant_frame": "neutral",
        },
    }

    report = evaluate(baseline, predictions)

    assert report["overall"]["support"] == 2
    assert report["overall"]["correct"] == 1
    assert report["fields"]["dominant_frame"]["support"] == 1
    assert report["fields"]["concern_bucket"]["support"] == 1
    assert report["fields"]["topic_relevance"]["missing_prediction_count"] == 1
    assert report["missing_article_count"] == 1
    assert report["unexpected_article_count"] == 1


def test_repository_weak_gold_files_are_loadable() -> None:
    baseline_path = Path("eval/annotations/czenec_batch_200_four_model_majority_labeled.csv")
    predictions_path = Path("eval/model_outputs/czenec_batch_200_chatgpt.json")

    baseline = load_baseline(baseline_path)
    predictions = load_predictions(predictions_path)
    report = evaluate(baseline, predictions)

    assert len(baseline) == 200
    assert len(predictions) == 200
    assert report["overall"]["support"] > 0
    assert report["fields"]["topic_relevance"]["support"] == 187


def test_repository_semi_gold_file_is_complete_default_baseline() -> None:
    baseline = load_baseline(DEFAULT_BASELINE_PATH)

    assert DEFAULT_BASELINE_PATH.name == "czech_batch_200_semi_gold.csv"
    assert len(baseline) == 200
    assert sum(len(row["labels"]) for row in baseline) == 1800


def test_load_markdown_table_records_supports_human_annotations() -> None:
    records = load_markdown_table_records(
        Path("eval/annotations/czech_batch_200_human.md")
    )
    baseline = load_baseline(Path("eval/annotations/czech_batch_200_human.md"))

    assert len(records) == 200
    assert len(baseline) == 200
    assert sum(len(row["labels"]) for row in baseline) == 1800
