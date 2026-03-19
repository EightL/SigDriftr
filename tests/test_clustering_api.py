#!/usr/bin/env python3
from unittest.mock import patch

import pytest


def test_pipeline_cluster_route_passes_scope_filters_through() -> None:
    pytest.importorskip("fastapi")
    from api.models import ClusterRunResponse
    from api.routes import pipeline as pipeline_route

    payload = {
        "run_id": "run-123",
        "topic": "inflace",
        "country": "DE",
        "source": "spiegel",
        "language": "de",
        "window_start": "2026-03-18T00:00:00+00:00",
        "window_end": "2026-03-19T00:00:00+00:00",
        "status": "completed",
        "n_articles": 10,
        "n_clusters": 3,
        "n_noise": 1,
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_version": "test-version",
        "umap_n_components": 8,
        "umap_n_neighbors": 9,
        "hdbscan_min_cluster_size": 5,
        "hdbscan_min_samples": 2,
        "duration_s": 0.42,
    }

    with patch("api.routes.pipeline.run_clustering", return_value=payload) as mock_cluster:
        result = pipeline_route.run_cluster_stage(
            topic="inflace",
            country="DE",
            source="spiegel",
            language="de",
            window_hours=48,
            min_cluster_size=5,
        )

    validated = ClusterRunResponse.model_validate(result)
    assert validated.run_id == "run-123"
    mock_cluster.assert_called_once_with(
        topic="inflace",
        country="DE",
        source="spiegel",
        language="de",
        window_hours=48,
        min_cluster_size=5,
    )


def test_pipeline_cluster_signal_route_passes_args_through() -> None:
    pytest.importorskip("fastapi")
    from api.models import ClusterSignalStageResponse
    from api.routes import pipeline as pipeline_route

    payload = {
        "run_id": "run-789",
        "selected_clusters": 4,
        "processed": 3,
        "skipped_existing": 1,
        "failed": 0,
        "provider": "ollama",
        "model_name": "qwen2.5:7b-instruct",
        "duration_s": 1.24,
    }

    with patch(
        "api.routes.pipeline.run_cluster_extraction",
        return_value=payload,
    ) as mock_extract:
        result = pipeline_route.run_cluster_signal_stage(
            run_id="run-789",
            overwrite=True,
            min_cluster_size=5,
        )

    validated = ClusterSignalStageResponse.model_validate(result)
    assert validated.processed == 3
    mock_extract.assert_called_once_with(
        run_id="run-789",
        overwrite=True,
        min_cluster_size=5,
    )


def test_pipeline_cluster_drift_route_passes_args_through() -> None:
    pytest.importorskip("fastapi")
    from api.models import ClusterDriftStageResponse
    from api.routes import pipeline as pipeline_route

    payload = {
        "run_id": "run-drift-1",
        "observed_clusters": 3,
        "matched_tracks": 2,
        "new_tracks": 1,
        "missing_tracks": 0,
        "segments": 4,
        "computed_at": "2026-03-19T09:30:00+00:00",
        "duration_s": 0.37,
    }

    with patch(
        "api.routes.pipeline.run_cluster_drift",
        return_value=payload,
    ) as mock_drift:
        result = pipeline_route.run_cluster_drift_stage(run_id="run-drift-1")

    validated = ClusterDriftStageResponse.model_validate(result)
    assert validated.matched_tracks == 2
    mock_drift.assert_called_once_with("run-drift-1")


def test_pipeline_latest_clusters_route_passes_scope_filters_through() -> None:
    pytest.importorskip("fastapi")
    from api.models import LatestClusterRunResponse
    from api.routes import pipeline as pipeline_route

    payload = {
        "run_id": "run-456",
        "topic": "inflace",
        "country": "DE",
        "source": "spiegel",
        "language": "de",
        "window_start": "2026-03-18T00:00:00+00:00",
        "window_end": "2026-03-19T00:00:00+00:00",
        "status": "completed",
        "n_articles": 10,
        "n_clusters": 2,
        "n_noise": 1,
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_version": "test-version",
        "umap_n_components": 8,
        "umap_n_neighbors": 9,
        "hdbscan_min_cluster_size": 3,
        "hdbscan_min_samples": 2,
        "created_at": "2026-03-19 09:00:00",
        "clusters": [
            {
                "cluster_id": 1,
                "cluster_label": 0,
                "size": 4,
                "centroid_vector": [0.1, 0.2],
                "centroid_dim": 2,
                "members": [
                    {
                        "article_id": "article-1",
                        "embedding_id": 10,
                        "membership_strength": 0.99,
                        "is_noise": False,
                    }
                ],
                "signal": {
                    "topic_label": "Inflation cluster",
                    "concern_level": 0.7,
                    "purchase_intent": 0.2,
                    "avoidance_signals": 0.3,
                    "sentiment": -0.1,
                    "dominant_frame": "fear",
                    "frame_detail": "uncertainty",
                    "seg_young_urban": 0.4,
                    "seg_family": 0.3,
                    "seg_senior": 0.2,
                    "seg_b2b": 0.1,
                    "evidence": ["Prices remain volatile."],
                    "raw_json": {"ok": True},
                    "member_count": 4,
                    "membership_fingerprint": "abc123",
                    "exemplar_article_ids": ["article-1", "article-2", "article-3"],
                    "extractor_provider": "ollama",
                    "extractor_model": "qwen2.5:7b-instruct",
                    "schema_version": "v1",
                    "extracted_at": "2026-03-19T09:10:00+00:00",
                },
            }
        ],
        "noise_members": [
            {
                "article_id": "article-9",
                "embedding_id": 19,
                "membership_strength": 0.2,
                "is_noise": True,
            }
        ],
    }

    with patch(
        "api.routes.pipeline.get_latest_cluster_run",
        return_value=payload,
    ) as mock_latest:
        result = pipeline_route.get_latest_clusters(
            topic="inflace",
            country="DE",
            source="spiegel",
            language="de",
        )

    validated = LatestClusterRunResponse.model_validate(result)
    assert validated.run_id == "run-456"
    assert validated.clusters[0].members[0].article_id == "article-1"
    assert validated.clusters[0].signal is not None
    assert validated.clusters[0].signal.topic_label == "Inflation cluster"
    mock_latest.assert_called_once_with(
        topic="inflace",
        country="DE",
        source="spiegel",
        language="de",
    )


def test_cluster_drift_route_passes_scope_filters_through() -> None:
    pytest.importorskip("fastapi")
    from api.models import ClusterDriftResponse
    from api.routes import calibration as calibration_route

    payload = {
        "topic": "inflace",
        "country": "DE",
        "source": "spiegel",
        "language": "de",
        "run_id": "run-drift-2",
        "computed_at": "2026-03-19T09:30:00+00:00",
        "segments": [
            {
                "segment": "young_urban",
                "topic": "inflace",
                "article_count": 8,
                "has_data": True,
                "current": {
                    "concern_level": 0.7,
                    "purchase_intent": 0.2,
                    "avoidance_signals": 0.3,
                },
                "baseline": {
                    "concern_level": 0.6,
                    "purchase_intent": 0.3,
                    "avoidance_signals": 0.25,
                },
                "deltas": {
                    "concern_level": 0.1,
                    "purchase_intent": -0.1,
                    "avoidance_signals": 0.05,
                },
                "drift_magnitude": 0.18,
                "frame_shift": False,
                "alert_level": "none",
                "dominant_frame": "fear",
                "baseline_frame": "fear",
                "confidence": 0.55,
                "baseline_is_learned": True,
                "baseline_sample_count": 31,
                "baseline_age_days": 1,
                "status": "ready",
                "domain": "commerce",
                "relevant_fields": [
                    "concern_level",
                    "purchase_intent",
                    "avoidance_signals",
                ],
                "direction": "rising",
                "centroid_shift": 0.11,
                "new_cluster_weight": 0.25,
                "tracked_cluster_count": 2,
                "matched_cluster_count": 1,
                "new_cluster_count": 1,
                "missing_cluster_count": 0,
            }
        ],
        "clusters": [
            {
                "track_id": "track-1",
                "cluster_id": 10,
                "cluster_label": 0,
                "topic_label": "Inflation cluster",
                "baseline_topic_label": "Inflation cluster",
                "match_type": "matched",
                "direction": "rising",
                "centroid_distance": 0.12,
                "segment_vector_distance": 0.08,
                "signal_drift": 0.16,
                "drift_magnitude": 0.15,
                "alert_level": "none",
                "confidence": 0.6,
                "member_count": 4,
                "mean_membership_strength": 0.95,
                "current": {
                    "concern_level": 0.7,
                    "purchase_intent": 0.2,
                    "avoidance_signals": 0.3,
                },
                "baseline": {
                    "concern_level": 0.6,
                    "purchase_intent": 0.3,
                    "avoidance_signals": 0.25,
                },
                "deltas": {
                    "concern_level": 0.1,
                    "purchase_intent": -0.1,
                    "avoidance_signals": 0.05,
                },
                "dominant_frame": "fear",
                "baseline_frame": "fear",
                "frame_shift": False,
            }
        ],
    }

    with patch(
        "api.routes.calibration.get_latest_cluster_drift",
        return_value=payload,
    ) as mock_latest:
        result = calibration_route.get_cluster_drift_view(
            topic="inflace",
            country="DE",
            source="spiegel",
            language="de",
        )

    validated = ClusterDriftResponse.model_validate(result)
    assert validated.run_id == "run-drift-2"
    assert validated.segments[0].direction == "rising"
    mock_latest.assert_called_once_with(
        topic="inflace",
        country="DE",
        source="spiegel",
        language="de",
    )


def test_pipeline_latest_clusters_route_raises_404_when_scope_missing() -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from api.routes import pipeline as pipeline_route

    with patch("api.routes.pipeline.get_latest_cluster_run", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            pipeline_route.get_latest_clusters(topic="inflace")

    assert exc_info.value.status_code == 404


def test_cluster_drift_route_raises_404_when_scope_missing() -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from api.routes import calibration as calibration_route

    with patch("api.routes.calibration.get_latest_cluster_drift", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            calibration_route.get_cluster_drift_view(topic="inflace")

    assert exc_info.value.status_code == 404
