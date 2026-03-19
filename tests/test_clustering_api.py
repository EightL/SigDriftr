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


def test_pipeline_latest_clusters_route_raises_404_when_scope_missing() -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from api.routes import pipeline as pipeline_route

    with patch("api.routes.pipeline.get_latest_cluster_run", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            pipeline_route.get_latest_clusters(topic="inflace")

    assert exc_info.value.status_code == 404
