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
