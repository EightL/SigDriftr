#!/usr/bin/env python3
import pytest
from unittest.mock import patch


def test_pipeline_embed_route_passes_filters_through() -> None:
    pytest.importorskip("fastapi")
    from api.routes import pipeline as pipeline_route

    with patch(
        "api.routes.pipeline.embed_pending_articles",
        return_value={
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dim": 384,
            "selected": 10,
            "embedded": 7,
            "already_current": 2,
            "retried_failed": 1,
            "stale_reembedded": 0,
            "failed": 0,
            "duration_s": 0.42,
        },
    ) as mock_embed:
        result = pipeline_route.run_embedding_stage(
            topic="inflace",
            country="DE",
            source="spiegel",
            limit=25,
        )

    assert result["model_name"] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert result["embedded"] == 7
    mock_embed.assert_called_once_with(
        limit=25,
        topic="inflace",
        country="DE",
        source="spiegel",
    )
