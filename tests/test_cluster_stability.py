#!/usr/bin/env python3
import hashlib
import json
from unittest.mock import patch

import numpy as np

import db.init
from clustering.stability import evaluate_cluster_stability
from db_helpers import cleanup_temp_db, setup_temp_db
from extraction.embedder import get_expected_dim, get_model_name


def make_vector(*leading: float) -> list[float]:
    values = [0.0] * get_expected_dim()
    for index, value in enumerate(leading):
        values[index] = value
    return values


def insert_article(article_id: str, *, title: str = "Inflation story") -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
        VALUES (?, 'irozhlas', ?, 'summary', 'body', ?, ?, 'inflace', 'CZ', 'cs',
                '2026-03-18T10:00:00+00:00', '2026-03-18T10:05:00+00:00')
        """,
        (
            article_id,
            title,
            f"https://example.test/{article_id}",
            f"https://example.test/{article_id}",
        ),
    )
    conn.commit()


def insert_embedding(article_id: str, vector: list[float]) -> None:
    conn = db.init.get_conn()
    embedding_text = f"Embedding text for {article_id}"
    text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO article_embeddings
        (article_id, model_name, model_version, embedding_dim, embedding_vector,
         embedding_text, embedding_text_hash, language, status, error_message,
         embedded_at, created_at, updated_at)
        VALUES (?, ?, 'test-version', ?, ?, ?, ?, 'cs', 'complete', NULL,
                '2026-03-18T10:10:00+00:00',
                '2026-03-18T10:10:00+00:00',
                '2026-03-18T10:10:00+00:00')
        """,
        (
            article_id,
            get_model_name(),
            get_expected_dim(),
            json.dumps(vector),
            embedding_text,
            text_hash,
        ),
    )
    conn.commit()


def fake_reduce(vectors: np.ndarray, *, n_components: int, n_neighbors: int, random_state: int) -> np.ndarray:
    return np.asarray(vectors)[:, : max(1, min(3, vectors.shape[1]))]


def fake_cluster_by_dominant_axis(
    reduced_vectors: np.ndarray,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(np.argmax(vector[:3])) for vector in reduced_vectors])
    probabilities = np.asarray([0.99] * len(reduced_vectors))
    return labels, probabilities


def test_evaluate_cluster_stability_reports_bootstrap_metrics() -> None:
    temp_dir = setup_temp_db()
    try:
        for group_index in range(3):
            for item_index in range(3):
                article_id = f"article-{group_index}-{item_index}"
                leading = [0.05, 0.05, 0.05]
                leading[group_index] = 3.0 + item_index
                insert_article(article_id, title=f"stability{group_index}{item_index}")
                insert_embedding(article_id, make_vector(*leading))

        with patch(
            "clustering.clustering_service.reduce_embeddings",
            side_effect=fake_reduce,
        ), patch(
            "clustering.clustering_service.cluster_reduced",
            side_effect=fake_cluster_by_dominant_axis,
        ):
            report = evaluate_cluster_stability(
                topic="inflace",
                window_hours=99999,
                bootstrap_samples=5,
                sample_fraction=0.8,
                random_state=7,
            )
    finally:
        cleanup_temp_db(temp_dir)

    assert report["status"] == "completed"
    assert report["base"]["n_articles"] == 9
    assert report["base"]["n_clusters"] == 3
    assert report["global"]["completed_bootstraps"] == 5
    assert report["global"]["ari"]["median"] == 1.0
    assert report["global"]["nmi"]["median"] == 1.0
    assert report["dedupe"]["duplicates_removed"] == 0
    assert len(report["clusters"]) == 3
