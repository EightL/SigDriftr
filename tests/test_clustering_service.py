#!/usr/bin/env python3
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import db.init
from clustering.clustering_service import get_latest_cluster_run, run_clustering
from extraction.embedder import get_expected_dim, get_model_name


ORIGINAL_DB_PATH = db.init.DB_PATH


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()


def insert_article(
    article_id: str,
    *,
    topic: str = "inflace",
    country: str = "CZ",
    language: str = "cs",
    outlet: str = "irozhlas",
    published_at: str = "2026-03-18T10:00:00+00:00",
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-03-18T10:05:00+00:00')
        """,
        (
            article_id,
            outlet,
            f"Title {article_id}",
            f"Summary {article_id}",
            f"Body {article_id}",
            f"https://example.test/{article_id}",
            f"https://example.test/{article_id}",
            topic,
            country,
            language,
            published_at,
        ),
    )
    conn.commit()


def insert_embedding(
    article_id: str,
    vector: list[float],
    *,
    embedding_text: str,
    status: str = "complete",
    embedded_at: str = "2026-03-18T10:10:00+00:00",
    model_version: str = "test-version",
    language: str = "cs",
) -> int:
    conn = db.init.get_conn()
    text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    cursor = conn.execute(
        """
        INSERT INTO article_embeddings
        (article_id, model_name, model_version, embedding_dim, embedding_vector,
         embedding_text, embedding_text_hash, language, status, error_message,
         embedded_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            article_id,
            get_model_name(),
            model_version,
            get_expected_dim(),
            json.dumps(vector),
            embedding_text,
            text_hash,
            language,
            status,
            embedded_at,
            embedded_at,
            embedded_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def make_vector(*leading: float) -> list[float]:
    dim = get_expected_dim()
    values = [0.0] * dim
    for index, value in enumerate(leading):
        values[index] = value
    return values


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


def test_db_migration_adds_cluster_tables_and_types() -> None:
    temp_dir = tempfile.TemporaryDirectory()
    try:
        db_path = Path(temp_dir.name) / "sigdriftr.db"
        legacy = sqlite3.connect(str(db_path))
        legacy.execute(
            """
            CREATE TABLE articles (
                id TEXT PRIMARY KEY,
                outlet TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT UNIQUE NOT NULL,
                topic TEXT,
                published_at TEXT,
                fetched_at TEXT NOT NULL
            )
            """
        )
        legacy.execute(
            """
            CREATE TABLE baselines (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                segment TEXT NOT NULL,
                concern_level REAL,
                purchase_intent REAL,
                avoidance_signals REAL,
                dominant_frame TEXT,
                seeded INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        legacy.commit()
        legacy.close()

        db.init.DB_PATH = db_path
        if hasattr(db.init._local, "conn"):
            db.init._local.conn.close()
            delattr(db.init._local, "conn")

        conn = db.init.get_conn()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        membership_columns = {
            row[1]: (row[2], row[4])
            for row in conn.execute("PRAGMA table_info(cluster_memberships)").fetchall()
        }
        cluster_signal_columns = {
            row[1]: (row[2], row[4])
            for row in conn.execute("PRAGMA table_info(cluster_signals)").fetchall()
        }
        run_columns = {
            row[1]: (row[2], row[4])
            for row in conn.execute("PRAGMA table_info(cluster_runs)").fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    finally:
        if hasattr(db.init._local, "conn"):
            db.init._local.conn.close()
            delattr(db.init._local, "conn")
        db.init.DB_PATH = ORIGINAL_DB_PATH
        temp_dir.cleanup()

    assert {"cluster_runs", "clusters", "cluster_memberships", "cluster_signals"}.issubset(
        tables
    )
    assert membership_columns["article_id"][0] == "TEXT"
    assert membership_columns["embedding_id"][0] == "INTEGER"
    assert cluster_signal_columns["cluster_id"][0] == "INTEGER"
    assert cluster_signal_columns["extractor_provider"][0] == "TEXT"
    assert "datetime" in str(membership_columns["created_at"][1]).lower()
    assert "datetime" in str(run_columns["created_at"][1]).lower()
    assert "idx_cm_run_article_unique" in indexes
    assert "idx_clusters_run_label" in indexes
    assert "idx_cluster_signals_run_id" in indexes


def test_run_clustering_uses_latest_complete_embedding_and_records_embedding_id() -> None:
    temp_dir = setup_temp_db()
    try:
        for index in range(1, 6):
            insert_article(f"article-{index}")

        old_id = insert_embedding(
            "article-1",
            make_vector(4.0, 0.0, 0.0),
            embedding_text="article-1-old",
            embedded_at="2026-03-18T10:10:00+00:00",
        )
        new_id = insert_embedding(
            "article-1",
            make_vector(0.0, 5.0, 0.0),
            embedding_text="article-1-new",
            embedded_at="2026-03-18T10:20:00+00:00",
        )
        assert old_id != new_id
        insert_embedding("article-2", make_vector(0.0, 4.0, 0.0), embedding_text="article-2")
        insert_embedding("article-3", make_vector(0.0, 3.0, 0.0), embedding_text="article-3")
        insert_embedding("article-4", make_vector(3.0, 0.0, 0.0), embedding_text="article-4")
        insert_embedding("article-5", make_vector(2.0, 0.0, 0.0), embedding_text="article-5")

        with patch(
            "clustering.clustering_service.reduce_embeddings",
            side_effect=fake_reduce,
        ), patch(
            "clustering.clustering_service.cluster_reduced",
            side_effect=fake_cluster_by_dominant_axis,
        ):
            result = run_clustering(
                topic="inflace",
                window_hours=99999,
                min_cluster_size=2,
            )

        membership_row = db.init.get_conn().execute(
            """
            SELECT embedding_id
            FROM cluster_memberships
            WHERE run_id = ? AND article_id = ?
            """,
            (result["run_id"], "article-1"),
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert result["status"] == "completed"
    assert result["n_clusters"] == 2
    assert membership_row == (new_id,)


def test_run_clustering_persists_skipped_small_sample_run() -> None:
    temp_dir = setup_temp_db()
    try:
        for index in range(1, 5):
            insert_article(f"article-{index}")
            insert_embedding(
                f"article-{index}",
                make_vector(float(index), 0.0, 0.0),
                embedding_text=f"article-{index}",
            )

        result = run_clustering(topic="inflace", window_hours=99999)
        conn = db.init.get_conn()
        run_row = conn.execute(
            """
            SELECT status, n_clusters, n_noise
            FROM cluster_runs
            WHERE run_id = ?
            """,
            (result["run_id"],),
        ).fetchone()
        cluster_count = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        membership_count = conn.execute(
            "SELECT COUNT(*) FROM cluster_memberships"
        ).fetchone()[0]
    finally:
        cleanup_temp_db(temp_dir)

    assert result["status"] == "skipped_small_sample"
    assert run_row == ("skipped_small_sample", 0, 0)
    assert cluster_count == 0
    assert membership_count == 0


def test_run_clustering_persists_all_noise_result() -> None:
    temp_dir = setup_temp_db()
    try:
        for index in range(1, 6):
            insert_article(f"article-{index}")
            insert_embedding(
                f"article-{index}",
                make_vector(0.0, 0.0, float(index)),
                embedding_text=f"article-{index}",
            )

        with patch(
            "clustering.clustering_service.reduce_embeddings",
            side_effect=fake_reduce,
        ), patch(
            "clustering.clustering_service.cluster_reduced",
            return_value=(np.asarray([-1, -1, -1, -1, -1]), np.asarray([0.2] * 5)),
        ):
            result = run_clustering(topic="inflace", window_hours=99999)

        rows = db.init.get_conn().execute(
            """
            SELECT cluster_id, is_noise
            FROM cluster_memberships
            WHERE run_id = ?
            ORDER BY article_id ASC
            """,
            (result["run_id"],),
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert result["status"] == "all_noise"
    assert result["n_clusters"] == 0
    assert result["n_noise"] == 5
    assert rows == [(None, 1)] * 5


def test_run_clustering_persists_clustered_centroids_and_latest_lookup() -> None:
    temp_dir = setup_temp_db()
    try:
        groups = [(0, 7), (1, 7), (2, 6)]
        article_index = 0
        for group_index, count in groups:
            for item_index in range(count):
                article_id = f"article-{article_index}"
                insert_article(article_id)
                leading = [0.05, 0.05, 0.05]
                leading[group_index] = 3.0 + item_index * 0.1
                insert_embedding(
                    article_id,
                    make_vector(*leading),
                    embedding_text=article_id,
                )
                article_index += 1

        with patch(
            "clustering.clustering_service.reduce_embeddings",
            side_effect=fake_reduce,
        ), patch(
            "clustering.clustering_service.cluster_reduced",
            side_effect=fake_cluster_by_dominant_axis,
        ):
            result = run_clustering(topic="inflace", window_hours=99999)

        conn = db.init.get_conn()
        cluster_rows = conn.execute(
            """
            SELECT centroid_vector
            FROM clusters
            WHERE run_id = ?
            ORDER BY cluster_label ASC
            """,
            (result["run_id"],),
        ).fetchall()
        membership_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM cluster_memberships
            WHERE run_id = ?
            """,
            (result["run_id"],),
        ).fetchone()[0]
        latest = get_latest_cluster_run(topic="inflace")
    finally:
        cleanup_temp_db(temp_dir)

    assert result["status"] == "completed"
    assert result["n_clusters"] == 3
    assert membership_count == 20
    assert len(cluster_rows) == 3
    for (centroid_json,) in cluster_rows:
        centroid = json.loads(centroid_json)
        assert len(centroid) == get_expected_dim()
        assert np.linalg.norm(np.asarray(centroid)) == pytest.approx(1.0, abs=1e-5)

    assert latest is not None
    assert latest["run_id"] == result["run_id"]
    assert len(latest["clusters"]) == 3
