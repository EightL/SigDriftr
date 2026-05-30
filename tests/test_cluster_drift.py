#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

import pytest

import db.init
from config.domains import get_domain_config
from delta.cluster_drift import (
    _assignment_metrics,
    get_cluster_drift,
    get_latest_cluster_drift,
    run_cluster_drift,
)
from extraction.embedder import get_model_name


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


def make_vector(*leading: float) -> list[float]:
    values = [0.0] * 384
    for index, value in enumerate(leading):
        values[index] = value
    return values


def insert_cluster_run(
    run_id: str,
    *,
    topic: str = "inflace",
    country: str = "",
    source: str = "",
    language: str = "cs",
    status: str = "completed",
    n_articles: int = 3,
    n_clusters: int = 1,
    n_noise: int = 0,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO cluster_runs
        (run_id, topic, country, source, language, window_start, window_end,
         status, n_articles, n_clusters, n_noise, umap_n_components,
         umap_n_neighbors, hdbscan_min_cluster_size, hdbscan_min_samples,
         model_name, model_version)
        VALUES (?, ?, ?, ?, ?, '2026-03-18T00:00:00+00:00', '2026-03-19T00:00:00+00:00',
                ?, ?, ?, ?, 8, 10, 3, 2, ?, 'test-version')
        """,
        (run_id, topic, country, source, language, status, n_articles, n_clusters, n_noise, get_model_name()),
    )
    conn.commit()


def insert_cluster(
    run_id: str,
    *,
    cluster_label: int = 0,
    size: int = 3,
    centroid_vector: list[float] | None = None,
) -> int:
    conn = db.init.get_conn()
    cursor = conn.execute(
        """
        INSERT INTO clusters
        (run_id, cluster_label, size, centroid_vector, centroid_dim)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            run_id,
            cluster_label,
            size,
            json.dumps(centroid_vector or make_vector(1.0, 0.0, 0.0)),
            len(centroid_vector or make_vector(1.0, 0.0, 0.0)),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_cluster_signal(
    cluster_id: int,
    *,
    run_id: str,
    topic_label: str = "Inflation cluster",
    concern: float = 0.7,
    purchase: float = 0.2,
    avoidance: float = 0.3,
    dominant_frame: str = "fear",
    seg_young_urban: float = 0.6,
    seg_family: float = 0.2,
    seg_senior: float = 0.1,
    seg_b2b: float = 0.1,
    member_count: int = 3,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO cluster_signals
        (cluster_id, run_id, topic_label, concern_level, purchase_intent,
         avoidance_signals, sentiment, dominant_frame, frame_detail,
         seg_young_urban, seg_family, seg_senior, seg_b2b, evidence_json,
         raw_json, member_count, membership_fingerprint, exemplar_article_ids,
         extractor_provider, extractor_model, schema_version, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, 0.0, ?, ?, ?, ?, ?, ?, '[]', '{}', ?, ?, '[]',
                'ollama', 'qwen2.5:7b-instruct', 'v1', '2026-03-19T09:00:00+00:00')
        """,
        (
            cluster_id,
            run_id,
            topic_label,
            concern,
            purchase,
            avoidance,
            dominant_frame,
            dominant_frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            member_count,
            f"fingerprint-{cluster_id}",
        ),
    )
    conn.commit()


def test_assignment_metrics_use_weighted_signal_cosine_distance() -> None:
    signal_weights = get_domain_config("commerce")["signal_weights"]
    metrics = _assignment_metrics(
        {
            "centroid_vector": make_vector(1.0, 0.0, 0.0),
            "topic_label": "Price shock",
            "concern_level": 1.0,
            "purchase_intent": 0.0,
            "avoidance_signals": 0.0,
        },
        {
            "baseline_centroid_vector": make_vector(1.0, 0.0, 0.0),
            "baseline_topic_label": "Demand rebound",
            "concern_level": 0.0,
            "purchase_intent": 1.0,
            "avoidance_signals": 0.0,
        },
        signal_weights,
    )

    assert metrics is not None
    assert metrics["centroid_distance"] == 0.0
    assert metrics["segment_vector_distance"] == 1.0
    assert metrics["assignment_cost"] == 0.25


def test_run_cluster_drift_requires_stage_four_cluster_signals() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_cluster_run("run-missing-signals", n_clusters=1)
        insert_cluster("run-missing-signals")

        with pytest.raises(RuntimeError):
            run_cluster_drift("run-missing-signals")
    finally:
        cleanup_temp_db(temp_dir)


def test_run_cluster_drift_reuses_track_ids_across_similar_runs() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_cluster_run("run-1", n_clusters=1, n_articles=3)
        cluster_one = insert_cluster(
            "run-1",
            centroid_vector=make_vector(1.0, 0.0, 0.0),
        )
        insert_cluster_signal(cluster_one, run_id="run-1", topic_label="Prices spike")

        first_result = run_cluster_drift("run-1")
        assert first_result["new_tracks"] == 1
        assert first_result["matched_tracks"] == 0

        conn = db.init.get_conn()
        first_track_id = conn.execute(
            "SELECT track_id FROM cluster_tracks ORDER BY first_seen_at ASC LIMIT 1"
        ).fetchone()[0]

        insert_cluster_run("run-2", n_clusters=1, n_articles=4)
        cluster_two = insert_cluster(
            "run-2",
            centroid_vector=make_vector(0.98, 0.02, 0.0),
        )
        insert_cluster_signal(
            cluster_two,
            run_id="run-2",
            topic_label="Prices spike",
            concern=0.74,
            purchase=0.18,
            avoidance=0.31,
            member_count=4,
        )

        second_result = run_cluster_drift("run-2")
        assert second_result["matched_tracks"] == 1
        observation_track_id = conn.execute(
            """
            SELECT track_id
            FROM cluster_drift_observations
            WHERE run_id = ? AND match_type = 'matched'
            """,
            ("run-2",),
        ).fetchone()[0]
        assert observation_track_id == first_track_id

        latest = get_cluster_drift("run-2")
        assert latest["segments"][0]["topic"] == "inflace"
        assert latest["clusters"][0]["match_type"] == "matched"
    finally:
        cleanup_temp_db(temp_dir)


def test_cluster_drift_backfills_canonical_scope_for_legacy_runs() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_cluster_run("run-energy", topic="energie", n_clusters=1, n_articles=3)
        cluster_id = insert_cluster("run-energy")
        insert_cluster_signal(cluster_id, run_id="run-energy", topic_label="Energy prices")

        run_cluster_drift("run-energy")

        conn = db.init.get_conn()
        run_canonical = conn.execute(
            "SELECT canonical_topic_id FROM cluster_runs WHERE run_id = ?",
            ("run-energy",),
        ).fetchone()[0]
        track_canonical = conn.execute(
            "SELECT canonical_topic_id FROM cluster_tracks WHERE topic = ?",
            ("energie",),
        ).fetchone()[0]
        latest = get_latest_cluster_drift(topic="energy")
    finally:
        cleanup_temp_db(temp_dir)

    assert run_canonical == "energy"
    assert track_canonical == "energy"
    assert latest is not None
    assert latest["run_id"] == "run-energy"
    assert latest["topic"] == "energie"
    assert latest["canonical_topic_id"] == "energy"


def test_run_cluster_drift_is_idempotent_for_same_run() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_cluster_run("run-idempotent", n_clusters=1, n_articles=3)
        cluster_id = insert_cluster("run-idempotent")
        insert_cluster_signal(cluster_id, run_id="run-idempotent")

        first = run_cluster_drift("run-idempotent")
        second = run_cluster_drift("run-idempotent")

        conn = db.init.get_conn()
        drift_runs = conn.execute("SELECT COUNT(*) FROM cluster_drift_runs").fetchone()[0]
        observations = conn.execute(
            "SELECT COUNT(*) FROM cluster_drift_observations WHERE run_id = ?",
            ("run-idempotent",),
        ).fetchone()[0]

        assert second == first
        assert drift_runs == 1
        assert observations == 1
    finally:
        cleanup_temp_db(temp_dir)


def test_run_cluster_drift_inactivates_tracks_after_three_missed_runs() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_cluster_run("run-1", n_clusters=1, n_articles=3)
        cluster_id = insert_cluster("run-1")
        insert_cluster_signal(cluster_id, run_id="run-1", member_count=3)
        run_cluster_drift("run-1")

        for index in range(2, 5):
            run_id = f"run-{index}"
            insert_cluster_run(
                run_id,
                status="all_noise",
                n_articles=0,
                n_clusters=0,
                n_noise=0,
            )
            result = run_cluster_drift(run_id)
            assert result["missing_tracks"] == 1

        conn = db.init.get_conn()
        status, missed_runs = conn.execute(
            "SELECT status, missed_runs FROM cluster_tracks LIMIT 1"
        ).fetchone()
        assert status == "inactive"
        assert missed_runs == 3
    finally:
        cleanup_temp_db(temp_dir)
