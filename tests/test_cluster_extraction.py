#!/usr/bin/env python3
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import db.init
from api.routes.signals import get_signals
from clustering.clustering_service import get_latest_cluster_run
from delta.engine import compute_drift
from delta.mapper import compute_segment_profiles
from delta.seeder import seed_baselines
from extraction.cluster_extractor import (
    ClusterMember,
    _coherence_score,
    _membership_fingerprint,
    _select_exemplars,
    run_cluster_extraction,
)
from extraction.cluster_llm_client import (
    _normalize_cluster_signal,
    get_cluster_signal_model,
    get_cluster_signal_provider,
)
from extraction.embedder import get_expected_dim, get_model_name


ORIGINAL_DB_PATH = db.init.DB_PATH
RECENT_TS = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    dim = get_expected_dim()
    values = [0.0] * dim
    for index, value in enumerate(leading):
        values[index] = value
    return values


def insert_article(
    article_id: str,
    *,
    topic: str = "inflace",
    outlet: str = "unit-test",
    title: str | None = None,
    summary: str | None = None,
    body: str | None = None,
    published_at: str = RECENT_TS,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'CZ', 'cs', ?, ?)
        """,
        (
            article_id,
            outlet,
            title or f"Title {article_id}",
            summary or f"Summary {article_id}",
            body or f"Body for {article_id}",
            f"https://example.test/{article_id}",
            f"https://example.test/{article_id}",
            topic,
            published_at,
            RECENT_TS,
        ),
    )
    conn.commit()


def insert_article_signal(
    article_id: str,
    *,
    topic: str = "inflace",
    concern: float = 0.8,
    purchase: float = 0.1,
    avoidance: float = 0.2,
    frame: str = "fear",
    seg_young_urban: float = 1.0,
    seg_family: float = 0.0,
    seg_senior: float = 0.0,
    seg_b2b: float = 0.0,
) -> None:
    insert_article(article_id, topic=topic)
    raw_json = {
        "concern_level": concern,
        "purchase_intent": purchase,
        "avoidance_signals": avoidance,
        "dominant_frame": frame,
        "seg_young_urban": seg_young_urban,
        "seg_family": seg_family,
        "seg_senior": seg_senior,
        "seg_b2b": seg_b2b,
    }
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            concern,
            purchase,
            avoidance,
            frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            json.dumps(raw_json),
            RECENT_TS,
        ),
    )
    conn.commit()


def insert_embedding(
    article_id: str,
    vector: list[float],
    *,
    embedding_text: str,
    embedded_at: str = "2026-03-18T10:20:00+00:00",
) -> int:
    conn = db.init.get_conn()
    text_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    cursor = conn.execute(
        """
        INSERT INTO article_embeddings
        (article_id, model_name, model_version, embedding_dim, embedding_vector,
         embedding_text, embedding_text_hash, language, status, error_message,
         embedded_at, created_at, updated_at)
        VALUES (?, ?, 'test-version', ?, ?, ?, ?, 'cs', 'complete', NULL, ?, ?, ?)
        """,
        (
            article_id,
            get_model_name(),
            get_expected_dim(),
            json.dumps(vector),
            embedding_text,
            text_hash,
            embedded_at,
            embedded_at,
            embedded_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_cluster_run(
    run_id: str,
    *,
    topic: str = "inflace",
    status: str = "completed",
    n_articles: int = 6,
    n_clusters: int = 2,
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
        VALUES (?, ?, '', '', 'cs', '2026-03-18T00:00:00+00:00', '2026-03-19T00:00:00+00:00',
                ?, ?, ?, ?, 8, 10, 3, 2, ?, 'test-version')
        """,
        (run_id, topic, status, n_articles, n_clusters, n_noise, get_model_name()),
    )
    conn.commit()


def insert_cluster(run_id: str, cluster_label: int, size: int, centroid_vector: list[float]) -> int:
    conn = db.init.get_conn()
    cursor = conn.execute(
        """
        INSERT INTO clusters
        (run_id, cluster_label, size, centroid_vector, centroid_dim)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, cluster_label, size, json.dumps(centroid_vector), len(centroid_vector)),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_membership(
    run_id: str,
    cluster_id: int,
    article_id: str,
    embedding_id: int,
    *,
    membership_strength: float = 0.99,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO cluster_memberships
        (run_id, cluster_id, article_id, embedding_id, membership_strength, is_noise)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (run_id, cluster_id, article_id, embedding_id, membership_strength),
    )
    conn.commit()


def valid_cluster_signal_payload(topic_label: str = "Inflation cluster") -> dict[str, object]:
    return {
        "topic_label": topic_label,
        "concern_level": 0.7,
        "purchase_intent": 0.2,
        "avoidance_signals": 0.3,
        "sentiment": -0.2,
        "dominant_frame": "fear",
        "frame_detail": "uncertainty",
        "seg_young_urban": 0.4,
        "seg_family": 0.3,
        "seg_senior": 0.2,
        "seg_b2b": 0.1,
        "evidence": ["Prices remain volatile.", "Households are cutting spending."],
        "raw_json": {"ok": True},
    }


def member(
    article_id: str,
    vector: list[float],
    *,
    membership_strength: float = 0.5,
    published_at: str = "2026-03-18T10:00:00+00:00",
) -> ClusterMember:
    return ClusterMember(
        article_id=article_id,
        embedding_id=1,
        embedding_vector=vector,
        membership_strength=membership_strength,
        title=f"Title {article_id}",
        summary=f"Summary {article_id}",
        body=f"Body {article_id}",
        outlet="unit-test",
        url=f"https://example.test/{article_id}",
        published_at=published_at,
    )


def test_normalize_cluster_signal_clamps_and_normalizes() -> None:
    result = _normalize_cluster_signal(
        {
            "topic_label": "This topic label is far too long for the schema to keep intact",
            "concern_level": 9.0,
            "purchase_intent": -2.0,
            "avoidance_signals": "bad",
            "sentiment": -8.0,
            "dominant_frame": "unsupported",
            "frame_detail": "moral concern",
            "seg_young_urban": 0.0,
            "seg_family": 0.2,
            "seg_senior": 2.0,
            "seg_b2b": -1.0,
            "evidence": ["  First evidence.  ", "", "Third evidence."],
        },
        fallback_topic_label="Fallback topic",
    )

    assert len(result["topic_label"].split()) <= 8
    assert result["concern_level"] == 1.0
    assert result["purchase_intent"] == 0.0
    assert result["avoidance_signals"] == 0.0
    assert result["sentiment"] == -1.0
    assert result["frame_detail"] == "moral_concern"
    assert result["dominant_frame"] == "fear"
    assert result["evidence"] == ["First evidence.", "Third evidence."]
    assert round(sum(result[key] for key in ("seg_young_urban", "seg_family", "seg_senior", "seg_b2b")), 4) == 1.0


def test_membership_fingerprint_is_order_invariant() -> None:
    members_a = [
        member("a-1", make_vector(1.0, 0.0), membership_strength=0.8),
        member("a-2", make_vector(0.8, 0.2), membership_strength=0.6),
    ]
    members_b = list(reversed(members_a))

    assert _membership_fingerprint(members_a) == _membership_fingerprint(members_b)


def test_coherence_score_averages_centroid_similarity() -> None:
    score = _coherence_score(
        [
            member("a-1", make_vector(1.0, 0.0), membership_strength=0.8),
            member("a-2", make_vector(0.0, 1.0), membership_strength=0.6),
        ],
        make_vector(1.0, 0.0),
    )

    assert score == 0.5


def test_select_exemplars_prefers_similarity_then_strength_then_recency() -> None:
    centroid = make_vector(1.0, 0.0)
    exemplars = _select_exemplars(
        [
            member("a", make_vector(1.0, 0.0), membership_strength=0.1, published_at="2026-03-18T08:00:00+00:00"),
            member("b", make_vector(0.95, 0.31), membership_strength=0.9, published_at="2026-03-18T10:00:00+00:00"),
            member("c", make_vector(0.90, 0.44), membership_strength=0.8, published_at="2026-03-18T09:00:00+00:00"),
            member("d", make_vector(0.95, 0.31), membership_strength=0.4, published_at="2026-03-18T11:00:00+00:00"),
        ],
        centroid,
    )

    assert [item.article_id for item in exemplars] == ["a", "b", "d", "c"]


def test_run_cluster_extraction_persists_one_row_per_eligible_cluster() -> None:
    temp_dir = setup_temp_db()
    try:
        run_id = "run-stage4"
        insert_cluster_run(run_id)
        cluster_a_id = insert_cluster(run_id, 0, 3, make_vector(1.0, 0.0))
        cluster_b_id = insert_cluster(run_id, 1, 3, make_vector(0.0, 1.0))
        for article_id, vector in (
            ("a1", make_vector(1.0, 0.0)),
            ("a2", make_vector(0.9, 0.1)),
            ("a3", make_vector(0.8, 0.2)),
            ("b1", make_vector(0.0, 1.0)),
            ("b2", make_vector(0.1, 0.9)),
            ("b3", make_vector(0.2, 0.8)),
        ):
            insert_article(article_id, body=f"Body {article_id} " * 20)
            embedding_id = insert_embedding(article_id, vector, embedding_text=article_id)
            if article_id.startswith("a"):
                cluster_id = cluster_a_id
            else:
                cluster_id = cluster_b_id
            insert_membership(run_id, cluster_id, article_id, embedding_id)

        with patch(
            "extraction.cluster_extractor.extract_cluster_signals",
            side_effect=[
                valid_cluster_signal_payload("Cluster A"),
                valid_cluster_signal_payload("Cluster B"),
            ],
        ) as mock_extract:
            result = run_cluster_extraction(run_id)
        latest = get_latest_cluster_run(topic="inflace", language="cs")

        rows = db.init.get_conn().execute(
            """
            SELECT cluster_id, topic_label, exemplar_article_ids, coherence_score,
                   extractor_provider, extractor_model
            FROM cluster_signals
            WHERE run_id = ?
            ORDER BY cluster_id ASC
            """,
            (run_id,),
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert result["selected_clusters"] == 2
    assert result["processed"] == 2
    assert result["failed"] == 0
    assert mock_extract.call_count == 2
    assert [row[0] for row in rows] == [cluster_a_id, cluster_b_id]
    assert rows[0][1] == "Cluster A"
    assert len(json.loads(rows[0][2])) == 3
    assert rows[0][3] > 0.95
    assert rows[0][4] == get_cluster_signal_provider()
    assert rows[0][5] == get_cluster_signal_model()
    assert latest is not None
    assert latest["clusters"][0]["signal"]["topic_label"] == "Cluster A"
    assert latest["clusters"][0]["signal"]["coherence_score"] > 0.95


def test_run_cluster_extraction_is_idempotent_until_overwrite() -> None:
    temp_dir = setup_temp_db()
    try:
        run_id = "run-idempotent"
        insert_cluster_run(run_id, n_articles=3, n_clusters=1)
        cluster_id = insert_cluster(run_id, 0, 3, make_vector(1.0, 0.0))
        for article_id, vector in (
            ("c1", make_vector(1.0, 0.0)),
            ("c2", make_vector(0.9, 0.1)),
            ("c3", make_vector(0.8, 0.2)),
        ):
            insert_article(article_id)
            embedding_id = insert_embedding(article_id, vector, embedding_text=article_id)
            insert_membership(run_id, cluster_id, article_id, embedding_id)

        with patch(
            "extraction.cluster_extractor.extract_cluster_signals",
            side_effect=[
                valid_cluster_signal_payload("First pass"),
                valid_cluster_signal_payload("Overwrite pass"),
            ],
        ) as mock_extract:
            first = run_cluster_extraction(run_id)
            second = run_cluster_extraction(run_id)
            third = run_cluster_extraction(run_id, overwrite=True)

        row = db.init.get_conn().execute(
            "SELECT topic_label FROM cluster_signals WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert first["processed"] == 1
    assert second["processed"] == 0
    assert second["skipped_existing"] == 1
    assert third["processed"] == 1
    assert mock_extract.call_count == 2
    assert row == ("Overwrite pass",)


def test_cluster_signals_do_not_change_existing_signal_consumers() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_signal("signal-1")
        run_id = "run-safety"
        insert_cluster_run(run_id, n_articles=1, n_clusters=1)
        cluster_id = insert_cluster(run_id, 0, 1, make_vector(1.0, 0.0))
        db.init.get_conn().execute(
            """
            INSERT INTO cluster_signals
            (cluster_id, run_id, topic_label, concern_level, purchase_intent,
             avoidance_signals, sentiment, dominant_frame, frame_detail,
             seg_young_urban, seg_family, seg_senior, seg_b2b, evidence_json,
             raw_json, member_count, membership_fingerprint, exemplar_article_ids,
             extractor_provider, extractor_model, schema_version, extracted_at)
            VALUES (?, ?, 'Cluster', 0.1, 0.2, 0.3, 0.0, 'neutral', 'neutral',
                    0.25, 0.25, 0.25, 0.25, '[]', '{}', 1, 'fp', '[]',
                    ?, ?, 'v1', '2026-03-18T10:20:00+00:00')
            """,
            (cluster_id, run_id, get_cluster_signal_provider(), get_cluster_signal_model()),
        )
        profiles = compute_segment_profiles("inflace")
        drift = compute_drift("inflace")
        rows = get_signals("inflace")
    finally:
        cleanup_temp_db(temp_dir)

    profile_by_segment = {profile["segment"]: profile for profile in profiles}
    drift_by_segment = {entry["segment"]: entry for entry in drift}
    assert len(rows) == 1
    assert rows[0]["article_id"] == "signal-1"
    assert profile_by_segment["young_urban"]["concern_level"] == 0.8
    assert profile_by_segment["young_urban"]["article_count"] == 1
    assert drift_by_segment["young_urban"]["current"]["concern_level"] == 0.8
