from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from clustering.centroid import compute_centroids
from clustering.hdbscan_clusterer import cluster_reduced
from clustering.umap_reducer import reduce_embeddings
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from extraction.embedder import (
    get_expected_dim,
    get_model_name,
    get_model_version,
)


DEFAULT_WINDOW_HOURS = 24
DEFAULT_UMAP_COMPONENTS = 10
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_SAMPLES = 2
MIN_ARTICLES_FOR_CLUSTERING = 5
UMAP_RANDOM_STATE = 42

RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_ALL_NOISE = "all_noise"
RUN_STATUS_SKIPPED_SMALL_SAMPLE = "skipped_small_sample"


@dataclass(frozen=True)
class SelectedEmbedding:
    article_id: str
    embedding_id: int
    vector: list[float]
    model_version: str | None


def _normalize_topic(topic: str) -> str:
    normalized = topic.strip()
    if not normalized:
        raise ValueError("topic is required for clustering.")
    return normalized


def _normalize_country(country: str | None) -> str:
    return (country or "").strip().upper()


def _normalize_source(source: str | None) -> str:
    return (source or "").strip().lower()


def _normalize_language(language: str | None) -> str | None:
    normalized = (language or "").strip().lower()
    return normalized or None


def _window_bounds(window_hours: int) -> tuple[str, str]:
    if window_hours < 1:
        raise ValueError("window_hours must be >= 1.")
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=window_hours)
    return window_start.isoformat(), window_end.isoformat()


def _parse_vector(raw_vector: str) -> list[float]:
    parsed = json.loads(raw_vector)
    if not isinstance(parsed, list):
        raise ValueError("Embedding vector must decode to a JSON list.")

    vector = [float(value) for value in parsed]
    expected_dim = get_expected_dim()
    if len(vector) != expected_dim:
        raise AssertionError(
            f"Unexpected embedding dim {len(vector)} for {get_model_name()}"
        )
    return vector


def _safe_json_object(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_list(raw: str | None) -> list[object]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _select_embeddings(
    *,
    topic: str,
    country: str,
    source: str,
    language: str | None,
    window_start: str,
    window_end: str,
) -> list[SelectedEmbedding]:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT
            a.id,
            e.id,
            e.embedding_vector,
            e.model_version
        FROM articles a
        JOIN article_embeddings e
          ON e.id = (
                SELECT e2.id
                FROM article_embeddings e2
                WHERE e2.article_id = a.id
                  AND e2.model_name = ?
                  AND e2.status = 'complete'
                ORDER BY COALESCE(e2.embedded_at, e2.updated_at, e2.created_at) DESC,
                         e2.id DESC
                LIMIT 1
          )
        WHERE 1 = 1
          {topic_sql}
    """
    params: list[object] = [get_model_name(), *topic_params]
    if country:
        query += " AND a.country = ?"
        params.append(country)
    if source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(source)
    if language is not None:
        query += " AND LOWER(a.language) = ?"
        params.append(language)
    query += """
        AND COALESCE(a.published_at, a.fetched_at) >= ?
        AND COALESCE(a.published_at, a.fetched_at) <= ?
        ORDER BY COALESCE(a.published_at, a.fetched_at) DESC, a.id DESC
    """
    params.extend([window_start, window_end])

    rows = conn.execute(query, params).fetchall()
    return [
        SelectedEmbedding(
            article_id=row[0],
            embedding_id=int(row[1]),
            vector=_parse_vector(row[2]),
            model_version=row[3],
        )
        for row in rows
    ]


def _resolve_model_version(rows: list[SelectedEmbedding]) -> str | None:
    versions = {row.model_version for row in rows if row.model_version}
    if len(versions) == 1:
        return next(iter(versions))
    if len(versions) > 1:
        return "mixed"
    return get_model_version()


def _effective_umap_params(
    n_articles: int,
    *,
    requested_components: int,
    requested_neighbors: int,
) -> tuple[int, int]:
    n_components = max(1, min(requested_components, n_articles - 2))
    n_neighbors = max(2, min(requested_neighbors, n_articles - 1))
    return n_components, n_neighbors


def _insert_run(
    *,
    run_id: str,
    topic: str,
    country: str,
    source: str,
    language: str | None,
    window_start: str,
    window_end: str,
    status: str,
    n_articles: int,
    n_clusters: int,
    n_noise: int,
    umap_n_components: int,
    umap_n_neighbors: int,
    min_cluster_size: int,
    min_samples: int,
    model_version: str | None,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO cluster_runs
        (run_id, topic, country, source, language, window_start, window_end,
         status, n_articles, n_clusters, n_noise, umap_n_components,
         umap_n_neighbors, hdbscan_min_cluster_size, hdbscan_min_samples,
         model_name, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            topic,
            country,
            source,
            language,
            window_start,
            window_end,
            status,
            n_articles,
            n_clusters,
            n_noise,
            umap_n_components,
            umap_n_neighbors,
            min_cluster_size,
            min_samples,
            get_model_name(),
            model_version,
        ),
    )


def _persist_clustering_result(
    *,
    run_id: str,
    rows: list[SelectedEmbedding],
    labels: np.ndarray,
    probabilities: np.ndarray,
    centroids: dict[int, list[float]],
) -> None:
    conn = get_conn()
    cluster_ids: dict[int, int] = {}
    for label in sorted(centroids):
        centroid = centroids[label]
        size = int(np.sum(labels == label))
        cursor = conn.execute(
            """
            INSERT INTO clusters
            (run_id, cluster_label, size, centroid_vector, centroid_dim)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, label, size, json.dumps(centroid), len(centroid)),
        )
        cluster_ids[label] = int(cursor.lastrowid)

    for row, label, probability in zip(rows, labels, probabilities):
        cluster_id = None if int(label) == -1 else cluster_ids[int(label)]
        conn.execute(
            """
            INSERT INTO cluster_memberships
            (run_id, cluster_id, article_id, embedding_id, membership_strength, is_noise)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                cluster_id,
                row.article_id,
                row.embedding_id,
                float(probability),
                int(label == -1),
            ),
        )


def run_clustering(
    *,
    topic: str,
    country: str | None = None,
    source: str | None = None,
    language: str | None = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, object]:
    started_at = time.perf_counter()
    normalized_topic = _normalize_topic(topic)
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    window_start, window_end = _window_bounds(window_hours)
    rows = _select_embeddings(
        topic=normalized_topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        window_start=window_start,
        window_end=window_end,
    )
    run_id = uuid.uuid4().hex
    model_version = _resolve_model_version(rows)
    conn = get_conn()

    if len(rows) < MIN_ARTICLES_FOR_CLUSTERING:
        conn.execute("BEGIN")
        try:
            _insert_run(
                run_id=run_id,
                topic=normalized_topic,
                country=normalized_country,
                source=normalized_source,
                language=normalized_language,
                window_start=window_start,
                window_end=window_end,
                status=RUN_STATUS_SKIPPED_SMALL_SAMPLE,
                n_articles=len(rows),
                n_clusters=0,
                n_noise=0,
                umap_n_components=0,
                umap_n_neighbors=0,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                model_version=model_version,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return {
            "run_id": run_id,
            "topic": normalized_topic,
            "country": normalized_country,
            "source": normalized_source,
            "language": normalized_language,
            "window_start": window_start,
            "window_end": window_end,
            "status": RUN_STATUS_SKIPPED_SMALL_SAMPLE,
            "n_articles": len(rows),
            "n_clusters": 0,
            "n_noise": 0,
            "model_name": get_model_name(),
            "model_version": model_version,
            "umap_n_components": 0,
            "umap_n_neighbors": 0,
            "hdbscan_min_cluster_size": min_cluster_size,
            "hdbscan_min_samples": min_samples,
            "duration_s": round(time.perf_counter() - started_at, 4),
        }

    original_vectors = np.asarray([row.vector for row in rows], dtype=float)
    n_components, n_neighbors = _effective_umap_params(
        len(rows),
        requested_components=umap_components,
        requested_neighbors=umap_neighbors,
    )
    reduced = reduce_embeddings(
        original_vectors,
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=UMAP_RANDOM_STATE,
    )
    labels, probabilities = cluster_reduced(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    centroids = compute_centroids(labels, original_vectors)
    n_noise = int(np.sum(labels == -1))
    n_clusters = len(centroids)
    status = RUN_STATUS_ALL_NOISE if n_clusters == 0 else RUN_STATUS_COMPLETED

    conn.execute("BEGIN")
    try:
        _insert_run(
            run_id=run_id,
            topic=normalized_topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            window_start=window_start,
            window_end=window_end,
            status=status,
            n_articles=len(rows),
            n_clusters=n_clusters,
            n_noise=n_noise,
            umap_n_components=n_components,
            umap_n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            model_version=model_version,
        )
        _persist_clustering_result(
            run_id=run_id,
            rows=rows,
            labels=labels,
            probabilities=probabilities,
            centroids=centroids,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return {
        "run_id": run_id,
        "topic": normalized_topic,
        "country": normalized_country,
        "source": normalized_source,
        "language": normalized_language,
        "window_start": window_start,
        "window_end": window_end,
        "status": status,
        "n_articles": len(rows),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "model_name": get_model_name(),
        "model_version": model_version,
        "umap_n_components": n_components,
        "umap_n_neighbors": n_neighbors,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": min_samples,
        "duration_s": round(time.perf_counter() - started_at, 4),
    }


def get_latest_cluster_run(
    *,
    topic: str,
    country: str | None = None,
    source: str | None = None,
    language: str | None = None,
) -> dict[str, object] | None:
    normalized_topic = _normalize_topic(topic)
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT
            run_id,
            topic,
            country,
            source,
            language,
            window_start,
            window_end,
            status,
            n_articles,
            n_clusters,
            n_noise,
            umap_n_components,
            umap_n_neighbors,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
            model_name,
            model_version,
            created_at
        FROM cluster_runs
        WHERE topic = ?
          AND country = ?
          AND source = ?
          AND ((? IS NULL AND language IS NULL) OR language = ?)
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (
            normalized_topic,
            normalized_country,
            normalized_source,
            normalized_language,
            normalized_language,
        ),
    ).fetchone()
    if row is None:
        return None

    run_id = row[0]
    cluster_rows = conn.execute(
        """
        SELECT id, cluster_label, size, centroid_vector, centroid_dim
        FROM clusters
        WHERE run_id = ?
        ORDER BY size DESC, cluster_label ASC
        """,
        (run_id,),
    ).fetchall()
    signal_rows = conn.execute(
        """
        SELECT
            cluster_id,
            topic_label,
            concern_level,
            purchase_intent,
            avoidance_signals,
            sentiment,
            dominant_frame,
            frame_detail,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            evidence_json,
            raw_json,
            member_count,
            membership_fingerprint,
            exemplar_article_ids,
            extractor_provider,
            extractor_model,
            schema_version,
            extracted_at
        FROM cluster_signals
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    membership_rows = conn.execute(
        """
        SELECT cluster_id, article_id, embedding_id, membership_strength, is_noise
        FROM cluster_memberships
        WHERE run_id = ?
        ORDER BY is_noise ASC, cluster_id ASC, membership_strength DESC, article_id ASC
        """,
        (run_id,),
    ).fetchall()
    signal_by_cluster = {
        int(row[0]): {
            "topic_label": row[1],
            "concern_level": row[2],
            "purchase_intent": row[3],
            "avoidance_signals": row[4],
            "sentiment": row[5],
            "dominant_frame": row[6],
            "frame_detail": row[7],
            "seg_young_urban": row[8],
            "seg_family": row[9],
            "seg_senior": row[10],
            "seg_b2b": row[11],
            "evidence": [str(item) for item in _safe_json_list(row[12]) if isinstance(item, str)],
            "raw_json": _safe_json_object(row[13]),
            "member_count": int(row[14]),
            "membership_fingerprint": row[15],
            "exemplar_article_ids": [
                str(item) for item in _safe_json_list(row[16]) if isinstance(item, str)
            ],
            "extractor_provider": row[17],
            "extractor_model": row[18],
            "schema_version": row[19],
            "extracted_at": row[20],
        }
        for row in signal_rows
    }

    members_by_cluster: dict[int, list[dict[str, object]]] = {}
    noise_members: list[dict[str, object]] = []
    for cluster_id, article_id, embedding_id, membership_strength, is_noise in membership_rows:
        member = {
            "article_id": article_id,
            "embedding_id": embedding_id,
            "membership_strength": membership_strength,
            "is_noise": bool(is_noise),
        }
        if bool(is_noise):
            noise_members.append(member)
            continue
        members_by_cluster.setdefault(int(cluster_id), []).append(member)

    clusters = [
        {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "size": size,
            "centroid_vector": json.loads(centroid_vector),
            "centroid_dim": centroid_dim,
            "members": members_by_cluster.get(int(cluster_id), []),
            "signal": signal_by_cluster.get(int(cluster_id)),
        }
        for cluster_id, cluster_label, size, centroid_vector, centroid_dim in cluster_rows
    ]

    return {
        "run_id": row[0],
        "topic": row[1],
        "country": row[2],
        "source": row[3],
        "language": row[4],
        "window_start": row[5],
        "window_end": row[6],
        "status": row[7],
        "n_articles": row[8],
        "n_clusters": row[9],
        "n_noise": row[10],
        "umap_n_components": row[11],
        "umap_n_neighbors": row[12],
        "hdbscan_min_cluster_size": row[13],
        "hdbscan_min_samples": row[14],
        "model_name": row[15],
        "model_version": row[16],
        "created_at": row[17],
        "clusters": clusters,
        "noise_members": noise_members,
    }
