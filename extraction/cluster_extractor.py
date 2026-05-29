from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from config.settings import (
    CLUSTER_SIGNAL_BODY_CHAR_LIMIT,
    CLUSTER_SIGNAL_MAX_EXEMPLARS,
    CLUSTER_SIGNAL_MIN_EXEMPLARS,
)
from db.init import get_conn
from extraction.cluster_llm_client import (
    extract_cluster_signals,
    get_cluster_signal_model,
    get_cluster_signal_provider,
)


SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class ClusterMember:
    article_id: str
    embedding_id: int
    embedding_vector: list[float]
    membership_strength: float
    title: str
    summary: str
    body: str
    outlet: str
    url: str
    published_at: str | None


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: int
    run_id: str
    cluster_label: int
    size: int
    centroid_vector: list[float]
    members: list[ClusterMember]


def _decode_vector(raw_vector: str) -> list[float]:
    parsed = json.loads(raw_vector)
    if not isinstance(parsed, list):
        raise ValueError("Embedding vector must decode to a JSON list.")
    return [float(value) for value in parsed]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _published_rank(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _collapse_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _body_excerpt(body: str, limit: int = CLUSTER_SIGNAL_BODY_CHAR_LIMIT) -> str:
    collapsed = _collapse_text(body)
    if len(collapsed) <= limit:
        return collapsed
    clipped = collapsed[: limit - 3].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _membership_fingerprint(members: list[ClusterMember]) -> str:
    payload = [
        {
            "article_id": member.article_id,
            "embedding_id": member.embedding_id,
            "membership_strength": round(member.membership_strength, 6),
        }
        for member in sorted(members, key=lambda item: (item.article_id, item.embedding_id))
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _select_exemplars(
    members: list[ClusterMember],
    centroid_vector: list[float],
) -> list[ClusterMember]:
    if not members:
        return []

    desired = min(len(members), CLUSTER_SIGNAL_MAX_EXEMPLARS)
    if len(members) >= CLUSTER_SIGNAL_MIN_EXEMPLARS:
        desired = max(desired, CLUSTER_SIGNAL_MIN_EXEMPLARS)

    ranked = sorted(
        members,
        key=lambda member: (
            -_cosine_similarity(member.embedding_vector, centroid_vector),
            -member.membership_strength,
            -_published_rank(member.published_at),
            member.article_id,
        ),
    )
    return ranked[:desired]


def _cluster_run_exists(run_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM cluster_runs WHERE run_id = ? LIMIT 1",
        (run_id,),
    ).fetchone()
    return row is not None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cluster_targets(run_id: str, min_cluster_size: int) -> list[ClusterTarget]:
    conn = get_conn()
    cluster_rows = conn.execute(
        """
        SELECT id, run_id, cluster_label, size, centroid_vector
        FROM clusters
        WHERE run_id = ?
          AND size >= ?
        ORDER BY size DESC, cluster_label ASC
        """,
        (run_id, min_cluster_size),
    ).fetchall()
    if not cluster_rows:
        return []

    centroids_by_cluster = {
        int(row[0]): {
            "run_id": row[1],
            "cluster_label": int(row[2]),
            "size": int(row[3]),
            "centroid_vector": _decode_vector(row[4]),
        }
        for row in cluster_rows
    }
    member_rows = conn.execute(
        """
        SELECT
            cm.cluster_id,
            cm.article_id,
            cm.embedding_id,
            COALESCE(cm.membership_strength, 0.0),
            ae.embedding_vector,
            a.title,
            a.summary,
            a.body,
            a.outlet,
            a.url,
            COALESCE(a.published_at, a.fetched_at)
        FROM cluster_memberships cm
        JOIN article_embeddings ae ON ae.id = cm.embedding_id
        JOIN articles a ON a.id = cm.article_id
        WHERE cm.run_id = ?
          AND cm.cluster_id IS NOT NULL
          AND cm.is_noise = 0
        ORDER BY cm.cluster_id ASC,
                 COALESCE(cm.membership_strength, 0.0) DESC,
                 COALESCE(a.published_at, a.fetched_at) DESC,
                 a.id ASC
        """,
        (run_id,),
    ).fetchall()

    members_by_cluster: dict[int, list[ClusterMember]] = {}
    for row in member_rows:
        cluster_id = int(row[0])
        if cluster_id not in centroids_by_cluster:
            continue
        members_by_cluster.setdefault(cluster_id, []).append(
            ClusterMember(
                article_id=row[1],
                embedding_id=int(row[2]),
                membership_strength=float(row[3] or 0.0),
                embedding_vector=_decode_vector(row[4]),
                title=row[5] or "",
                summary=row[6] or "",
                body=row[7] or "",
                outlet=row[8] or "",
                url=row[9] or "",
                published_at=row[10],
            )
        )

    targets: list[ClusterTarget] = []
    for cluster_id, cluster_data in centroids_by_cluster.items():
        members = members_by_cluster.get(cluster_id, [])
        targets.append(
            ClusterTarget(
                cluster_id=cluster_id,
                run_id=cluster_data["run_id"],
                cluster_label=cluster_data["cluster_label"],
                size=cluster_data["size"],
                centroid_vector=cluster_data["centroid_vector"],
                members=members,
            )
        )
    return targets


def _existing_cluster_signal_ids(run_id: str) -> set[int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT cluster_id FROM cluster_signals WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return {int(row[0]) for row in rows}


def _serialize_exemplars(exemplars: list[ClusterMember]) -> list[dict[str, object]]:
    return [
        {
            "article_id": member.article_id,
            "outlet": member.outlet,
            "title": _collapse_text(member.title),
            "summary": _collapse_text(member.summary),
            "body_excerpt": _body_excerpt(member.body),
            "url": member.url,
        }
        for member in exemplars
    ]


def run_cluster_extraction(
    run_id: str,
    *,
    overwrite: bool = False,
    min_cluster_size: int = 3,
) -> dict[str, object]:
    started_at = time.perf_counter()
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id is required.")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1.")
    if not _cluster_run_exists(normalized_run_id):
        raise LookupError(f"Cluster run '{normalized_run_id}' was not found.")

    targets = _load_cluster_targets(normalized_run_id, min_cluster_size)
    existing_ids = set() if overwrite else _existing_cluster_signal_ids(normalized_run_id)
    processed = 0
    skipped_existing = 0
    failed = 0
    conn = get_conn()

    for target in targets:
        if not overwrite and target.cluster_id in existing_ids:
            skipped_existing += 1
            continue

        exemplars = _select_exemplars(target.members, target.centroid_vector)
        if not exemplars:
            failed += 1
            continue

        try:
            signal = extract_cluster_signals(_serialize_exemplars(exemplars))
        except Exception:
            failed += 1
            continue

        conn.execute(
            """
            INSERT OR REPLACE INTO cluster_signals
            (cluster_id, run_id, topic_label, concern_level, purchase_intent,
             avoidance_signals, sentiment, dominant_frame, frame_detail,
             seg_young_urban, seg_family, seg_senior, seg_b2b, evidence_json,
             raw_json, member_count, membership_fingerprint, exemplar_article_ids,
             extractor_provider, extractor_model, schema_version, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target.cluster_id,
                normalized_run_id,
                signal["topic_label"],
                signal["concern_level"],
                signal["purchase_intent"],
                signal["avoidance_signals"],
                signal["sentiment"],
                signal["dominant_frame"],
                signal["frame_detail"],
                signal["seg_young_urban"],
                signal["seg_family"],
                signal["seg_senior"],
                signal["seg_b2b"],
                json.dumps(signal.get("evidence", []), ensure_ascii=False),
                json.dumps(signal.get("raw_json", {}), ensure_ascii=False),
                len(target.members),
                _membership_fingerprint(target.members),
                json.dumps([member.article_id for member in exemplars], ensure_ascii=False),
                get_cluster_signal_provider(),
                get_cluster_signal_model(),
                SCHEMA_VERSION,
                _utc_now_iso(),
            ),
        )
        processed += 1

    return {
        "run_id": normalized_run_id,
        "selected_clusters": len(targets),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "provider": get_cluster_signal_provider(),
        "model_name": get_cluster_signal_model(),
        "duration_s": round(time.perf_counter() - started_at, 4),
    }
