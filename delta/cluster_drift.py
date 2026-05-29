from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone

from config.domains import get_domain_config, topic_to_domain
from config.settings import (
    BASELINE_EMA_ALPHA,
    CONFIDENCE_ARTICLE_DENOMINATOR,
    CONFIDENCE_BASELINE_WEIGHT,
    CONFIDENCE_COVERAGE_WEIGHT,
    MIN_ARTICLES_FOR_BASELINE,
    MIN_BRIEF_CONFIDENCE,
)
from db.init import get_conn
from delta.mapper import SEGMENTS, SIGNAL_KEYS, canonicalize_frame


ALERT_THRESHOLDS = {
    "mild": 0.20,
    "strong": 0.45,
}

ACTIVE_TRACK_STATUS = "active"
INACTIVE_TRACK_STATUS = "inactive"
MATCHED = "matched"
NEW = "new"
MISSING = "missing"
MATCH_MAX_CENTROID_DISTANCE = 0.25
MATCH_WEIGHT_CENTROID = 0.75
MATCH_WEIGHT_SEGMENT = 0.15
MATCH_WEIGHT_LABEL = 0.10
UNMATCHED_COST = 0.5
DISALLOWED_COST = 1_000_000.0
MAX_MISSED_RUNS = 3

REQUIRED_STAGE_FOUR_TABLES = (
    "cluster_runs",
    "clusters",
    "cluster_memberships",
    "cluster_signals",
)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _baseline_age_days(updated_at: str | None) -> int | None:
    if not updated_at:
        return None
    updated = datetime.fromisoformat(updated_at)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - updated.astimezone(timezone.utc)
    return max(0, delta.days)


def _alert_level(drift_magnitude: float, has_data: bool) -> str:
    if not has_data:
        return "no_data"
    if drift_magnitude >= ALERT_THRESHOLDS["strong"]:
        return "strong"
    if drift_magnitude >= ALERT_THRESHOLDS["mild"]:
        return "mild"
    return "none"


def _segment_status(has_data: bool, is_learned: bool, confidence: float) -> str:
    if not has_data:
        return "no_data"
    if is_learned and confidence >= MIN_BRIEF_CONFIDENCE:
        return "ready"
    return "warming"


def _normalize_label(value: str | None) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else " " for char in (value or "")
    )
    return " ".join(normalized.split())


def _label_penalty(current: str | None, baseline: str | None) -> float:
    current_label = _normalize_label(current)
    baseline_label = _normalize_label(baseline)
    if not current_label or not baseline_label:
        return 1.0
    if current_label == baseline_label:
        return 0.0
    if current_label in baseline_label or baseline_label in current_label:
        return 0.25
    current_tokens = set(current_label.split())
    baseline_tokens = set(baseline_label.split())
    union = current_tokens | baseline_tokens
    if not union:
        return 1.0
    overlap = len(current_tokens & baseline_tokens) / len(union)
    return round(1.0 - overlap, 4)


def _decode_vector(raw_vector: str) -> list[float]:
    parsed = json.loads(raw_vector)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list vector.")
    return [float(value) for value in parsed]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value / norm) for value in vector]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    if not left:
        return 1.0
    left_normalized = _normalize_vector(left)
    right_normalized = _normalize_vector(right)
    similarity = sum(a * b for a, b in zip(left_normalized, right_normalized))
    similarity = _clamp(similarity, -1.0, 1.0)
    return round(1.0 - similarity, 4)


def _weighted_signal_vector(values: dict[str, float], signal_weights: dict[str, float]) -> list[float]:
    weighted = [
        float(values.get(key, 0.0) or 0.0) * float(signal_weights.get(key, 0.0) or 0.0)
        for key in SIGNAL_KEYS
    ]
    return _normalize_vector(weighted)


def _signal_payload(record: dict[str, object]) -> dict[str, float]:
    return {
        key: round(float(record.get(key, 0.0) or 0.0), 4)
        for key in SIGNAL_KEYS
    }


def _segment_payload(record: dict[str, object], prefix: str = "") -> dict[str, float]:
    return {
        segment: round(float(record.get(f"{prefix}{segment}", 0.0) or 0.0), 4)
        for segment in SEGMENTS
    }


def _direction_from_deltas(
    deltas: dict[str, float],
    *,
    frame_shift: bool = False,
    match_type: str = MATCHED,
) -> str:
    if match_type == NEW:
        return "emerging"
    if match_type == MISSING:
        return "fading"

    weighted_total = sum(deltas.values())
    if weighted_total >= 0.08:
        return "rising"
    if weighted_total <= -0.08:
        return "falling"
    if frame_shift:
        return "mixed"
    if any(abs(value) >= 0.03 for value in deltas.values()):
        return "mixed"
    return "stable"


def _compute_observation_confidence(
    member_count: int,
    mean_membership_strength: float,
    sample_count: int,
    is_learned: bool,
) -> float:
    coverage = min(1.0, member_count / CONFIDENCE_ARTICLE_DENOMINATOR)
    membership_quality = _clamp(mean_membership_strength)
    coverage_quality = (coverage + membership_quality) / 2
    if not is_learned:
        baseline_quality = 0.3
    else:
        baseline_quality = min(1.0, sample_count / CONFIDENCE_ARTICLE_DENOMINATOR) + 0.3
        baseline_quality = min(1.0, baseline_quality)
    confidence = (
        CONFIDENCE_COVERAGE_WEIGHT * coverage_quality
        + CONFIDENCE_BASELINE_WEIGHT * baseline_quality
    )
    return round(_clamp(confidence), 4)


def _linear_sum_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    size = len(cost_matrix)
    if size == 0:
        return []

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: list[tuple[int, int]] = []
    for column in range(1, size + 1):
        if p[column] != 0:
            assignments.append((p[column] - 1, column - 1))
    return assignments


def _assert_stage_four_ready(run_id: str) -> dict[str, object]:
    conn = get_conn()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing_tables = [name for name in REQUIRED_STAGE_FOUR_TABLES if name not in tables]
    if missing_tables:
        raise RuntimeError(
            "Cluster drift requires completed stage-4 tables. Missing: "
            + ", ".join(sorted(missing_tables))
        )

    row = conn.execute(
        """
        SELECT run_id, topic, country, source, language, status, n_clusters
        FROM cluster_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"Cluster run '{run_id}' was not found.")

    cluster_count = int(row[6] or 0)
    signal_count = conn.execute(
        "SELECT COUNT(*) FROM cluster_signals WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    if cluster_count > 0 and signal_count < cluster_count:
        raise RuntimeError(
            "Cluster drift requires stage 4 extraction for every cluster in the run."
        )

    return {
        "run_id": row[0],
        "topic": row[1],
        "country": row[2],
        "source": row[3],
        "language": row[4],
        "status": row[5],
        "n_clusters": cluster_count,
    }


def _cluster_drift_exists(run_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM cluster_drift_runs WHERE run_id = ? LIMIT 1",
        (run_id,),
    ).fetchone()
    return row is not None


def _load_current_clusters(run_id: str) -> list[dict[str, object]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.cluster_label,
            c.size,
            c.centroid_vector,
            cs.topic_label,
            cs.concern_level,
            cs.purchase_intent,
            cs.avoidance_signals,
            cs.dominant_frame,
            cs.seg_young_urban,
            cs.seg_family,
            cs.seg_senior,
            cs.seg_b2b,
            COALESCE(cs.member_count, c.size),
            COALESCE(AVG(cm.membership_strength), 0.0)
        FROM clusters c
        JOIN cluster_signals cs ON cs.cluster_id = c.id
        LEFT JOIN cluster_memberships cm
          ON cm.cluster_id = c.id
         AND cm.run_id = c.run_id
         AND cm.is_noise = 0
        WHERE c.run_id = ?
        GROUP BY
            c.id, c.cluster_label, c.size, c.centroid_vector,
            cs.topic_label, cs.concern_level, cs.purchase_intent, cs.avoidance_signals,
            cs.dominant_frame, cs.seg_young_urban, cs.seg_family, cs.seg_senior,
            cs.seg_b2b, cs.member_count
        ORDER BY c.size DESC, c.cluster_label ASC
        """,
        (run_id,),
    ).fetchall()

    current_clusters: list[dict[str, object]] = []
    for row in rows:
        current_clusters.append(
            {
                "cluster_id": int(row[0]),
                "cluster_label": int(row[1]),
                "size": int(row[2]),
                "centroid_vector": _decode_vector(row[3]),
                "topic_label": row[4] or "",
                "concern_level": float(row[5] or 0.0),
                "purchase_intent": float(row[6] or 0.0),
                "avoidance_signals": float(row[7] or 0.0),
                "dominant_frame": canonicalize_frame(row[8]),
                "seg_young_urban": float(row[9] or 0.0),
                "seg_family": float(row[10] or 0.0),
                "seg_senior": float(row[11] or 0.0),
                "seg_b2b": float(row[12] or 0.0),
                "member_count": int(row[13] or 0),
                "mean_membership_strength": float(row[14] or 0.0),
            }
        )
    return current_clusters


def _load_active_tracks(
    topic: str,
    country: str,
    source: str,
    language: str | None,
) -> list[dict[str, object]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            track_id,
            baseline_topic_label,
            baseline_centroid_vector,
            baseline_centroid_dim,
            concern_level,
            purchase_intent,
            avoidance_signals,
            dominant_frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            sample_count,
            is_learned,
            missed_runs,
            last_member_count,
            last_mean_membership_strength,
            first_seen_run_id,
            last_seen_run_id,
            first_seen_at,
            last_seen_at,
            updated_at
        FROM cluster_tracks
        WHERE topic = ?
          AND country = ?
          AND source = ?
          AND ((? IS NULL AND language IS NULL) OR language = ?)
          AND status = ?
        ORDER BY last_seen_at DESC, track_id ASC
        """,
        (
            topic,
            country,
            source,
            language,
            language,
            ACTIVE_TRACK_STATUS,
        ),
    ).fetchall()

    tracks: list[dict[str, object]] = []
    for row in rows:
        tracks.append(
            {
                "track_id": row[0],
                "baseline_topic_label": row[1] or "",
                "baseline_centroid_vector": _decode_vector(row[2]),
                "baseline_centroid_dim": int(row[3]),
                "concern_level": float(row[4] or 0.0),
                "purchase_intent": float(row[5] or 0.0),
                "avoidance_signals": float(row[6] or 0.0),
                "dominant_frame": canonicalize_frame(row[7]),
                "seg_young_urban": float(row[8] or 0.0),
                "seg_family": float(row[9] or 0.0),
                "seg_senior": float(row[10] or 0.0),
                "seg_b2b": float(row[11] or 0.0),
                "sample_count": int(row[12] or 0),
                "is_learned": bool(row[13]),
                "missed_runs": int(row[14] or 0),
                "last_member_count": int(row[15] or 0),
                "last_mean_membership_strength": float(row[16] or 0.0),
                "first_seen_run_id": row[17],
                "last_seen_run_id": row[18],
                "first_seen_at": row[19],
                "last_seen_at": row[20],
                "updated_at": row[21],
            }
        )
    return tracks


def _assignment_metrics(
    cluster: dict[str, object],
    track: dict[str, object],
    signal_weights: dict[str, float],
) -> dict[str, float] | None:
    centroid_distance = _cosine_distance(
        list(cluster["centroid_vector"]),
        list(track["baseline_centroid_vector"]),
    )
    if centroid_distance > MATCH_MAX_CENTROID_DISTANCE:
        return None

    current_signal_vector = _weighted_signal_vector(_signal_payload(cluster), signal_weights)
    baseline_signal_vector = _weighted_signal_vector(_signal_payload(track), signal_weights)
    segment_vector_distance = _cosine_distance(
        current_signal_vector,
        baseline_signal_vector,
    )
    label_penalty = _label_penalty(
        str(cluster.get("topic_label", "")),
        str(track.get("baseline_topic_label", "")),
    )
    assignment_cost = round(
        MATCH_WEIGHT_CENTROID * centroid_distance
        + MATCH_WEIGHT_SEGMENT * segment_vector_distance
        + MATCH_WEIGHT_LABEL * label_penalty,
        4,
    )
    return {
        "centroid_distance": centroid_distance,
        "segment_vector_distance": segment_vector_distance,
        "label_penalty": label_penalty,
        "assignment_cost": assignment_cost,
    }


def _match_clusters_to_tracks(
    current_clusters: list[dict[str, object]],
    tracks: list[dict[str, object]],
    signal_weights: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cluster_count = len(current_clusters)
    track_count = len(tracks)
    if cluster_count == 0 and track_count == 0:
        return [], [], []

    size = max(cluster_count, track_count)
    cost_matrix = [[0.0] * size for _ in range(size)]
    metrics_by_pair: dict[tuple[int, int], dict[str, float]] = {}

    for row_index in range(size):
        for column_index in range(size):
            is_real_cluster = row_index < cluster_count
            is_real_track = column_index < track_count
            if is_real_cluster and is_real_track:
                metrics = _assignment_metrics(
                    current_clusters[row_index],
                    tracks[column_index],
                    signal_weights,
                )
                if metrics is None:
                    cost_matrix[row_index][column_index] = DISALLOWED_COST
                else:
                    cost_matrix[row_index][column_index] = metrics["assignment_cost"]
                    metrics_by_pair[(row_index, column_index)] = metrics
            elif is_real_cluster or is_real_track:
                cost_matrix[row_index][column_index] = UNMATCHED_COST
            else:
                cost_matrix[row_index][column_index] = 0.0

    assignments = _linear_sum_assignment(cost_matrix)
    matched: list[dict[str, object]] = []
    matched_cluster_indexes: set[int] = set()
    matched_track_indexes: set[int] = set()

    for row_index, column_index in assignments:
        if row_index >= cluster_count or column_index >= track_count:
            continue
        metrics = metrics_by_pair.get((row_index, column_index))
        if metrics is None:
            continue
        if cost_matrix[row_index][column_index] >= UNMATCHED_COST:
            continue
        matched_cluster_indexes.add(row_index)
        matched_track_indexes.add(column_index)
        matched.append(
            {
                "cluster": current_clusters[row_index],
                "track": tracks[column_index],
                **metrics,
            }
        )

    new_clusters = [
        current_clusters[index]
        for index in range(cluster_count)
        if index not in matched_cluster_indexes
    ]
    missing_tracks = [
        tracks[index]
        for index in range(track_count)
        if index not in matched_track_indexes
    ]
    return matched, new_clusters, missing_tracks


def _insert_track(
    conn,
    *,
    run_meta: dict[str, object],
    cluster: dict[str, object],
    now: str,
) -> str:
    track_id = uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO cluster_tracks
        (track_id, topic, country, source, language, status, baseline_topic_label,
         baseline_centroid_vector, baseline_centroid_dim, concern_level,
         purchase_intent, avoidance_signals, dominant_frame, seg_young_urban,
         seg_family, seg_senior, seg_b2b, sample_count, is_learned, missed_runs,
         last_member_count, last_mean_membership_strength, first_seen_run_id,
         last_seen_run_id, first_seen_at, last_seen_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            run_meta["topic"],
            run_meta["country"],
            run_meta["source"],
            run_meta["language"],
            ACTIVE_TRACK_STATUS,
            cluster["topic_label"],
            json.dumps(_normalize_vector(list(cluster["centroid_vector"]))),
            len(list(cluster["centroid_vector"])),
            cluster["concern_level"],
            cluster["purchase_intent"],
            cluster["avoidance_signals"],
            canonicalize_frame(str(cluster["dominant_frame"])),
            cluster["seg_young_urban"],
            cluster["seg_family"],
            cluster["seg_senior"],
            cluster["seg_b2b"],
            int(cluster["member_count"] or 0),
            int(cluster["member_count"] or 0),
            float(cluster["mean_membership_strength"] or 0.0),
            run_meta["run_id"],
            run_meta["run_id"],
            now,
            now,
            now,
        ),
    )
    return track_id


def _update_matched_track(
    conn,
    *,
    track: dict[str, object],
    cluster: dict[str, object],
    run_id: str,
    now: str,
) -> tuple[int, bool]:
    blended_centroid = _normalize_vector(
        [
            BASELINE_EMA_ALPHA * float(old_value)
            + (1 - BASELINE_EMA_ALPHA) * float(new_value)
            for old_value, new_value in zip(
                list(track["baseline_centroid_vector"]),
                list(cluster["centroid_vector"]),
            )
        ]
    )
    blended_segments = {
        segment: round(
            BASELINE_EMA_ALPHA * float(track.get(segment, 0.0) or 0.0)
            + (1 - BASELINE_EMA_ALPHA) * float(cluster.get(f"seg_{segment}", 0.0) or 0.0),
            4,
        )
        for segment in SEGMENTS
    }
    next_sample_count = int(track["sample_count"]) + int(cluster["member_count"] or 0)
    is_learned = bool(track["is_learned"]) or next_sample_count >= MIN_ARTICLES_FOR_BASELINE

    conn.execute(
        """
        UPDATE cluster_tracks
        SET status = ?,
            baseline_topic_label = ?,
            baseline_centroid_vector = ?,
            baseline_centroid_dim = ?,
            concern_level = ?,
            purchase_intent = ?,
            avoidance_signals = ?,
            dominant_frame = ?,
            seg_young_urban = ?,
            seg_family = ?,
            seg_senior = ?,
            seg_b2b = ?,
            sample_count = ?,
            is_learned = ?,
            missed_runs = 0,
            last_member_count = ?,
            last_mean_membership_strength = ?,
            last_seen_run_id = ?,
            last_seen_at = ?,
            updated_at = ?
        WHERE track_id = ?
        """,
        (
            ACTIVE_TRACK_STATUS,
            cluster["topic_label"],
            json.dumps(blended_centroid),
            len(blended_centroid),
            round(
                BASELINE_EMA_ALPHA * float(track["concern_level"])
                + (1 - BASELINE_EMA_ALPHA) * float(cluster["concern_level"]),
                4,
            ),
            round(
                BASELINE_EMA_ALPHA * float(track["purchase_intent"])
                + (1 - BASELINE_EMA_ALPHA) * float(cluster["purchase_intent"]),
                4,
            ),
            round(
                BASELINE_EMA_ALPHA * float(track["avoidance_signals"])
                + (1 - BASELINE_EMA_ALPHA) * float(cluster["avoidance_signals"]),
                4,
            ),
            canonicalize_frame(str(cluster["dominant_frame"])),
            blended_segments["young_urban"],
            blended_segments["family"],
            blended_segments["senior"],
            blended_segments["b2b"],
            next_sample_count,
            int(is_learned),
            int(cluster["member_count"] or 0),
            float(cluster["mean_membership_strength"] or 0.0),
            run_id,
            now,
            now,
            track["track_id"],
        ),
    )
    return next_sample_count, bool(is_learned)


def _update_missing_track(conn, *, track: dict[str, object], now: str) -> None:
    missed_runs = int(track["missed_runs"] or 0) + 1
    status = ACTIVE_TRACK_STATUS if missed_runs < MAX_MISSED_RUNS else INACTIVE_TRACK_STATUS
    conn.execute(
        """
        UPDATE cluster_tracks
        SET status = ?,
            missed_runs = ?,
            updated_at = ?
        WHERE track_id = ?
        """,
        (
            status,
            missed_runs,
            now,
            track["track_id"],
        ),
    )


def _build_matched_observation(
    *,
    metrics: dict[str, object],
    signal_weights: dict[str, float],
) -> dict[str, object]:
    cluster = dict(metrics["cluster"])
    track = dict(metrics["track"])
    baseline_signal = _signal_payload(track)
    current_signal = _signal_payload(cluster)
    deltas = {
        key: round(current_signal[key] - baseline_signal[key], 4)
        for key in SIGNAL_KEYS
    }
    weighted_deltas = {
        key: float(signal_weights.get(key, 0.0) or 0.0) * deltas[key]
        for key in SIGNAL_KEYS
    }
    signal_drift = round(
        sum(abs(float(signal_weights.get(key, 0.0) or 0.0) * deltas[key]) for key in SIGNAL_KEYS),
        4,
    )
    frame_shift = canonicalize_frame(str(cluster["dominant_frame"])) != canonicalize_frame(
        str(track["dominant_frame"])
    )
    drift_magnitude = round(
        0.7 * signal_drift + 0.3 * float(metrics["centroid_distance"]),
        4,
    )
    confidence = _compute_observation_confidence(
        int(cluster["member_count"] or 0),
        float(cluster["mean_membership_strength"] or 0.0),
        int(track["sample_count"] or 0),
        bool(track["is_learned"]),
    )

    return {
        "track_id": track["track_id"],
        "cluster_id": cluster["cluster_id"],
        "cluster_label": cluster["cluster_label"],
        "topic_label": cluster["topic_label"],
        "baseline_topic_label": track["baseline_topic_label"],
        "match_type": MATCHED,
        "direction": _direction_from_deltas(
            weighted_deltas,
            frame_shift=frame_shift,
            match_type=MATCHED,
        ),
        "centroid_distance": float(metrics["centroid_distance"]),
        "segment_vector_distance": float(metrics["segment_vector_distance"]),
        "signal_drift": signal_drift,
        "drift_magnitude": drift_magnitude,
        "alert_level": _alert_level(drift_magnitude, has_data=True),
        "confidence": confidence,
        "member_count": int(cluster["member_count"] or 0),
        "baseline_member_count": int(track["last_member_count"] or 0),
        "mean_membership_strength": float(cluster["mean_membership_strength"] or 0.0),
        "current": current_signal,
        "baseline": baseline_signal,
        "deltas": deltas,
        "dominant_frame": canonicalize_frame(str(cluster["dominant_frame"])),
        "baseline_frame": canonicalize_frame(str(track["dominant_frame"])),
        "segments": _segment_payload(cluster, "seg_"),
        "baseline_segments": _segment_payload(track),
        "frame_shift": frame_shift,
        "baseline_is_learned": bool(track["is_learned"]),
        "baseline_sample_count": int(track["sample_count"] or 0),
        "baseline_age_days": _baseline_age_days(str(track["updated_at"] or "")),
    }


def _build_new_observation(
    *,
    track_id: str,
    cluster: dict[str, object],
) -> dict[str, object]:
    current_signal = _signal_payload(cluster)
    confidence = _compute_observation_confidence(
        int(cluster["member_count"] or 0),
        float(cluster["mean_membership_strength"] or 0.0),
        int(cluster["member_count"] or 0),
        False,
    )
    return {
        "track_id": track_id,
        "cluster_id": cluster["cluster_id"],
        "cluster_label": cluster["cluster_label"],
        "topic_label": cluster["topic_label"],
        "baseline_topic_label": cluster["topic_label"],
        "match_type": NEW,
        "direction": "emerging",
        "centroid_distance": 0.0,
        "segment_vector_distance": 0.0,
        "signal_drift": 0.0,
        "drift_magnitude": 0.0,
        "alert_level": "none",
        "confidence": confidence,
        "member_count": int(cluster["member_count"] or 0),
        "baseline_member_count": 0,
        "mean_membership_strength": float(cluster["mean_membership_strength"] or 0.0),
        "current": current_signal,
        "baseline": None,
        "deltas": {key: 0.0 for key in SIGNAL_KEYS},
        "dominant_frame": canonicalize_frame(str(cluster["dominant_frame"])),
        "baseline_frame": None,
        "segments": _segment_payload(cluster, "seg_"),
        "baseline_segments": {segment: 0.0 for segment in SEGMENTS},
        "frame_shift": False,
        "baseline_is_learned": False,
        "baseline_sample_count": int(cluster["member_count"] or 0),
        "baseline_age_days": 0,
    }


def _build_missing_observation(track: dict[str, object]) -> dict[str, object]:
    baseline_signal = _signal_payload(track)
    confidence = _compute_observation_confidence(
        int(track["last_member_count"] or 0),
        float(track["last_mean_membership_strength"] or 0.0),
        int(track["sample_count"] or 0),
        bool(track["is_learned"]),
    )
    return {
        "track_id": track["track_id"],
        "cluster_id": None,
        "cluster_label": None,
        "topic_label": track["baseline_topic_label"],
        "baseline_topic_label": track["baseline_topic_label"],
        "match_type": MISSING,
        "direction": "fading",
        "centroid_distance": 0.0,
        "segment_vector_distance": 0.0,
        "signal_drift": 0.0,
        "drift_magnitude": 0.0,
        "alert_level": "none",
        "confidence": confidence,
        "member_count": 0,
        "baseline_member_count": int(track["last_member_count"] or 0),
        "mean_membership_strength": float(track["last_mean_membership_strength"] or 0.0),
        "current": {key: 0.0 for key in SIGNAL_KEYS},
        "baseline": baseline_signal,
        "deltas": {key: 0.0 for key in SIGNAL_KEYS},
        "dominant_frame": "neutral",
        "baseline_frame": canonicalize_frame(str(track["dominant_frame"])),
        "segments": {segment: 0.0 for segment in SEGMENTS},
        "baseline_segments": _segment_payload(track),
        "frame_shift": False,
        "baseline_is_learned": bool(track["is_learned"]),
        "baseline_sample_count": int(track["sample_count"] or 0),
        "baseline_age_days": _baseline_age_days(str(track["updated_at"] or "")),
    }


def _persist_observations(conn, *, run_id: str, observations: list[dict[str, object]], now: str) -> None:
    for item in observations:
        baseline = item["baseline"]
        conn.execute(
            """
            INSERT INTO cluster_drift_observations
            (run_id, track_id, cluster_id, cluster_label, topic_label, baseline_topic_label,
             match_type, direction, centroid_distance, segment_vector_distance,
             signal_drift, drift_magnitude, alert_level, confidence, member_count,
             mean_membership_strength, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, baseline_concern_level, baseline_purchase_intent,
             baseline_avoidance_signals, baseline_dominant_frame, delta_concern_level,
             delta_purchase_intent, delta_avoidance_signals, seg_young_urban, seg_family,
             seg_senior, seg_b2b, baseline_seg_young_urban, baseline_seg_family,
             baseline_seg_senior, baseline_seg_b2b, frame_shift, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item["track_id"],
                item["cluster_id"],
                item["cluster_label"],
                item["topic_label"],
                item["baseline_topic_label"],
                item["match_type"],
                item["direction"],
                item["centroid_distance"],
                item["segment_vector_distance"],
                item["signal_drift"],
                item["drift_magnitude"],
                item["alert_level"],
                item["confidence"],
                item["member_count"],
                item["mean_membership_strength"],
                item["current"]["concern_level"],
                item["current"]["purchase_intent"],
                item["current"]["avoidance_signals"],
                item["dominant_frame"],
                baseline["concern_level"] if baseline is not None else None,
                baseline["purchase_intent"] if baseline is not None else None,
                baseline["avoidance_signals"] if baseline is not None else None,
                item["baseline_frame"],
                item["deltas"]["concern_level"],
                item["deltas"]["purchase_intent"],
                item["deltas"]["avoidance_signals"],
                item["segments"]["young_urban"],
                item["segments"]["family"],
                item["segments"]["senior"],
                item["segments"]["b2b"],
                item["baseline_segments"]["young_urban"],
                item["baseline_segments"]["family"],
                item["baseline_segments"]["senior"],
                item["baseline_segments"]["b2b"],
                int(item["frame_shift"]),
                now,
            ),
        )


def _weighted_average(values: list[tuple[float, float]]) -> float:
    numerator = sum(value * weight for value, weight in values if weight > 0)
    denominator = sum(weight for _, weight in values if weight > 0)
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _dominant_frame(frame_weights: dict[str, float]) -> str:
    if not frame_weights:
        return "neutral"
    return max(frame_weights, key=frame_weights.get)


def _aggregate_segments(
    observations: list[dict[str, object]],
    *,
    topic: str,
) -> list[dict[str, object]]:
    domain = topic_to_domain(topic)
    domain_config = get_domain_config(domain)
    relevant_fields = list(domain_config["relevant_fields"])
    signal_weights = domain_config["signal_weights"]
    segments: list[dict[str, object]] = []

    for segment in SEGMENTS:
        current_signal_values = {key: [] for key in SIGNAL_KEYS}
        baseline_signal_values = {key: [] for key in SIGNAL_KEYS}
        current_frames: dict[str, float] = {}
        baseline_frames: dict[str, float] = {}
        matched_drift_pairs: list[tuple[float, float]] = []
        matched_centroid_pairs: list[tuple[float, float]] = []
        confidence_pairs: list[tuple[float, float]] = []
        article_count = 0
        tracked_cluster_count = 0
        matched_cluster_count = 0
        new_cluster_count = 0
        missing_cluster_count = 0
        learned_track_present = False
        baseline_sample_count = 0
        baseline_age_days_values: list[int] = []

        current_total_weight = 0.0
        new_total_weight = 0.0
        missing_total_weight = 0.0

        for observation in observations:
            current_weight = (
                float(observation["member_count"] or 0)
                * float(observation["segments"][segment] or 0.0)
            )
            baseline_reference_count = int(
                observation.get("baseline_member_count", observation["member_count"]) or 0
            )
            baseline_weight = (
                baseline_reference_count
                * float(observation["baseline_segments"][segment] or 0.0)
            )

            if observation["match_type"] in {MATCHED, NEW} and current_weight > 0:
                article_count += int(observation["member_count"] or 0)
                current_total_weight += current_weight
                for key in SIGNAL_KEYS:
                    current_signal_values[key].append((observation["current"][key], current_weight))
                current_frame = canonicalize_frame(str(observation["dominant_frame"]))
                current_frames[current_frame] = current_frames.get(current_frame, 0.0) + current_weight
                confidence_pairs.append((float(observation["confidence"]), current_weight))
                if observation["match_type"] == NEW:
                    new_total_weight += current_weight
                    new_cluster_count += 1

            if observation["match_type"] in {MATCHED, MISSING} and baseline_weight > 0:
                for key in SIGNAL_KEYS:
                    baseline_signal = (
                        observation["baseline"][key]
                        if observation["baseline"] is not None
                        else 0.0
                    )
                    baseline_signal_values[key].append((baseline_signal, baseline_weight))
                baseline_frame = canonicalize_frame(str(observation["baseline_frame"]))
                baseline_frames[baseline_frame] = baseline_frames.get(baseline_frame, 0.0) + baseline_weight
                if observation["match_type"] == MISSING:
                    missing_total_weight += baseline_weight
                    missing_cluster_count += 1

            if observation["match_type"] == MATCHED:
                tracked_cluster_count += 1
                matched_cluster_count += 1
                drift_weight = current_weight if current_weight > 0 else baseline_weight
                matched_drift_pairs.append((float(observation["drift_magnitude"]), drift_weight))
                matched_centroid_pairs.append((float(observation["centroid_distance"]), drift_weight))
            elif observation["match_type"] == MISSING and baseline_weight > 0:
                tracked_cluster_count += 1
                confidence_pairs.append((float(observation["confidence"]), baseline_weight))

            if observation["baseline_is_learned"]:
                learned_track_present = True
            baseline_sample_count = max(
                baseline_sample_count,
                int(observation.get("baseline_sample_count", 0) or 0),
            )
            baseline_age = observation.get("baseline_age_days")
            if baseline_age is not None:
                baseline_age_days_values.append(int(baseline_age))

        current = {
            key: _weighted_average(current_signal_values[key])
            for key in SIGNAL_KEYS
        }
        baseline_available = any(values for values in baseline_signal_values.values())
        baseline = (
            {
                key: _weighted_average(baseline_signal_values[key])
                for key in SIGNAL_KEYS
            }
            if baseline_available
            else None
        )
        deltas = {
            key: round(current[key] - ((baseline or {}).get(key, 0.0)), 4)
            for key in SIGNAL_KEYS
        }
        drift_magnitude = _weighted_average(matched_drift_pairs)
        centroid_shift = _weighted_average(matched_centroid_pairs)
        dominant_frame = _dominant_frame(current_frames)
        baseline_frame = _dominant_frame(baseline_frames) if baseline_available else None
        frame_shift = (
            baseline_frame is not None
            and canonicalize_frame(dominant_frame) != canonicalize_frame(baseline_frame)
        )
        confidence = _weighted_average(confidence_pairs)
        new_cluster_weight = round(
            new_total_weight / current_total_weight, 4,
        ) if current_total_weight > 0 else 0.0
        missing_cluster_weight = round(
            missing_total_weight / (current_total_weight + missing_total_weight), 4,
        ) if (current_total_weight + missing_total_weight) > 0 else 0.0
        has_data = current_total_weight > 0 or missing_total_weight > 0

        if new_cluster_weight >= 0.5:
            direction = "emerging"
        elif missing_cluster_weight >= 0.5:
            direction = "fading"
        else:
            weighted_direction = {
                key: float(signal_weights.get(key, 0.0) or 0.0) * deltas[key]
                for key in SIGNAL_KEYS
            }
            direction = _direction_from_deltas(
                weighted_direction,
                frame_shift=frame_shift,
                match_type=MATCHED,
            )

        segments.append(
            {
                "segment": segment,
                "topic": topic,
                "article_count": article_count,
                "has_data": has_data,
                "current": current,
                "baseline": baseline,
                "deltas": deltas,
                "drift_magnitude": drift_magnitude,
                "frame_shift": frame_shift,
                "alert_level": _alert_level(drift_magnitude, has_data),
                "dominant_frame": dominant_frame,
                "baseline_frame": baseline_frame,
                "confidence": confidence,
                "baseline_is_learned": learned_track_present,
                "baseline_sample_count": baseline_sample_count,
                "baseline_age_days": min(baseline_age_days_values) if baseline_age_days_values else None,
                "status": _segment_status(has_data, learned_track_present, confidence),
                "domain": domain,
                "relevant_fields": relevant_fields,
                "direction": direction,
                "centroid_shift": centroid_shift,
                "new_cluster_weight": new_cluster_weight,
                "tracked_cluster_count": tracked_cluster_count,
                "matched_cluster_count": matched_cluster_count,
                "new_cluster_count": new_cluster_count,
                "missing_cluster_count": missing_cluster_count,
            }
        )

    return segments


def _persist_segment_drifts(conn, *, run_id: str, segments: list[dict[str, object]], now: str) -> None:
    for item in segments:
        baseline = item["baseline"]
        conn.execute(
            """
            INSERT INTO cluster_segment_drifts
            (run_id, segment, article_count, has_data, concern_level, purchase_intent,
             avoidance_signals, dominant_frame, baseline_concern_level,
             baseline_purchase_intent, baseline_avoidance_signals, baseline_dominant_frame,
             delta_concern_level, delta_purchase_intent, delta_avoidance_signals,
             drift_magnitude, frame_shift, alert_level, confidence,
             baseline_is_learned, baseline_sample_count, baseline_age_days, status,
             direction, centroid_shift, new_cluster_weight, tracked_cluster_count,
             matched_cluster_count, new_cluster_count, missing_cluster_count, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item["segment"],
                item["article_count"],
                int(item["has_data"]),
                item["current"]["concern_level"],
                item["current"]["purchase_intent"],
                item["current"]["avoidance_signals"],
                item["dominant_frame"],
                baseline["concern_level"] if baseline is not None else None,
                baseline["purchase_intent"] if baseline is not None else None,
                baseline["avoidance_signals"] if baseline is not None else None,
                item["baseline_frame"],
                item["deltas"]["concern_level"],
                item["deltas"]["purchase_intent"],
                item["deltas"]["avoidance_signals"],
                item["drift_magnitude"],
                int(item["frame_shift"]),
                item["alert_level"],
                item["confidence"],
                int(item["baseline_is_learned"]),
                item["baseline_sample_count"],
                item["baseline_age_days"],
                item["status"],
                item["direction"],
                item["centroid_shift"],
                item["new_cluster_weight"],
                item["tracked_cluster_count"],
                item["matched_cluster_count"],
                item["new_cluster_count"],
                item["missing_cluster_count"],
                now,
            ),
        )


def run_cluster_drift(run_id: str) -> dict[str, object]:
    started_at = time.perf_counter()
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id is required.")

    run_meta = _assert_stage_four_ready(normalized_run_id)
    if _cluster_drift_exists(normalized_run_id):
        return get_cluster_drift_stage_result(normalized_run_id)

    topic = str(run_meta["topic"])
    domain = topic_to_domain(topic)
    domain_config = get_domain_config(domain)
    signal_weights = domain_config["signal_weights"]
    current_clusters = _load_current_clusters(normalized_run_id)
    active_tracks = _load_active_tracks(
        topic,
        str(run_meta["country"]),
        str(run_meta["source"]),
        run_meta["language"],
    )
    matched, new_clusters, missing_tracks = _match_clusters_to_tracks(
        current_clusters,
        active_tracks,
        signal_weights,
    )
    now = _utc_now_iso()
    conn = get_conn()
    observations: list[dict[str, object]] = []

    conn.execute("BEGIN")
    try:
        for match in matched:
            observations.append(
                _build_matched_observation(
                    metrics=match,
                    signal_weights=signal_weights,
                )
            )

        for cluster in new_clusters:
            track_id = _insert_track(
                conn,
                run_meta=run_meta,
                cluster=cluster,
                now=now,
            )
            observations.append(
                _build_new_observation(
                    track_id=track_id,
                    cluster=cluster,
                )
            )

        for track in missing_tracks:
            observations.append(_build_missing_observation(track))

        _persist_observations(
            conn,
            run_id=normalized_run_id,
            observations=observations,
            now=now,
        )

        segments = _aggregate_segments(observations, topic=topic)
        _persist_segment_drifts(
            conn,
            run_id=normalized_run_id,
            segments=segments,
            now=now,
        )

        for match in matched:
            _update_matched_track(
                conn,
                track=match["track"],
                cluster=match["cluster"],
                run_id=normalized_run_id,
                now=now,
            )

        for track in missing_tracks:
            _update_missing_track(conn, track=track, now=now)

        duration_s = round(time.perf_counter() - started_at, 4)
        conn.execute(
            """
            INSERT INTO cluster_drift_runs
            (run_id, topic, country, source, language, observed_cluster_count,
             matched_track_count, new_track_count, missing_track_count,
             segment_count, computed_at, duration_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_run_id,
                topic,
                run_meta["country"],
                run_meta["source"],
                run_meta["language"],
                len(current_clusters),
                len(matched),
                len(new_clusters),
                len(missing_tracks),
                len(segments),
                now,
                duration_s,
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return get_cluster_drift_stage_result(normalized_run_id)


def get_cluster_drift_stage_result(run_id: str) -> dict[str, object]:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT run_id, observed_cluster_count, matched_track_count, new_track_count,
               missing_track_count, segment_count, computed_at, duration_s
        FROM cluster_drift_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"Cluster drift run '{run_id}' was not found.")
    return {
        "run_id": row[0],
        "observed_clusters": int(row[1]),
        "matched_tracks": int(row[2]),
        "new_tracks": int(row[3]),
        "missing_tracks": int(row[4]),
        "segments": int(row[5]),
        "computed_at": row[6],
        "duration_s": float(row[7]),
    }


def get_cluster_drift(run_id: str) -> dict[str, object]:
    conn = get_conn()
    run_row = conn.execute(
        """
        SELECT cr.run_id, cr.topic, cr.country, cr.source, cr.language, cdr.computed_at
        FROM cluster_runs cr
        JOIN cluster_drift_runs cdr ON cdr.run_id = cr.run_id
        WHERE cr.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if run_row is None:
        raise LookupError(f"Cluster drift run '{run_id}' was not found.")

    segment_rows = conn.execute(
        """
        SELECT
            segment, article_count, has_data, concern_level, purchase_intent,
            avoidance_signals, dominant_frame, baseline_concern_level,
            baseline_purchase_intent, baseline_avoidance_signals, baseline_dominant_frame,
            delta_concern_level, delta_purchase_intent, delta_avoidance_signals,
            drift_magnitude, frame_shift, alert_level, confidence,
            baseline_is_learned, baseline_sample_count, baseline_age_days,
            status, direction, centroid_shift, new_cluster_weight,
            tracked_cluster_count, matched_cluster_count, new_cluster_count,
            missing_cluster_count
        FROM cluster_segment_drifts
        WHERE run_id = ?
        ORDER BY segment ASC
        """,
        (run_id,),
    ).fetchall()
    observation_rows = conn.execute(
        """
        SELECT
            track_id, cluster_id, cluster_label, topic_label, baseline_topic_label,
            match_type, direction, centroid_distance, segment_vector_distance,
            signal_drift, drift_magnitude, alert_level, confidence, member_count,
            mean_membership_strength, concern_level, purchase_intent,
            avoidance_signals, dominant_frame, baseline_concern_level,
            baseline_purchase_intent, baseline_avoidance_signals, baseline_dominant_frame,
            delta_concern_level, delta_purchase_intent, delta_avoidance_signals,
            frame_shift
        FROM cluster_drift_observations
        WHERE run_id = ?
        ORDER BY match_type ASC, drift_magnitude DESC, topic_label ASC, track_id ASC
        """,
        (run_id,),
    ).fetchall()

    domain = topic_to_domain(run_row[1])
    relevant_fields = list(get_domain_config(domain)["relevant_fields"])

    segments = [
        {
            "segment": row[0],
            "topic": run_row[1],
            "article_count": int(row[1]),
            "has_data": bool(row[2]),
            "current": {
                "concern_level": float(row[3] or 0.0),
                "purchase_intent": float(row[4] or 0.0),
                "avoidance_signals": float(row[5] or 0.0),
            },
            "baseline": (
                {
                    "concern_level": float(row[7] or 0.0),
                    "purchase_intent": float(row[8] or 0.0),
                    "avoidance_signals": float(row[9] or 0.0),
                }
                if row[7] is not None or row[8] is not None or row[9] is not None
                else None
            ),
            "deltas": {
                "concern_level": float(row[11] or 0.0),
                "purchase_intent": float(row[12] or 0.0),
                "avoidance_signals": float(row[13] or 0.0),
            },
            "drift_magnitude": float(row[14] or 0.0),
            "frame_shift": bool(row[15]),
            "alert_level": row[16],
            "dominant_frame": row[6],
            "baseline_frame": row[10],
            "confidence": float(row[17] or 0.0),
            "baseline_is_learned": bool(row[18]),
            "baseline_sample_count": int(row[19] or 0),
            "baseline_age_days": row[20],
            "status": row[21],
            "domain": domain,
            "relevant_fields": relevant_fields,
            "direction": row[22],
            "centroid_shift": float(row[23] or 0.0),
            "new_cluster_weight": float(row[24] or 0.0),
            "tracked_cluster_count": int(row[25] or 0),
            "matched_cluster_count": int(row[26] or 0),
            "new_cluster_count": int(row[27] or 0),
            "missing_cluster_count": int(row[28] or 0),
        }
        for row in segment_rows
    ]
    observations = [
        {
            "track_id": row[0],
            "cluster_id": row[1],
            "cluster_label": row[2],
            "topic_label": row[3],
            "baseline_topic_label": row[4],
            "match_type": row[5],
            "direction": row[6],
            "centroid_distance": float(row[7] or 0.0),
            "segment_vector_distance": float(row[8] or 0.0),
            "signal_drift": float(row[9] or 0.0),
            "drift_magnitude": float(row[10] or 0.0),
            "alert_level": row[11],
            "confidence": float(row[12] or 0.0),
            "member_count": int(row[13] or 0),
            "mean_membership_strength": float(row[14] or 0.0),
            "current": {
                "concern_level": float(row[15] or 0.0),
                "purchase_intent": float(row[16] or 0.0),
                "avoidance_signals": float(row[17] or 0.0),
            },
            "baseline": (
                {
                    "concern_level": float(row[19] or 0.0),
                    "purchase_intent": float(row[20] or 0.0),
                    "avoidance_signals": float(row[21] or 0.0),
                }
                if row[19] is not None or row[20] is not None or row[21] is not None
                else None
            ),
            "deltas": {
                "concern_level": float(row[23] or 0.0),
                "purchase_intent": float(row[24] or 0.0),
                "avoidance_signals": float(row[25] or 0.0),
            },
            "dominant_frame": row[18],
            "baseline_frame": row[22],
            "frame_shift": bool(row[26]),
        }
        for row in observation_rows
    ]
    return {
        "topic": run_row[1],
        "country": run_row[2],
        "source": run_row[3],
        "language": run_row[4],
        "run_id": run_row[0],
        "computed_at": run_row[5],
        "segments": segments,
        "clusters": observations,
    }


def get_latest_cluster_drift(
    *,
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object] | None:
    conn = get_conn()
    normalized_country = (country or "").strip().upper()
    normalized_source = (source or "").strip().lower()
    normalized_language = ((language or "").strip().lower() or None)
    query = """
        SELECT run_id
        FROM cluster_drift_runs
        WHERE topic = ?
          AND country = ?
          AND source = ?
        ORDER BY computed_at DESC, run_id DESC
        LIMIT 1
    """
    params: list[object] = [topic, normalized_country, normalized_source]
    if normalized_language is not None:
        query = query.replace(
            "ORDER BY computed_at DESC, run_id DESC",
            "AND language = ?\n        ORDER BY computed_at DESC, run_id DESC",
        )
        params.append(normalized_language)
    row = conn.execute(query, params).fetchone()
    if row is None:
        return None
    return get_cluster_drift(row[0])
