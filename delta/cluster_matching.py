from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from config.settings import (
    CONFIDENCE_ARTICLE_DENOMINATOR,
    CONFIDENCE_BASELINE_WEIGHT,
    CONFIDENCE_COVERAGE_WEIGHT,
    MIN_BRIEF_CONFIDENCE,
)
from delta.mapper import SEGMENTS, SIGNAL_KEYS


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


def _weighted_signal_vector(
    values: dict[str, float],
    signal_weights: dict[str, float],
) -> list[float]:
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
