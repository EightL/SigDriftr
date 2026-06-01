from __future__ import annotations

import json
import time
import uuid

from config.domains import get_domain_config
from config.settings import (
    BASELINE_EMA_ALPHA,
    MIN_ARTICLES_FOR_BASELINE,
)
from config.topics import domain_for_topic
from db.init import get_conn
from db.topic_resolver import resolve_topic
from delta.cluster_matching import (
    ACTIVE_TRACK_STATUS,
    ALERT_THRESHOLDS,
    DISALLOWED_COST,
    INACTIVE_TRACK_STATUS,
    MATCHED,
    MATCH_MAX_CENTROID_DISTANCE,
    MATCH_WEIGHT_CENTROID,
    MATCH_WEIGHT_LABEL,
    MATCH_WEIGHT_SEGMENT,
    MAX_MISSED_RUNS,
    MISSING,
    NEW,
    UNMATCHED_COST,
    _alert_level,
    _assignment_metrics,
    _baseline_age_days,
    _clamp,
    _compute_observation_confidence,
    _cosine_distance,
    _decode_vector,
    _direction_from_deltas,
    _label_penalty,
    _linear_sum_assignment,
    _match_clusters_to_tracks,
    _normalize_label,
    _normalize_vector,
    _segment_payload,
    _segment_status,
    _signal_payload,
    _utc_now_iso,
    _weighted_signal_vector,
)
from delta.cluster_store import (
    REQUIRED_STAGE_FOUR_TABLES,
    _assert_stage_four_ready,
    _cluster_drift_exists,
    _load_active_tracks,
    _load_current_clusters,
)
from delta.mapper import SEGMENTS, SIGNAL_KEYS, canonicalize_frame



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
        (track_id, topic, canonical_topic_id, country, source, language, status, baseline_topic_label,
         baseline_centroid_vector, baseline_centroid_dim, concern_level,
         purchase_intent, avoidance_signals, dominant_frame, seg_young_urban,
         seg_family, seg_senior, seg_b2b, sample_count, is_learned, missed_runs,
         last_member_count, last_mean_membership_strength, first_seen_run_id,
         last_seen_run_id, first_seen_at, last_seen_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            run_meta["topic"],
            run_meta.get("canonical_topic_id") or "",
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
    domain = domain_for_topic(topic)
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
    canonical_topic_id = str(run_meta.get("canonical_topic_id") or "")
    domain = domain_for_topic(canonical_topic_id or topic)
    domain_config = get_domain_config(domain)
    signal_weights = domain_config["signal_weights"]
    current_clusters = _load_current_clusters(normalized_run_id)
    active_tracks = _load_active_tracks(
        topic,
        canonical_topic_id,
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

        segments = _aggregate_segments(observations, topic=canonical_topic_id or topic)
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
            (run_id, topic, canonical_topic_id, country, source, language, observed_cluster_count,
             matched_track_count, new_track_count, missing_track_count,
             segment_count, computed_at, duration_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_run_id,
                topic,
                canonical_topic_id,
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
        SELECT cr.run_id, cr.topic, cr.canonical_topic_id, cr.country, cr.source,
               cr.language, cdr.computed_at
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

    domain = domain_for_topic(run_row[2] or run_row[1])
    relevant_fields = list(get_domain_config(domain)["relevant_fields"])

    segments = [
        {
            "segment": row[0],
            "topic": run_row[1],
            "canonical_topic_id": run_row[2],
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
        "canonical_topic_id": run_row[2],
        "country": run_row[3],
        "source": run_row[4],
        "language": run_row[5],
        "run_id": run_row[0],
        "computed_at": run_row[6],
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
    canonical_topic_id = resolve_topic(topic).canonical_topic_id if topic else ""
    normalized_country = (country or "").strip().upper()
    normalized_source = (source or "").strip().lower()
    normalized_language = ((language or "").strip().lower() or None)
    query = """
        SELECT run_id
        FROM cluster_drift_runs
        WHERE (
              topic = ?
              OR (? != '' AND canonical_topic_id = ?)
        )
          AND country = ?
          AND source = ?
        ORDER BY computed_at DESC, run_id DESC
        LIMIT 1
    """
    params: list[object] = [
        topic,
        canonical_topic_id,
        canonical_topic_id,
        normalized_country,
        normalized_source,
    ]
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
