import hashlib
import math
from datetime import datetime, timedelta, timezone

from config.domains import get_domain_config, topic_to_domain
from config.source_panels import fixed_panel_outlets
from config.settings import (
    BASELINE_EMA_ALPHA,
    CONFIDENCE_ARTICLE_DENOMINATOR,
    CONFIDENCE_BASELINE_WEIGHT,
    CONFIDENCE_COVERAGE_WEIGHT,
    MIN_BRIEF_CONFIDENCE,
    MIN_ARTICLES_FOR_BASELINE,
)
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from delta.mapper import SIGNAL_KEYS, canonicalize_frame, compute_segment_profiles
from delta.seeder import ensure_topic_baselines


ALERT_THRESHOLDS = {
    "mild": 0.20,
    "strong": 0.45,
}
SEGMENT_COLUMNS = {
    "young_urban": "seg_young_urban",
    "family": "seg_family",
    "senior": "seg_senior",
    "b2b": "seg_b2b",
}


def _get_baseline(topic: str, segment: str) -> dict | None:
    conn = get_conn()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
    row = conn.execute(
        """
        SELECT concern_level, purchase_intent, avoidance_signals, dominant_frame,
               sample_count, is_learned, updated_at
        FROM baselines WHERE id = ?
        """,
        (row_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "concern_level": row[0],
        "purchase_intent": row[1],
        "avoidance_signals": row[2],
        "dominant_frame": canonicalize_frame(row[3]),
        "sample_count": row[4] or 0,
        "is_learned": bool(row[5]),
        "updated_at": row[6],
    }


def _alert_level(drift_magnitude: float, article_count: int) -> str:
    if article_count == 0:
        return "no_data"
    if drift_magnitude >= ALERT_THRESHOLDS["strong"]:
        return "strong"
    if drift_magnitude >= ALERT_THRESHOLDS["mild"]:
        return "mild"
    return "none"


def _segment_status(
    article_count: int,
    is_learned: bool,
    confidence: float,
) -> str:
    if article_count == 0:
        return "no_data"
    if is_learned and confidence >= MIN_BRIEF_CONFIDENCE:
        return "ready"
    return "warming"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _baseline_age_days(updated_at: str | None) -> int | None:
    if not updated_at:
        return None
    updated = datetime.fromisoformat(updated_at)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - updated.astimezone(timezone.utc)
    return max(0, delta.days)


def _compute_confidence(
    article_count: int,
    sample_count: int,
    is_learned: bool,
) -> float:
    coverage = min(1.0, article_count / CONFIDENCE_ARTICLE_DENOMINATOR)
    if not is_learned:
        baseline_quality = 0.3
    else:
        baseline_quality = min(1.0, sample_count / CONFIDENCE_ARTICLE_DENOMINATOR) + 0.3
        baseline_quality = min(1.0, baseline_quality)

    confidence = (
        CONFIDENCE_COVERAGE_WEIGHT * coverage
        + CONFIDENCE_BASELINE_WEIGHT * baseline_quality
    )
    return round(_clamp(confidence), 4)


def _outlet_distribution(
    topic: str,
    *,
    start_at: str,
    end_at: str | None = None,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, int]:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT a.outlet, COUNT(DISTINCT a.id)
        FROM articles a
        WHERE COALESCE(a.published_at, a.fetched_at) >= ?
          {topic_sql}
    """
    params: list[object] = [start_at, *topic_params]
    if end_at is not None:
        query += " AND COALESCE(a.published_at, a.fetched_at) < ?"
        params.append(end_at)
    normalized_country = (country or "").strip().upper()
    normalized_source = (source or "").strip().lower()
    normalized_language = ((language or "").strip().lower() or None)
    if normalized_country:
        query += " AND a.country = ?"
        params.append(normalized_country)
    if normalized_source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(normalized_source)
    if normalized_language is not None:
        query += " AND LOWER(a.language) = ?"
        params.append(normalized_language)
    query += " GROUP BY a.outlet ORDER BY COUNT(DISTINCT a.id) DESC, a.outlet ASC"
    return {str(row[0]): int(row[1] or 0) for row in conn.execute(query, params).fetchall()}


def _normalize_distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(max(0, value) for value in counts.values())
    if total <= 0:
        return {}
    return {
        outlet: round(max(0, count) / total, 6)
        for outlet, count in counts.items()
        if count > 0
    }


def _kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    total = 0.0
    for key, value in left.items():
        if value <= 0:
            continue
        other = right.get(key, 0.0)
        if other <= 0:
            continue
        total += value * math.log(value / other, 2)
    return total


def _jensen_shannon_divergence(
    current: dict[str, float],
    reference: dict[str, float],
) -> float | None:
    if not current or not reference:
        return None
    keys = set(current) | set(reference)
    midpoint = {
        key: (current.get(key, 0.0) + reference.get(key, 0.0)) / 2.0
        for key in keys
    }
    jsd = 0.5 * _kl_divergence(current, midpoint) + 0.5 * _kl_divergence(reference, midpoint)
    return round(max(0.0, min(1.0, jsd)), 4)


def compute_source_mix(
    topic: str,
    days_back: int = 7,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=days_back)
    reference_start = current_start - timedelta(days=days_back)
    current_counts = _outlet_distribution(
        topic,
        start_at=current_start.isoformat(),
        country=country,
        source=source,
        language=language,
    )
    reference_counts = _outlet_distribution(
        topic,
        start_at=reference_start.isoformat(),
        end_at=current_start.isoformat(),
        country=country,
        source=source,
        language=language,
    )
    current_distribution = _normalize_distribution(current_counts)
    reference_distribution = _normalize_distribution(reference_counts)
    divergence = _jensen_shannon_divergence(
        current_distribution,
        reference_distribution,
    )
    current_total = sum(current_counts.values())
    reference_total = sum(reference_counts.values())
    if current_total == 0:
        warning = "no_current_articles"
    elif reference_total == 0:
        warning = "no_reference_window"
    elif current_total < 5 or reference_total < 5:
        warning = "low_sample"
    elif divergence is not None and divergence >= 0.25:
        warning = "outlet_mix_shift"
    else:
        warning = "none"

    return {
        "current_window_days": days_back,
        "reference_window_days": days_back,
        "current": {
            "article_count": current_total,
            "article_count_by_outlet": current_counts,
            "distribution": current_distribution,
        },
        "reference": {
            "article_count": reference_total,
            "article_count_by_outlet": reference_counts,
            "distribution": reference_distribution,
        },
        "jensen_shannon_divergence": divergence,
        "warning": warning,
    }


def _source_panel_for_scope(topic: str, country: str = "", source: str = "") -> list[str]:
    normalized_source = (source or "").strip().lower()
    if normalized_source:
        return [normalized_source]
    return fixed_panel_outlets(topic=topic, country=country)


def _current_rows_by_outlet(
    topic: str,
    days_back: int,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> list[tuple]:
    conn = get_conn()
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT a.outlet,
               s.concern_level, s.purchase_intent, s.avoidance_signals,
               s.dominant_frame,
               s.seg_young_urban, s.seg_family, s.seg_senior, s.seg_b2b
        FROM signals s
        JOIN articles a ON a.id = s.article_id
        WHERE COALESCE(a.published_at, a.fetched_at) >= ?
          {topic_sql}
    """
    params: list[object] = [since, *topic_params]
    normalized_country = (country or "").strip().upper()
    normalized_source = (source or "").strip().lower()
    normalized_language = ((language or "").strip().lower() or None)
    if normalized_country:
        query += " AND a.country = ?"
        params.append(normalized_country)
    if normalized_source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(normalized_source)
    if normalized_language is not None:
        query += " AND LOWER(a.language) = ?"
        params.append(normalized_language)
    return conn.execute(query, params).fetchall()


def _per_outlet_segment_profiles(
    rows: list[tuple],
) -> dict[str, dict[str, dict[str, object]]]:
    profiles: dict[str, dict[str, dict[str, object]]] = {}

    for row in rows:
        outlet = str(row[0] or "").strip().lower()
        if not outlet:
            continue
        by_segment = profiles.setdefault(outlet, {})
        for segment_index, segment in enumerate(SEGMENT_COLUMNS):
            weight = float(row[5 + segment_index] or 0.0)
            if weight <= 0.0:
                continue
            profile = by_segment.setdefault(
                segment,
                {
                    "article_count": 0,
                    "total_weight": 0.0,
                    "signals": {key: 0.0 for key in SIGNAL_KEYS},
                    "frames": {},
                },
            )
            profile["article_count"] = int(profile["article_count"]) + 1
            profile["total_weight"] = float(profile["total_weight"]) + weight
            for signal_index, key in enumerate(SIGNAL_KEYS):
                profile["signals"][key] += float(row[1 + signal_index] or 0.0) * weight
            frame = canonicalize_frame(row[4])
            frames = profile["frames"]
            frames[frame] = float(frames.get(frame, 0.0)) + weight

    for by_segment in profiles.values():
        for profile in by_segment.values():
            total_weight = float(profile["total_weight"])
            if total_weight <= 0.0:
                continue
            profile["current"] = {
                key: round(float(value) / total_weight, 4)
                for key, value in profile["signals"].items()
            }
            frames = profile["frames"]
            profile["dominant_frame"] = (
                max(frames, key=frames.get) if frames else "neutral"
            )

    return profiles


def _combine_panel_profile(
    outlet_profiles: dict[str, dict[str, dict[str, object]]],
    *,
    segment: str,
    panel_outlets: list[str],
) -> dict[str, object] | None:
    observed = [
        outlet
        for outlet in panel_outlets
        if segment in outlet_profiles.get(outlet, {})
    ]
    if not observed:
        return None

    outlet_weight = 1.0 / len(observed)
    combined = {key: 0.0 for key in SIGNAL_KEYS}
    frame_counts: dict[str, float] = {}
    article_count = 0
    for outlet in observed:
        profile = outlet_profiles[outlet][segment]
        current = profile.get("current", {})
        for key in SIGNAL_KEYS:
            combined[key] += outlet_weight * float(current.get(key, 0.0))
        frame = str(profile.get("dominant_frame", "neutral"))
        frame_counts[frame] = frame_counts.get(frame, 0.0) + outlet_weight
        article_count += int(profile.get("article_count", 0) or 0)

    return {
        "current": {key: round(value, 4) for key, value in combined.items()},
        "dominant_frame": max(frame_counts, key=frame_counts.get) if frame_counts else "neutral",
        "article_count": article_count,
        "observed_outlets": observed,
        "missing_outlets": [outlet for outlet in panel_outlets if outlet not in observed],
    }


def _source_normalized_entry(
    raw_entry: dict[str, object],
    normalized_profile: dict[str, object] | None,
    baseline: dict | None,
    signal_weights: dict[str, float],
    panel_outlets: list[str],
) -> dict[str, object]:
    if normalized_profile is None or baseline is None:
        reason = "no_panel_overlap" if normalized_profile is None else "missing_baseline"
        return {
            "status": reason,
            "panel_outlets": panel_outlets,
            "observed_outlets": [],
            "missing_outlets": panel_outlets,
            "current": None,
            "baseline": (
                {key: baseline[key] for key in SIGNAL_KEYS}
                if baseline is not None
                else None
            ),
            "deltas": None,
            "drift_magnitude": None,
            "alert_level": "no_data" if normalized_profile is None else "none",
            "normalization_effect": None,
            "interpretation": reason,
        }

    current = dict(normalized_profile["current"])
    deltas = {
        key: round(float(current[key]) - float(baseline[key] or 0.0), 4)
        for key in SIGNAL_KEYS
    }
    drift_magnitude = round(
        sum(signal_weights.get(key, 0.0) * abs(delta) for key, delta in deltas.items()),
        4,
    )
    raw_magnitude = float(raw_entry.get("drift_magnitude", 0.0) or 0.0)
    normalization_effect = round(raw_magnitude - drift_magnitude, 4)
    if abs(normalization_effect) < 0.05:
        interpretation = "similar_to_raw"
    elif normalization_effect > 0:
        interpretation = "raw_drift_partly_source_mix"
    else:
        interpretation = "source_normalized_drift_stronger"

    return {
        "status": (
            "ready"
            if not normalized_profile["missing_outlets"]
            else "partial_panel"
        ),
        "panel_outlets": panel_outlets,
        "observed_outlets": normalized_profile["observed_outlets"],
        "missing_outlets": normalized_profile["missing_outlets"],
        "article_count": normalized_profile["article_count"],
        "current": current,
        "baseline": {key: baseline[key] for key in SIGNAL_KEYS},
        "deltas": deltas,
        "drift_magnitude": drift_magnitude,
        "alert_level": _alert_level(drift_magnitude, int(normalized_profile["article_count"])),
        "dominant_frame": normalized_profile["dominant_frame"],
        "baseline_frame": baseline["dominant_frame"],
        "normalization_effect": normalization_effect,
        "interpretation": interpretation,
    }


def compute_source_normalized_drift(
    raw_results: list[dict],
    topic: str,
    days_back: int = 7,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    signal_weights: dict[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    panel_outlets = _source_panel_for_scope(topic, country=country, source=source)
    rows = _current_rows_by_outlet(
        topic,
        days_back,
        country=country,
        source=source,
        language=language,
    )
    outlet_profiles = _per_outlet_segment_profiles(rows)
    weights = signal_weights or get_domain_config(topic_to_domain(topic))["signal_weights"]
    normalized: dict[str, dict[str, object]] = {}

    for raw_entry in raw_results:
        segment = str(raw_entry["segment"])
        baseline = _get_baseline(topic, segment)
        normalized_profile = _combine_panel_profile(
            outlet_profiles,
            segment=segment,
            panel_outlets=panel_outlets,
        )
        normalized[segment] = _source_normalized_entry(
            raw_entry,
            normalized_profile,
            baseline,
            weights,
            panel_outlets,
        )

    return normalized


def compute_drift(
    topic: str,
    days_back: int = 7,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> list[dict]:
    ensure_topic_baselines(topic)
    profiles = compute_segment_profiles(
        topic,
        days_back,
        learn_baseline=False,
        country=country,
        source=source,
        language=language,
    )
    domain = topic_to_domain(topic)
    domain_config = get_domain_config(domain)
    signal_weights = domain_config["signal_weights"]
    relevant_fields = list(domain_config["relevant_fields"])
    results: list[dict] = []

    for profile in profiles:
        segment = profile["segment"]
        baseline = _get_baseline(topic, segment)
        has_data = profile["article_count"] > 0
        sample_count = baseline["sample_count"] if baseline else 0
        is_learned = baseline["is_learned"] if baseline else False
        baseline_age_days = _baseline_age_days(
            baseline["updated_at"] if baseline else None
        )
        confidence = _compute_confidence(
            profile["article_count"],
            sample_count,
            is_learned,
        )
        status = _segment_status(
            profile["article_count"],
            is_learned,
            confidence,
        )

        if not has_data:
            results.append(
                {
                    "segment": segment,
                    "topic": topic,
                    "article_count": profile["article_count"],
                    "has_data": False,
                    "current": {key: profile[key] for key in SIGNAL_KEYS},
                    "baseline": (
                        {key: baseline[key] for key in SIGNAL_KEYS}
                        if baseline is not None
                        else None
                    ),
                    "deltas": {key: 0.0 for key in SIGNAL_KEYS},
                    "drift_magnitude": 0.0,
                    "frame_shift": False,
                    "alert_level": "no_data",
                    "dominant_frame": profile["dominant_frame"],
                    "baseline_frame": baseline["dominant_frame"] if baseline else None,
                    "confidence": confidence,
                    "baseline_is_learned": is_learned,
                    "baseline_sample_count": sample_count,
                    "baseline_age_days": baseline_age_days,
                    "status": status,
                    "domain": domain,
                    "relevant_fields": relevant_fields,
                }
            )
            continue

        if baseline is None:
            results.append(
                {
                    "segment": segment,
                    "topic": topic,
                    "article_count": profile["article_count"],
                    "has_data": True,
                    "current": {key: profile[key] for key in SIGNAL_KEYS},
                    "baseline": None,
                    "deltas": {key: 0.0 for key in SIGNAL_KEYS},
                    "drift_magnitude": 0.0,
                    "frame_shift": False,
                    "alert_level": "none",
                    "dominant_frame": profile["dominant_frame"],
                    "baseline_frame": None,
                    "confidence": confidence,
                    "baseline_is_learned": is_learned,
                    "baseline_sample_count": sample_count,
                    "baseline_age_days": baseline_age_days,
                    "status": status,
                    "domain": domain,
                    "relevant_fields": relevant_fields,
                }
            )
            continue

        deltas = {
            key: round(profile[key] - (baseline[key] or 0.0), 4)
            for key in SIGNAL_KEYS
        }
        drift_magnitude = round(
            sum(signal_weights.get(key, 0.0) * abs(delta) for key, delta in deltas.items()),
            4,
        )
        frame_shift = (
            canonicalize_frame(profile["dominant_frame"])
            != canonicalize_frame(baseline["dominant_frame"])
        )

        results.append(
            {
                "segment": segment,
                "topic": topic,
                "article_count": profile["article_count"],
                "has_data": True,
                "current": {key: profile[key] for key in SIGNAL_KEYS},
                "baseline": {key: baseline[key] for key in SIGNAL_KEYS},
                "deltas": deltas,
                "drift_magnitude": drift_magnitude,
                "frame_shift": frame_shift,
                "alert_level": _alert_level(drift_magnitude, profile["article_count"]),
                "dominant_frame": profile["dominant_frame"],
                "baseline_frame": baseline["dominant_frame"],
                "confidence": confidence,
                "baseline_is_learned": is_learned,
                "baseline_sample_count": sample_count,
                "baseline_age_days": baseline_age_days,
                "status": status,
                "domain": domain,
                "relevant_fields": relevant_fields,
            }
        )

    source_mix = compute_source_mix(
        topic,
        days_back=days_back,
        country=country,
        source=source,
        language=language,
    )
    source_normalized = compute_source_normalized_drift(
        results,
        topic,
        days_back=days_back,
        country=country,
        source=source,
        language=language,
        signal_weights=signal_weights,
    )
    for item in results:
        item["source_mix"] = source_mix
        item["source_normalized"] = source_normalized.get(item["segment"])

    return results


def update_baseline_from_profile(
    topic: str,
    segment: str,
    profile: dict,
    article_count: int | None = None,
    conn=None,
) -> bool:
    if article_count is None:
        article_count = int(profile.get("article_count", 0) or 0)
    if article_count < MIN_ARTICLES_FOR_BASELINE:
        return False

    if conn is None:
        conn = get_conn()

    now = datetime.now(timezone.utc).isoformat()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
    old = _get_baseline(topic, segment)

    if old is None:
        conn.execute(
            """
            INSERT INTO baselines
            (id, topic, segment, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seeded, sample_count, is_learned, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, 1, ?)
            """,
            (
                row_id,
                topic,
                segment,
                round(profile["concern_level"], 4),
                round(profile["purchase_intent"], 4),
                round(profile["avoidance_signals"], 4),
                canonicalize_frame(profile["dominant_frame"]),
                now,
            ),
        )
        conn.commit()
        return True

    blended = {
        key: round(
            BASELINE_EMA_ALPHA * (old[key] or 0.0)
            + (1 - BASELINE_EMA_ALPHA) * profile[key],
            4,
        )
        for key in SIGNAL_KEYS
    }

    conn.execute(
        """
        UPDATE baselines
        SET concern_level=?, purchase_intent=?, avoidance_signals=?,
            dominant_frame=?, seeded=0, sample_count=sample_count + 1,
            is_learned=1, updated_at=?
        WHERE id=?
        """,
        (
            blended["concern_level"],
            blended["purchase_intent"],
            blended["avoidance_signals"],
            canonicalize_frame(profile["dominant_frame"]),
            now,
            row_id,
        ),
    )
    conn.commit()
    return True
