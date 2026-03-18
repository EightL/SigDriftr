import hashlib
from datetime import datetime, timezone

from config.settings import (
    BASELINE_EMA_ALPHA,
    CONFIDENCE_ARTICLE_DENOMINATOR,
    CONFIDENCE_BASELINE_WEIGHT,
    CONFIDENCE_COVERAGE_WEIGHT,
    MIN_ARTICLES_FOR_BASELINE,
)
from db.init import get_conn
from delta.mapper import SIGNAL_KEYS, canonicalize_frame, compute_segment_profiles
from delta.seeder import ensure_topic_baselines


ALERT_THRESHOLDS = {
    "mild": 0.20,
    "strong": 0.45,
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


def compute_drift(topic: str, days_back: int = 7) -> list[dict]:
    ensure_topic_baselines(topic)
    profiles = compute_segment_profiles(topic, days_back, learn_baseline=False)
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
                    "confidence": confidence,
                    "baseline_is_learned": is_learned,
                    "baseline_sample_count": sample_count,
                    "baseline_age_days": baseline_age_days,
                }
            )
            continue

        deltas = {
            key: round(profile[key] - (baseline[key] or 0.0), 4)
            for key in SIGNAL_KEYS
        }
        drift_magnitude = round(sum(abs(value) for value in deltas.values()), 4)
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
            }
        )

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
