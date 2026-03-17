import hashlib
from datetime import datetime, timezone

from db.init import get_conn
from delta.mapper import SEGMENTS, SIGNAL_KEYS, compute_segment_profiles
from delta.seeder import ensure_topic_baselines


ALERT_THRESHOLDS = {
    "mild": 0.10,
    "strong": 0.25,
}


def _get_baseline(topic: str, segment: str) -> dict | None:
    conn = get_conn()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
    row = conn.execute(
        """
        SELECT concern_level, purchase_intent, avoidance_signals, dominant_frame
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
        "dominant_frame": row[3],
    }


def _alert_level(deltas: dict[str, float]) -> str:
    max_abs = max(abs(value) for value in deltas.values()) if deltas else 0.0
    if max_abs >= ALERT_THRESHOLDS["strong"]:
        return "strong"
    if max_abs >= ALERT_THRESHOLDS["mild"]:
        return "mild"
    return "none"


def compute_drift(topic: str, days_back: int = 7) -> list[dict]:
    ensure_topic_baselines(topic)
    profiles = compute_segment_profiles(topic, days_back)
    results: list[dict] = []

    for profile in profiles:
        segment = profile["segment"]
        baseline = _get_baseline(topic, segment)

        if baseline is None:
            results.append(
                {
                    "segment": segment,
                    "topic": topic,
                    "article_count": profile["article_count"],
                    "current": {key: profile[key] for key in SIGNAL_KEYS},
                    "baseline": None,
                    "deltas": {key: 0.0 for key in SIGNAL_KEYS},
                    "drift_magnitude": 0.0,
                    "frame_shift": False,
                    "alert_level": "none",
                    "dominant_frame": profile["dominant_frame"],
                }
            )
            continue

        deltas = {
            key: round(profile[key] - (baseline[key] or 0.0), 4)
            for key in SIGNAL_KEYS
        }
        drift_magnitude = round(sum(abs(value) for value in deltas.values()), 4)
        frame_shift = profile["dominant_frame"] != baseline["dominant_frame"]

        results.append(
            {
                "segment": segment,
                "topic": topic,
                "article_count": profile["article_count"],
                "current": {key: profile[key] for key in SIGNAL_KEYS},
                "baseline": {key: baseline[key] for key in SIGNAL_KEYS},
                "deltas": deltas,
                "drift_magnitude": drift_magnitude,
                "frame_shift": frame_shift,
                "alert_level": _alert_level(deltas),
                "dominant_frame": profile["dominant_frame"],
                "baseline_frame": baseline["dominant_frame"],
            }
        )

    return results


def update_baseline_from_profile(topic: str, segment: str, profile: dict) -> None:
    if profile.get("article_count", 0) < 10:
        return

    conn = get_conn()
    old = _get_baseline(topic, segment)
    if old is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
    blended = {
        key: round(0.8 * (old[key] or 0.0) + 0.2 * profile[key], 4)
        for key in SIGNAL_KEYS
    }

    conn.execute(
        """
        UPDATE baselines
        SET concern_level=?, purchase_intent=?, avoidance_signals=?,
            dominant_frame=?, seeded=0, updated_at=?
        WHERE id=?
        """,
        (
            blended["concern_level"],
            blended["purchase_intent"],
            blended["avoidance_signals"],
            profile["dominant_frame"],
            now,
            row_id,
        ),
    )
    conn.commit()
