import hashlib
from datetime import datetime, timedelta, timezone

from db.init import get_conn


SEGMENTS = ["young_urban", "family", "senior", "b2b"]
SIGNAL_KEYS = ["concern_level", "purchase_intent", "avoidance_signals"]
FRAME_ALIASES = {
    "alert": "fear",
    "anxiety": "fear",
    "concern": "fear",
    "fear": "fear",
    "risk": "fear",
    "threat": "fear",
    "benefit": "opportunity",
    "growth": "opportunity",
    "opportunity": "opportunity",
    "solution": "opportunity",
    "controversy": "conflict",
    "conflict": "conflict",
    "debate": "conflict",
    "dispute": "conflict",
    "explanatory": "neutral",
    "informational": "neutral",
    "mixed": "neutral",
    "neutral": "neutral",
}


def canonicalize_frame(frame: str | None) -> str:
    if not frame:
        return "neutral"

    normalized = frame.strip().lower().replace("-", "_").replace(" ", "_")
    return FRAME_ALIASES.get(normalized, normalized)


def _window_start(days_back: int = 7) -> str:
    today = datetime.now(timezone.utc)
    window_begin = today - timedelta(days=days_back)
    return window_begin.isoformat()


def compute_segment_profiles(
    topic: str,
    days_back: int = 7,
    learn_baseline: bool = False,
) -> list[dict]:
    conn = get_conn()
    since = _window_start(days_back)

    query = """
        SELECT
            s.concern_level, s.purchase_intent, s.avoidance_signals,
            s.dominant_frame,
            s.seg_young_urban, s.seg_family, s.seg_senior, s.seg_b2b
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE s.extracted_at >= ?
    """
    params: list[str] = [since]
    if topic:
        query += " AND a.topic = ?"
        params.append(topic)

    rows = conn.execute(query, params).fetchall()

    profiles: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    window_start_str = since[:10]

    for seg_idx, segment in enumerate(SEGMENTS):
        seg_col = 4 + seg_idx
        weighted_signals = {key: 0.0 for key in SIGNAL_KEYS}
        frame_counts: dict[str, float] = {}
        total_weight = 0.0
        article_count = 0

        for row in rows:
            weight = row[seg_col] or 0.0
            if weight <= 0:
                continue
            total_weight += weight
            article_count += 1
            for idx, key in enumerate(SIGNAL_KEYS):
                signal_value = row[idx] or 0.0
                weighted_signals[key] += signal_value * weight
            frame = canonicalize_frame(row[3])
            frame_counts[frame] = frame_counts.get(frame, 0.0) + weight

        if total_weight > 0:
            profile_signals = {
                key: round(value / total_weight, 4)
                for key, value in weighted_signals.items()
            }
            dominant_frame = max(frame_counts, key=frame_counts.get)
        else:
            profile_signals = {key: 0.0 for key in SIGNAL_KEYS}
            dominant_frame = "neutral"

        profile_id = hashlib.sha256(
            f"{topic}:{segment}:{window_start_str}".encode()
        ).hexdigest()

        conn.execute(
            """
            INSERT OR REPLACE INTO segment_profiles
            (id, topic, segment, window_start, window_days,
             concern_level, purchase_intent, avoidance_signals,
             dominant_frame, article_count, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                topic,
                segment,
                window_start_str,
                days_back,
                profile_signals["concern_level"],
                profile_signals["purchase_intent"],
                profile_signals["avoidance_signals"],
                dominant_frame,
                article_count,
                now,
            ),
        )

        profiles.append(
            {
                "segment": segment,
                "topic": topic,
                "window_start": window_start_str,
                "window_days": days_back,
                "article_count": article_count,
                **profile_signals,
                "dominant_frame": dominant_frame,
            }
        )

    conn.commit()

    if learn_baseline:
        from delta.engine import update_baseline_from_profile

        for profile in profiles:
            update_baseline_from_profile(
                topic,
                profile["segment"],
                profile,
                article_count=profile["article_count"],
                conn=conn,
            )

    return profiles
