import hashlib
from datetime import datetime, timezone

from db.init import get_conn


SEED_PRIORS: dict[str, dict[str, float | str]] = {
    "young_urban": {
        "concern_level": 0.38,
        "purchase_intent": 0.31,
        "avoidance_signals": 0.18,
        "dominant_frame": "opportunity",
    },
    "family": {
        "concern_level": 0.52,
        "purchase_intent": 0.27,
        "avoidance_signals": 0.29,
        "dominant_frame": "fear",
    },
    "senior": {
        "concern_level": 0.61,
        "purchase_intent": 0.14,
        "avoidance_signals": 0.41,
        "dominant_frame": "fear",
    },
    "b2b": {
        "concern_level": 0.44,
        "purchase_intent": 0.48,
        "avoidance_signals": 0.21,
        "dominant_frame": "conflict",
    },
}

SEGMENTS = list(SEED_PRIORS.keys())

DEFAULT_TOPICS = [
    "",
    "energie",
    "inflace",
    "bydleni",
    "zdravi",
    "politika",
    "ekonomika",
]


def ensure_topic_baselines(topic: str) -> int:
    return seed_baselines([topic])


def seed_baselines(topics: list[str] | None = None) -> int:
    conn = get_conn()
    if topics is None:
        topics = DEFAULT_TOPICS

    inserted = 0
    now = datetime.now(timezone.utc).isoformat()

    for topic in topics:
        for segment in SEGMENTS:
            row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
            prior = SEED_PRIORS[segment]
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO baselines
                (id, topic, segment, concern_level, purchase_intent,
                 avoidance_signals, dominant_frame, seeded, sample_count,
                 is_learned, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, 0, ?)
                """,
                (
                    row_id,
                    topic,
                    segment,
                    prior["concern_level"],
                    prior["purchase_intent"],
                    prior["avoidance_signals"],
                    prior["dominant_frame"],
                    now,
                ),
            )
            inserted += cursor.rowcount

    conn.commit()
    return inserted
