from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.domains import get_domain_config, topic_to_domain
from db.init import get_conn
from delta.mapper import SEGMENTS, SIGNAL_KEYS
from delta.seeder import ensure_topic_baselines


def topic_filter_sql(article_alias: str, topic: str) -> tuple[str, list[object]]:
    if not topic:
        return "", []
    return (
        f"""
          AND EXISTS (
              SELECT 1
              FROM article_topics at
              WHERE at.article_id = {article_alias}.id
                AND at.topic = ?
          )
        """,
        [topic],
    )


def get_digest_articles(
    topic: str,
    country: str = "",
    source: str = "",
    limit: int = 8,
) -> list[dict]:
    conn = get_conn()
    query = """
        SELECT
            a.id,
            a.title,
            a.summary,
            a.body,
            a.url,
            COALESCE(a.canonical_url, a.url) AS canonical_url,
            a.outlet,
            a.country,
            a.language,
            a.published_at,
            at.relevance_score,
            at.matched_at
        FROM article_topics at
        JOIN articles a ON a.id = at.article_id
        WHERE at.topic = ?
    """
    params: list[object] = [topic]
    if country:
        query += " AND a.country = ?"
        params.append(country.strip().upper())
    if source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(source.strip().lower())

    query += """
        ORDER BY at.relevance_score DESC, a.published_at DESC, at.matched_at DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [
        {
            "article_id": row[0],
            "title": row[1],
            "summary": row[2],
            "body": row[3],
            "url": row[4],
            "canonical_url": row[5],
            "outlet": row[6],
            "country": row[7],
            "language": row[8],
            "published_at": row[9],
            "relevance_score": float(row[10] or 0.0),
            "matched_at": row[11],
        }
        for row in rows
    ]


def get_profile_history(
    topic: str,
    days_back: int,
    segment: str | None = None,
) -> list[dict]:
    ensure_topic_baselines(topic)

    conn = get_conn()
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    target_segments = [segment] if segment else list(SEGMENTS)
    domain_config = get_domain_config(topic_to_domain(topic))
    signal_weights = domain_config["signal_weights"]

    rows = conn.execute(
        """
        SELECT
            segment,
            SUBSTR(computed_at, 1, 10) AS day,
            AVG(COALESCE(concern_level, 0.0)) AS concern_level,
            AVG(COALESCE(purchase_intent, 0.0)) AS purchase_intent,
            AVG(COALESCE(avoidance_signals, 0.0)) AS avoidance_signals
        FROM segment_profiles
        WHERE topic = ?
          AND computed_at >= ?
          AND (? IS NULL OR segment = ?)
        GROUP BY segment, day
        ORDER BY segment ASC, day ASC
        """,
        (topic, since, segment, segment),
    ).fetchall()

    baselines = {
        row[0]: {
            "concern_level": float(row[1] or 0.0),
            "purchase_intent": float(row[2] or 0.0),
            "avoidance_signals": float(row[3] or 0.0),
        }
        for row in conn.execute(
            """
            SELECT segment, concern_level, purchase_intent, avoidance_signals
            FROM baselines
            WHERE topic = ?
            """,
            (topic,),
        ).fetchall()
    }

    history_by_segment = {
        name: {
            "segment": name,
            "dates": [],
            "concern": [],
            "purchase_intent": [],
            "avoidance": [],
            "drift_magnitude": [],
        }
        for name in target_segments
    }

    for row in rows:
        current_segment = row[0]
        entry = history_by_segment.get(current_segment)
        if entry is None:
            continue

        current_signals = {
            "concern_level": round(float(row[2] or 0.0), 4),
            "purchase_intent": round(float(row[3] or 0.0), 4),
            "avoidance_signals": round(float(row[4] or 0.0), 4),
        }
        baseline = baselines.get(current_segment)
        drift_magnitude = 0.0
        if baseline is not None:
            drift_magnitude = round(
                sum(
                    signal_weights.get(key, 0.0) * abs(current_signals[key] - baseline[key])
                    for key in SIGNAL_KEYS
                ),
                4,
            )

        entry["dates"].append(row[1])
        entry["concern"].append(current_signals["concern_level"])
        entry["purchase_intent"].append(current_signals["purchase_intent"])
        entry["avoidance"].append(current_signals["avoidance_signals"])
        entry["drift_magnitude"].append(drift_magnitude)

    return [history_by_segment[name] for name in target_segments]
