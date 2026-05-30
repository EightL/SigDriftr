from __future__ import annotations

from db.init import get_conn
from db.topic_resolver import resolve_topic


def topic_filter_sql(article_alias: str, topic: str) -> tuple[str, list[object]]:
    if not topic:
        return "", []
    resolution = resolve_topic(topic)
    return (
        f"""
          AND EXISTS (
              SELECT 1
              FROM article_topics at
              WHERE at.article_id = {article_alias}.id
                AND (
                    at.canonical_topic_id = ?
                    OR at.topic = ?
                )
          )
        """,
        [resolution.canonical_topic_id, topic],
    )


def get_digest_articles(
    topic: str,
    country: str = "",
    source: str = "",
    limit: int = 8,
) -> list[dict]:
    conn = get_conn()
    resolution = resolve_topic(topic)
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
            at.topic,
            at.raw_topic,
            at.canonical_topic_id,
            at.relevance_score,
            at.matched_at
        FROM article_topics at
        JOIN articles a ON a.id = at.article_id
        WHERE (
            at.canonical_topic_id = ?
            OR at.topic = ?
        )
    """
    params: list[object] = [resolution.canonical_topic_id, topic]
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
            "topic": row[10],
            "raw_topic": row[11],
            "canonical_topic_id": row[12],
            "relevance_score": float(row[13] or 0.0),
            "matched_at": row[14],
        }
        for row in rows
    ]
