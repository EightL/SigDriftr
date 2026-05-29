import json
from datetime import datetime, timezone

from config.feeds import FEEDS
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from extraction.entities import extract_entities, normalize_entity_key
from extraction.llm_client import extract_signals
from ingestion.bandit import record_signal_reward

_OUTLET_AFFINITY = {feed["outlet"]: feed["affinity_tag"] for feed in FEEDS}
_OUTLET_FEEDS = {feed["outlet"]: feed for feed in FEEDS}


def _topic_relevance_label(score: float | None) -> str:
    if score is None:
        return ""
    if score < 0.55:
        return "0"
    if score < 0.80:
        return "1"
    return "2"


def run_extraction(topic: str, record_bandit_reward: bool = True) -> int:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    rows = conn.execute(
        f"""
        SELECT a.id, a.title, a.summary, a.body, a.outlet, a.topic, a.published_at,
               COALESCE(
                   (SELECT MAX(at.relevance_score)
                    FROM article_topics at
                    WHERE at.article_id = a.id
                      AND (? = '' OR at.topic = ?)),
                   1.0
               ) AS topic_relevance_score
        FROM articles a
        LEFT JOIN signals s ON a.id = s.article_id
        WHERE s.article_id IS NULL
          {topic_sql}
        """,
        [topic, topic, *topic_params],
    ).fetchall()

    processed = 0

    for (
        article_id,
        title,
        summary,
        body,
        outlet,
        article_topic,
        published_at,
        topic_relevance_score,
    ) in rows:
        affinity_tag = _OUTLET_AFFINITY.get(outlet, "mainstream")
        resolved_topic = article_topic or topic
        extract_kwargs = {
            "affinity_tag": affinity_tag,
            "topic": resolved_topic,
        }
        if body:
            extract_kwargs["body"] = body
        signals = extract_signals(title or "", summary or "", **extract_kwargs)
        signals["topic_relevance_score"] = round(float(topic_relevance_score or 0.0), 4)
        signals["topic_relevance"] = _topic_relevance_label(
            signals["topic_relevance_score"]
        )
        entities = extract_entities(f"{title or ''}\n{summary or ''}\n{body or ''}")
        if entities:
            signals["entities"] = entities
        extracted_at = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO signals
            (article_id, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
             seg_young_urban_relevance, seg_family_relevance,
             seg_senior_relevance, seg_b2b_relevance,
             raw_json, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                signals["concern_level"],
                signals["purchase_intent"],
                signals["avoidance_signals"],
                signals["dominant_frame"],
                signals["seg_young_urban"],
                signals["seg_family"],
                signals["seg_senior"],
                signals["seg_b2b"],
                signals.get("seg_young_urban_relevance", signals["seg_young_urban"]),
                signals.get("seg_family_relevance", signals["seg_family"]),
                signals.get("seg_senior_relevance", signals["seg_senior"]),
                signals.get("seg_b2b_relevance", signals["seg_b2b"]),
                json.dumps(signals),
                extracted_at,
            ),
        )
        if cursor.rowcount:
            processed += 1
            for entity in entities:
                entity_text = entity["text"]
                entity_label = entity["label"]
                conn.execute(
                    """
                    INSERT OR IGNORE INTO article_entities
                    (article_id, entity_text, entity_norm, entity_label)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        article_id,
                        entity_text,
                        normalize_entity_key(entity_text),
                        entity_label,
                    ),
                )
            if record_bandit_reward:
                feed = _OUTLET_FEEDS.get(outlet)
                if feed is None:
                    continue
                record_signal_reward(
                    outlet,
                    resolved_topic,
                    signals,
                    when=published_at or extracted_at,
                    feed=feed,
                )

    return processed
