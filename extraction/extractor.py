import json
from datetime import datetime, timezone

from config.feeds import FEEDS
from db.queries import topic_filter_sql
from db.init import get_conn
from extraction.entities import extract_entities, normalize_entity_key
from extraction.llm_client import extract_signals
from ingestion.bandit import record_signal_reward

_OUTLET_AFFINITY = {feed["outlet"]: feed["affinity_tag"] for feed in FEEDS}
_OUTLET_FEEDS = {feed["outlet"]: feed for feed in FEEDS}


def run_extraction(topic: str, record_bandit_reward: bool = True) -> int:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    rows = conn.execute(
        f"""
        SELECT a.id, a.title, a.summary, a.outlet, a.topic, a.published_at
        FROM articles a
        LEFT JOIN signals s ON a.id = s.article_id
        WHERE s.article_id IS NULL
          {topic_sql}
        """,
        topic_params,
    ).fetchall()

    processed = 0

    for article_id, title, summary, outlet, article_topic, published_at in rows:
        affinity_tag = _OUTLET_AFFINITY.get(outlet, "mainstream")
        resolved_topic = article_topic or topic
        signals = extract_signals(
            title or "",
            summary or "",
            affinity_tag=affinity_tag,
            topic=resolved_topic,
        )
        entities = extract_entities(f"{title or ''}\n{summary or ''}")
        if entities:
            signals["entities"] = entities
        extracted_at = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO signals
            (article_id, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
             raw_json, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
