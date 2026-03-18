import json
from datetime import datetime, timezone

from config.feeds import FEEDS
from db.init import get_conn
from extraction.entities import extract_entities
from extraction.llm_client import extract_signals
from ingestion.bandit import reward_from_signals, update_feed_reward

_OUTLET_AFFINITY = {feed["outlet"]: feed["affinity_tag"] for feed in FEEDS}
_OUTLET_FEEDS = {feed["outlet"]: feed for feed in FEEDS}


def run_extraction(topic: str) -> int:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.summary, a.outlet, a.topic, a.published_at
        FROM articles a
        LEFT JOIN signals s ON a.id = s.article_id
        WHERE s.article_id IS NULL
          AND (a.topic = ? OR ? = '')
        """,
        (topic, topic),
    ).fetchall()

    processed = 0
    reward_batches: dict[tuple[str, str], dict[str, object]] = {}

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
            reward_key = (outlet, resolved_topic)
            batch = reward_batches.setdefault(
                reward_key,
                {
                    "feed": _OUTLET_FEEDS.get(outlet),
                    "published_at": published_at or extracted_at,
                    "rewards": [],
                },
            )
            batch["rewards"].append(reward_from_signals(resolved_topic, signals))

    for (outlet, resolved_topic), batch in reward_batches.items():
        rewards = batch["rewards"]
        if not rewards or batch["feed"] is None:
            continue
        average_reward = sum(rewards) / len(rewards)
        update_feed_reward(
            outlet,
            resolved_topic,
            average_reward,
            when=batch["published_at"],
            feed=batch["feed"],
        )

    return processed
