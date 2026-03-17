import json
from datetime import datetime, timezone

from config.feeds import FEEDS
from db.init import get_conn
from extraction.llm_client import extract_signals

_OUTLET_AFFINITY = {feed["outlet"]: feed["affinity_tag"] for feed in FEEDS}


def run_extraction(topic: str) -> int:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.summary, a.outlet
        FROM articles a
        LEFT JOIN signals s ON a.id = s.article_id
        WHERE s.article_id IS NULL
          AND (a.topic = ? OR ? = '')
        """,
        (topic, topic),
    ).fetchall()

    processed = 0
    extracted_at = datetime.now(timezone.utc).isoformat()

    for article_id, title, summary, outlet in rows:
        affinity_tag = _OUTLET_AFFINITY.get(outlet, "mainstream")
        signals = extract_signals(
            title or "", summary or "", affinity_tag=affinity_tag
        )
        conn.execute(
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
        processed += 1

    conn.commit()
    return processed
