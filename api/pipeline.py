from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from db.init import get_conn
from ingestion.bandit import record_signal_reward
from ingestion.crawler import _crawl_async


def _decode_reward_signals(
    raw_json: str | None,
    concern_level: float | None,
    purchase_intent: float | None,
    avoidance_signals: float | None,
) -> dict[str, object]:
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "concern_level": concern_level or 0.0,
        "purchase_intent": purchase_intent or 0.0,
        "avoidance_signals": avoidance_signals or 0.0,
    }


def record_recent_signal_rewards(topic: str, crawl_start: str) -> int:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT a.outlet, a.topic, a.published_at,
               s.concern_level, s.purchase_intent, s.avoidance_signals, s.raw_json
        FROM signals s
        JOIN articles a ON a.id = s.article_id
        WHERE (a.topic = ? OR ? = '')
          AND a.fetched_at >= ?
        ORDER BY s.extracted_at ASC, s.article_id ASC
        """,
        (topic, topic, crawl_start),
    ).fetchall()

    rewards_recorded = 0
    for (
        outlet,
        article_topic,
        published_at,
        concern_level,
        purchase_intent,
        avoidance_signals,
        raw_json,
    ) in rows:
        signals = _decode_reward_signals(
            raw_json,
            concern_level,
            purchase_intent,
            avoidance_signals,
        )
        record_signal_reward(
            outlet,
            article_topic or topic,
            signals,
            when=published_at,
        )
        rewards_recorded += 1

    return rewards_recorded


async def run_collection_cycle(topic: str) -> dict[str, int | str]:
    crawl_start = datetime.now(timezone.utc).isoformat()
    inserted = await _crawl_async(topic)
    processed = 0
    rewards_recorded = 0

    if inserted > 0:
        from extraction.extractor import run_extraction

        processed = run_extraction(topic, record_bandit_reward=False)
        rewards_recorded = record_recent_signal_rewards(topic, crawl_start)
        if processed > 0:
            from brief.generator import clear_brief_cache

            clear_brief_cache()

    return {
        "inserted": inserted,
        "extracted": processed,
        "rewards_recorded": rewards_recorded,
        "topic": topic,
    }


def run_collection_cycle_sync(topic: str) -> dict[str, int | str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_collection_cycle(topic))

    raise RuntimeError(
        "run_collection_cycle_sync() cannot run inside an active event loop; "
        "await run_collection_cycle(topic) instead."
    )
