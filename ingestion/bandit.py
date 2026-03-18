from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Sequence

from config.domains import DOMAIN_SIGNAL_KEYS, DOMAINS, get_domain_config, topic_to_domain
from config.feeds import FEEDS
from config.settings import (
    BANDIT_ALPHA,
    BANDIT_MAX_FEEDS_PER_CRAWL,
    BANDIT_TIME_BUCKET_HOURS,
)
from db.init import get_conn


_AFFINITY_TAGS = sorted({feed["affinity_tag"] for feed in FEEDS})
_DOMAIN_KEYS = sorted(DOMAINS)
_TIME_BUCKET_COUNT = len(range(0, 24, BANDIT_TIME_BUCKET_HOURS))
_CONTEXT_DIM = 1 + len(_AFFINITY_TAGS) + len(_DOMAIN_KEYS) + _TIME_BUCKET_COUNT


def _identity_matrix(size: int) -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def _zero_vector(size: int) -> list[float]:
    return [0.0 for _ in range(size)]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _solve_linear_system(matrix: list[list[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    augmented = [
        [float(value) for value in row] + [float(vector[index])]
        for index, row in enumerate(matrix)
    ]

    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[pivot_row][pivot]) < 1e-9:
            raise ValueError("Bandit matrix is singular.")
        if pivot_row != pivot:
            augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

        pivot_value = augmented[pivot][pivot]
        for col in range(pivot, size + 1):
            augmented[pivot][col] /= pivot_value

        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor == 0:
                continue
            for col in range(pivot, size + 1):
                augmented[row][col] -= factor * augmented[pivot][col]

    return [augmented[row][-1] for row in range(size)]


def _utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        if not value:
            return datetime.now(timezone.utc)
        parsed = datetime.fromisoformat(value)
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _find_feed(outlet: str, feeds: Sequence[dict] | None = None) -> dict:
    active_feeds = feeds or FEEDS
    for feed in active_feeds:
        if feed["outlet"] == outlet:
            return feed
    raise KeyError(f"Unknown outlet for bandit arm: {outlet}")


def build_context_vector(
    topic: str,
    feed: dict,
    when: datetime | str | None = None,
) -> list[float]:
    resolved = _utc_datetime(when)
    hour_bucket = min(
        _TIME_BUCKET_COUNT - 1,
        resolved.hour // BANDIT_TIME_BUCKET_HOURS,
    )
    domain = topic_to_domain(topic)

    vector = [0.0 for _ in range(_CONTEXT_DIM)]
    vector[0] = 1.0

    affinity_offset = 1
    if feed.get("affinity_tag") in _AFFINITY_TAGS:
        vector[affinity_offset + _AFFINITY_TAGS.index(feed["affinity_tag"])] = 1.0

    domain_offset = affinity_offset + len(_AFFINITY_TAGS)
    if domain in _DOMAIN_KEYS:
        vector[domain_offset + _DOMAIN_KEYS.index(domain)] = 1.0

    time_offset = domain_offset + len(_DOMAIN_KEYS)
    vector[time_offset + hour_bucket] = 1.0
    return vector


def _default_state() -> dict[str, object]:
    return {
        "context_dim": _CONTEXT_DIM,
        "a": _identity_matrix(_CONTEXT_DIM),
        "b": _zero_vector(_CONTEXT_DIM),
        "pulls": 0,
        "total_reward": 0.0,
    }


def _load_state(outlet: str) -> dict[str, object]:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT context_dim, a_json, b_json, pulls, total_reward
        FROM bandit_state
        WHERE outlet = ?
        """,
        (outlet,),
    ).fetchone()
    if row is None:
        return _default_state()
    return {
        "context_dim": row[0],
        "a": json.loads(row[1]),
        "b": json.loads(row[2]),
        "pulls": row[3],
        "total_reward": row[4],
    }


def get_bandit_snapshot(outlet: str) -> dict[str, object]:
    state = _load_state(outlet)
    return {
        "outlet": outlet,
        "context_dim": state["context_dim"],
        "a": state["a"],
        "b": state["b"],
        "pulls": state["pulls"],
        "total_reward": state["total_reward"],
    }


def _save_state(outlet: str, state: dict[str, object]) -> None:
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO bandit_state
        (outlet, context_dim, a_json, b_json, pulls, total_reward, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(outlet) DO UPDATE SET
            context_dim = excluded.context_dim,
            a_json = excluded.a_json,
            b_json = excluded.b_json,
            pulls = excluded.pulls,
            total_reward = excluded.total_reward,
            updated_at = excluded.updated_at
        """,
        (
            outlet,
            state["context_dim"],
            json.dumps(state["a"]),
            json.dumps(state["b"]),
            state["pulls"],
            round(float(state["total_reward"]), 4),
            now,
        ),
    )
    conn.commit()


def score_feed(
    topic: str,
    feed: dict,
    when: datetime | str | None = None,
) -> dict[str, object]:
    state = _load_state(feed["outlet"])
    context = build_context_vector(topic, feed, when)
    theta = _solve_linear_system(state["a"], state["b"])
    confidence_direction = _solve_linear_system(state["a"], context)
    uncertainty = max(0.0, _dot(context, confidence_direction))
    score = _dot(theta, context) + BANDIT_ALPHA * math.sqrt(uncertainty)
    return {
        "feed": feed,
        "context": context,
        "score": round(score, 6),
        "pulls": int(state["pulls"]),
        "total_reward": float(state["total_reward"]),
    }


def select_feeds(
    topic: str,
    now: datetime | str | None = None,
    k: int = BANDIT_MAX_FEEDS_PER_CRAWL,
    feeds: Sequence[dict] | None = None,
) -> list[dict]:
    active_feeds = list(feeds or FEEDS)
    if not active_feeds:
        return []

    scored = [score_feed(topic, feed, now) for feed in active_feeds]
    scored.sort(
        key=lambda item: (
            item["pulls"] > 0,
            -float(item["score"]),
            int(item["pulls"]),
            item["feed"]["outlet"],
        )
    )
    limit = max(1, min(k, len(scored)))
    return [item["feed"] for item in scored[:limit]]


def update_feed_reward(
    outlet: str,
    topic: str,
    reward: float,
    when: datetime | str | None = None,
    feed: dict | None = None,
) -> float:
    arm = feed or _find_feed(outlet)
    context = build_context_vector(topic, arm, when)
    state = _load_state(outlet)
    clipped_reward = max(0.0, min(1.0, float(reward)))

    for row in range(_CONTEXT_DIM):
        for col in range(_CONTEXT_DIM):
            state["a"][row][col] += context[row] * context[col]
        state["b"][row] += clipped_reward * context[row]

    state["pulls"] += 1
    state["total_reward"] += clipped_reward
    _save_state(outlet, state)
    return round(clipped_reward, 4)


def reward_from_signals(topic: str, signals: dict) -> float:
    domain = str(signals.get("domain") or topic_to_domain(topic))
    weights = get_domain_config(domain)["signal_weights"]
    reward = sum(
        float(weights.get(key, 0.0)) * float(signals.get(key, 0.0))
        for key in DOMAIN_SIGNAL_KEYS
    )
    return round(max(0.0, min(1.0, reward)), 4)


def record_signal_reward(
    outlet: str,
    topic: str,
    signals: dict,
    when: datetime | str | None = None,
    feed: dict | None = None,
) -> float:
    reward = reward_from_signals(topic, signals)
    return update_feed_reward(outlet, topic, reward, when=when, feed=feed)


def record_crawl_miss(
    outlet: str,
    topic: str,
    when: datetime | str | None = None,
    feed: dict | None = None,
) -> float:
    return update_feed_reward(outlet, topic, 0.0, when=when, feed=feed)


def reset_bandit_state() -> None:
    conn = get_conn()
    conn.execute("DELETE FROM bandit_state")
    conn.commit()


def warm_start_from_history(topic: str = "", limit: int | None = None) -> int:
    conn = get_conn()
    query = """
        SELECT a.outlet, a.topic, a.published_at, s.raw_json,
               s.concern_level, s.purchase_intent, s.avoidance_signals
        FROM signals s
        JOIN articles a ON a.id = s.article_id
        WHERE (? = '' OR a.topic = ?)
        ORDER BY s.extracted_at ASC, s.article_id ASC
    """
    params: list[object] = [topic, topic]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    updated = 0
    for outlet, article_topic, published_at, raw_json, concern, purchase, avoidance in rows:
        if outlet not in {feed["outlet"] for feed in FEEDS}:
            continue
        signals = json.loads(raw_json) if raw_json else {}
        if not signals:
            signals = {
                "concern_level": concern or 0.0,
                "purchase_intent": purchase or 0.0,
                "avoidance_signals": avoidance or 0.0,
            }
        record_signal_reward(outlet, article_topic or topic, signals, when=published_at)
        updated += 1

    return updated
