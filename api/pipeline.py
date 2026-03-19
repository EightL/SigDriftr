from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone

from brief.generator import generate_brief_cached, generate_hierarchical_brief_cached
from db.topic_queries import topic_filter_sql
from db.init import get_conn
from ingestion.bandit import record_signal_reward
from ingestion.crawler import _crawl_async


PipelineScopeKey = tuple[str, str, str, str | None]
_pipeline_run_cache: dict[PipelineScopeKey, dict[str, object]] = {}
_pipeline_run_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_country(value: str | None) -> str:
    return (value or "").strip().upper()


def _normalize_source(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_language(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized or None


def _scope_cache_key(
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> PipelineScopeKey:
    return (
        topic.strip(),
        _normalize_country(country),
        _normalize_source(source),
        _normalize_language(language),
    )


def _model_dump(model: object) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)
    return dict(model)


def _stage_status(
    status: str,
    *,
    detail: str | None = None,
    count: int | None = None,
    duration_s: float | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "detail": detail,
        "count": count,
        "duration_s": duration_s,
        "metadata": metadata or {},
    }


def _store_pipeline_run_summary(summary: dict[str, object]) -> None:
    scope = summary.get("scope", {})
    key = _scope_cache_key(
        str(scope.get("topic", "")),
        str(scope.get("country", "")),
        str(scope.get("source", "")),
        scope.get("language"),
    )
    with _pipeline_run_lock:
        _pipeline_run_cache[key] = dict(summary)


def get_cached_pipeline_run_summary(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object] | None:
    key = _scope_cache_key(topic, country=country, source=source, language=language)
    with _pipeline_run_lock:
        cached = _pipeline_run_cache.get(key)
    return dict(cached) if cached is not None else None


def get_scope_counts(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, int]:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic)
    filters = topic_sql
    params: list[object] = [*topic_params]
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    if normalized_country:
        filters += " AND a.country = ?"
        params.append(normalized_country)
    if normalized_source:
        filters += " AND LOWER(a.outlet) = ?"
        params.append(normalized_source)
    if normalized_language is not None:
        filters += " AND LOWER(a.language) = ?"
        params.append(normalized_language)

    article_count = int(
        conn.execute(
            f"""
            SELECT COUNT(DISTINCT a.id)
            FROM articles a
            WHERE 1 = 1
              {filters}
            """,
            params,
        ).fetchone()[0]
        or 0
    )
    signal_count = int(
        conn.execute(
            f"""
            SELECT COUNT(DISTINCT s.article_id)
            FROM signals s
            JOIN articles a ON a.id = s.article_id
            WHERE 1 = 1
              {filters}
            """,
            params,
        ).fetchone()[0]
        or 0
    )
    embedding_count = int(
        conn.execute(
            f"""
            SELECT COUNT(DISTINCT e.article_id)
            FROM article_embeddings e
            JOIN articles a ON a.id = e.article_id
            WHERE e.status = 'complete'
              AND e.model_name = (
                    SELECT model_name
                    FROM article_embeddings latest
                    WHERE latest.article_id = e.article_id
                    ORDER BY COALESCE(latest.embedded_at, latest.updated_at, latest.created_at) DESC,
                             latest.id DESC
                    LIMIT 1
              )
              {filters}
            """,
            params,
        ).fetchone()[0]
        or 0
    )
    return {
        "article_count": article_count,
        "signal_count": signal_count,
        "embedding_count": embedding_count,
    }


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
    if topic:
        rows = conn.execute(
            """
            SELECT a.outlet, at.topic, a.published_at,
                   s.concern_level, s.purchase_intent, s.avoidance_signals, s.raw_json
            FROM signals s
            JOIN articles a ON a.id = s.article_id
            JOIN article_topics at ON at.article_id = a.id
            WHERE at.topic = ?
              AND at.matched_at >= ?
            ORDER BY at.matched_at ASC, s.article_id ASC
            """,
            (topic, crawl_start),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT a.outlet, COALESCE(a.topic, ''), a.published_at,
                   s.concern_level, s.purchase_intent, s.avoidance_signals, s.raw_json
            FROM signals s
            JOIN articles a ON a.id = s.article_id
            WHERE a.fetched_at >= ?
            ORDER BY s.extracted_at ASC, s.article_id ASC
            """,
            (crawl_start,),
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


async def run_collection_cycle(
    topic: str,
    country: str = "",
    source: str = "",
) -> dict[str, int | str]:
    crawl_start = datetime.now(timezone.utc).isoformat()
    inserted = await _crawl_async(topic, country=country, source=source)
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
        "country": country,
        "source": source,
    }


def run_collection_cycle_sync(
    topic: str,
    country: str = "",
    source: str = "",
) -> dict[str, int | str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_collection_cycle(topic, country=country, source=source))

    raise RuntimeError(
        "run_collection_cycle_sync() cannot run inside an active event loop; "
        "await run_collection_cycle(topic, country=..., source=...) instead."
    )


async def run_full_pipeline(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    window_hours: int = 24,
    min_cluster_size: int = 3,
) -> dict[str, object]:
    from clustering.clustering_service import run_clustering
    from delta.cluster_drift import run_cluster_drift
    from extraction.cluster_extractor import run_cluster_extraction
    from extraction.embedding_service import embed_pending_articles

    started_at = time.perf_counter()
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)

    collect_started = time.perf_counter()
    collect_result = await run_collection_cycle(
        topic,
        country=normalized_country,
        source=normalized_source,
    )
    collect_duration = round(time.perf_counter() - collect_started, 4)

    embed_started = time.perf_counter()
    embed_result = embed_pending_articles(
        limit=5000,
        topic=topic,
        country=normalized_country,
        source=normalized_source,
    )
    embed_duration = round(time.perf_counter() - embed_started, 4)

    cluster_started = time.perf_counter()
    cluster_result = run_clustering(
        topic=topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        window_hours=window_hours,
        min_cluster_size=min_cluster_size,
    )
    cluster_duration = round(time.perf_counter() - cluster_started, 4)

    cluster_signal_result: dict[str, object] | None = None
    cluster_drift_result: dict[str, object] | None = None

    if (
        str(cluster_result.get("status", "")) == "completed"
        and int(cluster_result.get("n_clusters", 0) or 0) > 0
    ):
        signal_started = time.perf_counter()
        cluster_signal_result = run_cluster_extraction(
            run_id=str(cluster_result["run_id"]),
            overwrite=False,
            min_cluster_size=min_cluster_size,
        )
        cluster_signal_result["duration_s"] = round(
            time.perf_counter() - signal_started,
            4,
        )
        completed_cluster_signals = int(cluster_signal_result.get("processed", 0) or 0) + int(
            cluster_signal_result.get("skipped_existing", 0) or 0
        )
        if (
            int(cluster_signal_result.get("failed", 0) or 0) == 0
            and completed_cluster_signals >= int(cluster_result.get("n_clusters", 0) or 0)
        ):
            drift_started = time.perf_counter()
            cluster_drift_result = run_cluster_drift(str(cluster_result["run_id"]))
            cluster_drift_result["duration_s"] = round(
                time.perf_counter() - drift_started,
                4,
            )

    if cluster_drift_result is not None:
        brief = generate_hierarchical_brief_cached(
            topic=topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            run_id=str(cluster_result["run_id"]),
        )
    else:
        brief = generate_brief_cached(
            topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            prefer_cluster=False,
            require_cluster=False,
        )

    counts = get_scope_counts(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
    )

    cluster_status = str(cluster_result.get("status", "not_run"))
    cluster_stage_status = (
        "completed"
        if cluster_status == "completed"
        else cluster_status
    )
    if cluster_signal_result is None:
        cluster_signal_stage = _stage_status(
            "skipped",
            detail="Cluster signals were skipped because clustering did not produce stable groups.",
            count=0,
        )
    else:
        failed = int(cluster_signal_result.get("failed", 0) or 0)
        processed = int(cluster_signal_result.get("processed", 0) or 0)
        cluster_signal_stage = _stage_status(
            "completed" if failed == 0 else ("partial" if processed > 0 else "failed"),
            detail="Cluster signal extraction completed." if failed == 0 else "Some clusters failed signal extraction.",
            count=processed,
            duration_s=float(cluster_signal_result.get("duration_s", 0.0) or 0.0),
            metadata=dict(cluster_signal_result),
        )
    if cluster_drift_result is None:
        cluster_drift_stage = _stage_status(
            "skipped",
            detail="Cluster drift was unavailable for the current run.",
            count=0,
        )
    else:
        cluster_drift_stage = _stage_status(
            "completed",
            detail="Cluster drift computed for the latest run.",
            count=int(cluster_drift_result.get("observed_clusters", 0) or 0),
            duration_s=float(cluster_drift_result.get("duration_s", 0.0) or 0.0),
            metadata=dict(cluster_drift_result),
        )

    brief_payload = _model_dump(brief)
    summary = {
        "scope": {
            "topic": topic,
            "country": normalized_country,
            "source": normalized_source,
            "language": normalized_language,
        },
        "run_id": cluster_result.get("run_id"),
        "generated_at": _utc_now_iso(),
        "duration_s": round(time.perf_counter() - started_at, 4),
        "cluster_status": cluster_status,
        "brief_status": str(brief.status),
        "pipeline": {
            **counts,
            "cluster_count": int(cluster_result.get("n_clusters", 0) or 0),
            "noise_count": int(cluster_result.get("n_noise", 0) or 0),
            "cluster_status": cluster_status,
            "brief_status": str(brief.status),
            "strongest_segment": brief.most_affected_segment,
            "stages": {
                "collect": _stage_status(
                    "completed",
                    detail="Collection cycle finished.",
                    count=int(collect_result.get("inserted", 0) or 0),
                    duration_s=collect_duration,
                    metadata=dict(collect_result),
                ),
                "extract": _stage_status(
                    "completed",
                    detail="Article-level extraction finished.",
                    count=int(collect_result.get("extracted", 0) or 0),
                    metadata={
                        "rewards_recorded": int(collect_result.get("rewards_recorded", 0) or 0),
                    },
                ),
                "embed": _stage_status(
                    "completed"
                    if int(embed_result.get("failed", 0) or 0) == 0
                    else (
                        "partial"
                        if int(embed_result.get("embedded", 0) or 0) > 0
                        else "failed"
                    ),
                    detail="Embedding stage finished.",
                    count=int(embed_result.get("embedded", 0) or 0),
                    duration_s=embed_duration,
                    metadata=dict(embed_result),
                ),
                "cluster": _stage_status(
                    cluster_stage_status,
                    detail="Cluster run finished for the requested scope.",
                    count=int(cluster_result.get("n_clusters", 0) or 0),
                    duration_s=cluster_duration,
                    metadata=dict(cluster_result),
                ),
                "cluster_signals": cluster_signal_stage,
                "cluster_drift": cluster_drift_stage,
                "brief": _stage_status(
                    str(brief.status),
                    detail=f"Brief generated via {brief.generation_mode or 'unknown'} mode.",
                    count=len(brief.hypotheses),
                    metadata={
                        "alert_level": brief.alert_level,
                        "generation_mode": brief.generation_mode,
                    },
                ),
            },
        },
        "brief": {
            "topic": brief.topic,
            "status": brief.status,
            "headline": brief.headline,
            "alert_level": brief.alert_level,
            "most_affected_segment": brief.most_affected_segment,
            "generation_mode": brief.generation_mode,
        },
    }
    _store_pipeline_run_summary(summary)
    return summary
