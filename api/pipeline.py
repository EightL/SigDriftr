from __future__ import annotations

import asyncio
import json
import threading
import time

from api.common import (
    normalize_country,
    normalize_language,
    normalize_source,
    pipeline_stage,
    utc_now_iso,
)
from brief.generator import generate_brief_cached, generate_hierarchical_brief_cached
from config.settings import BANDIT_REWARD_MODE, COLLECTION_MODE
from db.topic_queries import topic_filter_sql
from db.init import get_conn
from db.topic_resolver import resolve_topic
from ingestion.bandit import record_signal_reward
from ingestion.crawler import _crawl_async_report


PipelineScopeKey = tuple[str, str, str, str | None]
_pipeline_run_cache: dict[PipelineScopeKey, dict[str, object]] = {}
_pipeline_run_lock = threading.Lock()


def _scope_cache_key(
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> PipelineScopeKey:
    normalized_topic = topic.strip()
    if normalized_topic:
        normalized_topic = resolve_topic(normalized_topic).canonical_topic_id
    return (
        normalized_topic,
        normalize_country(country),
        normalize_source(source),
        normalize_language(language),
    )


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
    normalized_country = normalize_country(country)
    normalized_source = normalize_source(source)
    normalized_language = normalize_language(language)
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
        canonical_topic_id = resolve_topic(topic).canonical_topic_id
        rows = conn.execute(
            """
            SELECT a.outlet, COALESCE(MAX(at.canonical_topic_id), MAX(at.topic)), a.published_at,
                   s.concern_level, s.purchase_intent, s.avoidance_signals, s.raw_json
            FROM signals s
            JOIN articles a ON a.id = s.article_id
            JOIN article_topics at ON at.article_id = a.id
            WHERE (
                    at.canonical_topic_id = ?
                    OR at.topic = ?
                  )
              AND at.matched_at >= ?
            GROUP BY s.article_id
            ORDER BY MIN(at.matched_at) ASC, s.article_id ASC
            """,
            (canonical_topic_id, topic, crawl_start),
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
    *,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> dict[str, object]:
    crawl_start = utc_now_iso()
    resolved_reward_mode = (reward_mode or BANDIT_REWARD_MODE).strip().lower() or "yield"
    report = await _crawl_async_report(
        topic,
        country=country,
        source=source,
        collection_mode=collection_mode or COLLECTION_MODE,
        reward_mode=resolved_reward_mode,
    )
    canonical_topic_id = getattr(
        report,
        "canonical_topic_id",
        resolve_topic(topic).canonical_topic_id if topic else "",
    )
    inserted = report.inserted
    processed = 0
    rewards_recorded = 0

    if inserted > 0:
        from extraction.extractor import run_extraction

        processed = run_extraction(topic, record_bandit_reward=False)
        if resolved_reward_mode == "signal":
            rewards_recorded = record_recent_signal_rewards(topic, crawl_start)
        if processed > 0:
            from brief.generator import clear_brief_cache

            clear_brief_cache()

    return {
        "inserted": inserted,
        "extracted": processed,
        "rewards_recorded": rewards_recorded,
        "topic": topic,
        "canonical_topic_id": canonical_topic_id,
        "country": country,
        "source": source,
        "collection_mode": report.collection_mode,
        "reward_mode": report.reward_mode,
        "run_id": report.run_id,
        "eligible_feeds": report.eligible_feeds,
        "selected_feeds": report.selected_feeds,
        "accepted": report.accepted,
        "duplicates": report.duplicates,
        "feed_stats": [item for item in report.to_dict()["feed_stats"]],
    }


def run_collection_cycle_sync(
    topic: str,
    country: str = "",
    source: str = "",
    *,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> dict[str, object]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_collection_cycle(
                topic,
                country=country,
                source=source,
                collection_mode=collection_mode,
                reward_mode=reward_mode,
            )
        )

    raise RuntimeError(
        "run_collection_cycle_sync() cannot run inside an active event loop; "
        "await run_collection_cycle(topic, country=..., source=...) instead."
    )


def _run_embedding_stage(
    topic: str,
    *,
    country: str,
    source: str,
) -> tuple[dict[str, object], float]:
    from extraction.embedding_service import embed_pending_articles

    started = time.perf_counter()
    result = embed_pending_articles(
        limit=5000,
        topic=topic,
        country=country,
        source=source,
    )
    return result, round(time.perf_counter() - started, 4)


def _cluster_fallback_result(
    *,
    topic: str,
    country: str,
    source: str,
    language: str | None,
    status: str,
    min_cluster_size: int,
    n_articles: int = 0,
) -> dict[str, object]:
    return {
        "run_id": None,
        "topic": topic,
        "country": country,
        "source": source,
        "language": language,
        "window_start": None,
        "window_end": None,
        "status": status,
        "n_articles": n_articles,
        "n_clusters": 0,
        "n_noise": 0,
        "model_name": None,
        "model_version": None,
        "umap_n_components": 0,
        "umap_n_neighbors": 0,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples": 0,
        "duration_s": 0.0,
    }


def _run_cluster_stage(
    topic: str,
    *,
    country: str,
    source: str,
    language: str | None,
    window_hours: int,
    min_cluster_size: int,
) -> tuple[dict[str, object], float, str | None]:
    from clustering.clustering_service import run_clustering

    started = time.perf_counter()
    error_detail: str | None = None
    try:
        result = run_clustering(
            topic=topic,
            country=country,
            source=source,
            language=language,
            window_hours=window_hours,
            min_cluster_size=min_cluster_size,
        )
    except ModuleNotFoundError as exc:
        error_detail = (
            f"Clustering dependencies are unavailable in this environment ({exc.name}). "
            "Install stage-3 dependencies and rerun to enable storyline grouping."
        )
        result = _cluster_fallback_result(
            topic=topic,
            country=country,
            source=source,
            language=language,
            status="dependency_missing",
            min_cluster_size=min_cluster_size,
        )
    except Exception as exc:
        error_detail = f"Clustering failed: {exc}"
        result = _cluster_fallback_result(
            topic=topic,
            country=country,
            source=source,
            language=language,
            status="failed",
            min_cluster_size=min_cluster_size,
        )
    return result, round(time.perf_counter() - started, 4), error_detail


def _cluster_produced_groups(cluster_result: dict[str, object]) -> bool:
    return (
        str(cluster_result.get("status", "")) == "completed"
        and int(cluster_result.get("n_clusters", 0) or 0) > 0
    )


def _run_cluster_signal_stage(
    cluster_result: dict[str, object],
    *,
    min_cluster_size: int,
) -> tuple[dict[str, object], str | None, int]:
    from extraction.cluster_extractor import run_cluster_extraction

    started = time.perf_counter()
    try:
        result = run_cluster_extraction(
            run_id=str(cluster_result["run_id"]),
            overwrite=False,
            min_cluster_size=min_cluster_size,
        )
        result["duration_s"] = round(time.perf_counter() - started, 4)
        completed = int(result.get("processed", 0) or 0) + int(
            result.get("skipped_existing", 0) or 0
        )
        return result, None, completed
    except Exception as exc:
        return (
            {
                "run_id": str(cluster_result["run_id"]),
                "selected_clusters": int(cluster_result.get("n_clusters", 0) or 0),
                "processed": 0,
                "skipped_existing": 0,
                "failed": int(cluster_result.get("n_clusters", 0) or 0),
                "provider": "",
                "model_name": "",
                "duration_s": round(time.perf_counter() - started, 4),
            },
            f"Cluster signal extraction failed: {exc}",
            0,
        )


def _run_cluster_drift_stage(
    cluster_result: dict[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    from delta.cluster_drift import run_cluster_drift

    started = time.perf_counter()
    try:
        result = run_cluster_drift(str(cluster_result["run_id"]))
        result["duration_s"] = round(time.perf_counter() - started, 4)
        return result, None
    except Exception as exc:
        return None, f"Cluster drift failed: {exc}"


def _run_cluster_signal_and_drift(
    cluster_result: dict[str, object],
    *,
    min_cluster_size: int,
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    str | None,
    str | None,
]:
    if not _cluster_produced_groups(cluster_result):
        return None, None, None, None

    signal_result, signal_error, completed_signals = _run_cluster_signal_stage(
        cluster_result,
        min_cluster_size=min_cluster_size,
    )
    required_signals = int(cluster_result.get("n_clusters", 0) or 0)
    if (
        signal_error is None
        and int(signal_result.get("failed", 0) or 0) == 0
        and completed_signals >= required_signals
    ):
        drift_result, drift_error = _run_cluster_drift_stage(cluster_result)
    else:
        drift_result, drift_error = None, None

    return signal_result, drift_result, signal_error, drift_error


def _generate_pipeline_brief(
    topic: str,
    *,
    country: str,
    source: str,
    language: str | None,
    cluster_result: dict[str, object],
    cluster_drift_result: dict[str, object] | None,
):
    if cluster_drift_result is not None:
        return generate_hierarchical_brief_cached(
            topic=topic,
            country=country,
            source=source,
            language=language,
            run_id=str(cluster_result["run_id"]),
        )
    return generate_brief_cached(
        topic,
        country=country,
        source=source,
        language=language,
        prefer_cluster=False,
        require_cluster=False,
    )


def _build_cluster_signal_pipeline_stage(
    *,
    cluster_result: dict[str, object],
    cluster_error_detail: str | None,
    cluster_signal_result: dict[str, object] | None,
    cluster_signal_error_detail: str | None,
) -> dict[str, object]:
    if cluster_signal_result is None:
        return pipeline_stage(
            "skipped",
            detail=cluster_error_detail
            or "Cluster signals were skipped because clustering did not produce stable groups.",
            count=0,
        )

    failed = int(cluster_signal_result.get("failed", 0) or 0)
    processed = int(cluster_signal_result.get("processed", 0) or 0)
    return pipeline_stage(
        "completed" if failed == 0 else ("partial" if processed > 0 else "failed"),
        detail=(
            "Cluster signal extraction completed."
            if failed == 0
            else (cluster_signal_error_detail or "Some clusters failed signal extraction.")
        ),
        count=processed,
        duration_s=float(cluster_signal_result.get("duration_s", 0.0) or 0.0),
        metadata=dict(cluster_signal_result),
    )


def _build_cluster_drift_pipeline_stage(
    *,
    cluster_drift_result: dict[str, object] | None,
    cluster_drift_error_detail: str | None,
) -> dict[str, object]:
    if cluster_drift_result is None:
        return pipeline_stage(
            "skipped",
            detail=cluster_drift_error_detail
            or "Cluster drift was unavailable for the current run.",
            count=0,
        )

    return pipeline_stage(
        "completed",
        detail="Cluster drift computed for the latest run.",
        count=int(cluster_drift_result.get("observed_clusters", 0) or 0),
        duration_s=float(cluster_drift_result.get("duration_s", 0.0) or 0.0),
        metadata=dict(cluster_drift_result),
    )


def _build_embedding_pipeline_stage(
    embed_result: dict[str, object],
    *,
    duration_s: float,
) -> dict[str, object]:
    failed = int(embed_result.get("failed", 0) or 0)
    embedded = int(embed_result.get("embedded", 0) or 0)
    return pipeline_stage(
        "completed" if failed == 0 else ("partial" if embedded > 0 else "failed"),
        detail="Embedding stage finished.",
        count=embedded,
        duration_s=duration_s,
        metadata=dict(embed_result),
    )


def _build_full_pipeline_summary(
    *,
    topic: str,
    topic_resolution,
    country: str,
    source: str,
    language: str | None,
    collection_result: dict[str, object],
    collection_duration: float,
    embed_result: dict[str, object],
    embed_duration: float,
    cluster_result: dict[str, object],
    cluster_duration: float,
    cluster_error_detail: str | None,
    cluster_signal_result: dict[str, object] | None,
    cluster_signal_error_detail: str | None,
    cluster_drift_result: dict[str, object] | None,
    cluster_drift_error_detail: str | None,
    brief,
    counts: dict[str, int],
    started_at: float,
) -> dict[str, object]:
    cluster_status = str(cluster_result.get("status", "not_run"))
    cluster_stage_status = "completed" if cluster_status == "completed" else cluster_status
    cluster_signal_stage = _build_cluster_signal_pipeline_stage(
        cluster_result=cluster_result,
        cluster_error_detail=cluster_error_detail,
        cluster_signal_result=cluster_signal_result,
        cluster_signal_error_detail=cluster_signal_error_detail,
    )
    cluster_drift_stage = _build_cluster_drift_pipeline_stage(
        cluster_drift_result=cluster_drift_result,
        cluster_drift_error_detail=cluster_drift_error_detail,
    )

    return {
        "scope": {
            "topic": topic,
            "requested_topic": topic,
            "canonical_topic_id": (
                topic_resolution.canonical_topic_id if topic_resolution else ""
            ),
            "canonical_display_name": (
                topic_resolution.display_name if topic_resolution else None
            ),
            "country": country,
            "source": source,
            "language": language,
            "collection_mode": str(collection_result.get("collection_mode", "")),
            "reward_mode": str(collection_result.get("reward_mode", "")),
        },
        "run_id": cluster_result.get("run_id"),
        "generated_at": utc_now_iso(),
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
                "collect": pipeline_stage(
                    "completed",
                    detail="Collection cycle finished.",
                    count=int(collection_result.get("inserted", 0) or 0),
                    duration_s=collection_duration,
                    metadata=dict(collection_result),
                ),
                "extract": pipeline_stage(
                    "completed",
                    detail="Article-level extraction finished.",
                    count=int(collection_result.get("extracted", 0) or 0),
                    metadata={
                        "rewards_recorded": int(
                            collection_result.get("rewards_recorded", 0) or 0
                        ),
                    },
                ),
                "embed": _build_embedding_pipeline_stage(
                    embed_result,
                    duration_s=embed_duration,
                ),
                "cluster": pipeline_stage(
                    cluster_stage_status,
                    detail=cluster_error_detail
                    or "Cluster run finished for the requested scope.",
                    count=int(cluster_result.get("n_clusters", 0) or 0),
                    duration_s=cluster_duration,
                    metadata=dict(cluster_result),
                ),
                "cluster_signals": cluster_signal_stage,
                "cluster_drift": cluster_drift_stage,
                "brief": pipeline_stage(
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


async def run_full_pipeline(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    window_hours: int = 24,
    min_cluster_size: int = 3,
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    normalized_country = normalize_country(country)
    normalized_source = normalize_source(source)
    normalized_language = normalize_language(language)
    topic_resolution = resolve_topic(topic) if topic.strip() else None

    collect_started = time.perf_counter()
    collect_result = await run_collection_cycle(
        topic,
        country=normalized_country,
        source=normalized_source,
        collection_mode=collection_mode,
        reward_mode=reward_mode,
    )
    collect_duration = round(time.perf_counter() - collect_started, 4)

    embed_result, embed_duration = _run_embedding_stage(
        topic,
        country=normalized_country,
        source=normalized_source,
    )
    cluster_result, cluster_duration, cluster_error_detail = _run_cluster_stage(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        window_hours=window_hours,
        min_cluster_size=min_cluster_size,
    )
    (
        cluster_signal_result,
        cluster_drift_result,
        cluster_signal_error_detail,
        cluster_drift_error_detail,
    ) = _run_cluster_signal_and_drift(
        cluster_result,
        min_cluster_size=min_cluster_size,
    )

    brief = _generate_pipeline_brief(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        cluster_result=cluster_result,
        cluster_drift_result=cluster_drift_result,
    )
    counts = get_scope_counts(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
    )
    summary = _build_full_pipeline_summary(
        topic=topic,
        topic_resolution=topic_resolution,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        collection_result=collect_result,
        collection_duration=collect_duration,
        embed_result=embed_result,
        embed_duration=embed_duration,
        cluster_result=cluster_result,
        cluster_duration=cluster_duration,
        cluster_error_detail=cluster_error_detail,
        cluster_signal_result=cluster_signal_result,
        cluster_signal_error_detail=cluster_signal_error_detail,
        cluster_drift_result=cluster_drift_result,
        cluster_drift_error_detail=cluster_drift_error_detail,
        brief=brief,
        counts=counts,
        started_at=started_at,
    )
    _store_pipeline_run_summary(summary)
    return summary
