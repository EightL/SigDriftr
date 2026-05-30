from __future__ import annotations

from datetime import datetime, timezone

from brief.generator import (
    enrich_cluster_observations,
    generate_brief_cached,
    generate_hierarchical_brief_cached,
    get_brief_support,
)
from brief.digest import generate_digest
from clustering.clustering_service import get_latest_cluster_run
from delta.cluster_drift import get_cluster_drift
from db.topic_resolver import resolve_topic

from api.pipeline import get_cached_pipeline_run_summary, get_scope_counts
from api.routes.signals import query_signals


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_country(value: str | None) -> str:
    return (value or "").strip().upper()


def _normalize_source(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_language(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized or None


def _model_dump(model: object) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)
    return dict(model)


def _pipeline_stage(
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


def _fallback_pipeline_summary(
    *,
    topic: str,
    country: str,
    source: str,
    language: str | None,
    latest_cluster_run: dict[str, object] | None,
    cluster_drift_bundle: dict[str, object],
    brief: object,
) -> dict[str, object]:
    counts = get_scope_counts(
        topic,
        country=country,
        source=source,
        language=language,
    )
    cluster_count = int((latest_cluster_run or {}).get("n_clusters", 0) or 0)
    noise_count = int((latest_cluster_run or {}).get("n_noise", 0) or 0)
    cluster_signal_count = sum(
        1 for cluster in (latest_cluster_run or {}).get("clusters", []) if cluster.get("signal")
    )
    cluster_status = str((latest_cluster_run or {}).get("status", "not_run"))
    if latest_cluster_run is None:
        cluster_stage = _pipeline_stage(
            "pending",
            detail="No scoped cluster run exists yet.",
            count=0,
        )
        cluster_signal_stage = _pipeline_stage(
            "pending",
            detail="Cluster signals are unavailable until clustering runs.",
            count=0,
        )
    else:
        cluster_stage = _pipeline_stage(
            "completed" if cluster_status == "completed" else cluster_status,
            detail="Loaded the latest scoped cluster run.",
            count=cluster_count,
            metadata={
                "run_id": latest_cluster_run.get("run_id"),
                "created_at": latest_cluster_run.get("created_at"),
            },
        )
        cluster_signal_stage = _pipeline_stage(
            "completed"
            if cluster_count > 0 and cluster_signal_count == cluster_count
            else ("partial" if cluster_signal_count > 0 else "skipped"),
            detail="Loaded cached cluster signal output for the latest run."
            if cluster_signal_count > 0
            else "Cluster signal output is unavailable for the latest run.",
            count=cluster_signal_count,
        )

    return {
        **counts,
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "cluster_status": cluster_status,
        "brief_status": getattr(brief, "status", "insufficient_data"),
        "strongest_segment": getattr(brief, "most_affected_segment", None),
        "stages": {
            "collect": _pipeline_stage(
                "completed" if counts["article_count"] > 0 else "pending",
                detail="Scoped articles are available."
                if counts["article_count"] > 0
                else "No scoped articles are stored yet.",
                count=counts["article_count"],
            ),
            "extract": _pipeline_stage(
                "completed" if counts["signal_count"] > 0 else "pending",
                detail="Scoped article signals are available."
                if counts["signal_count"] > 0
                else "No scoped article signals are stored yet.",
                count=counts["signal_count"],
            ),
            "embed": _pipeline_stage(
                "completed" if counts["embedding_count"] > 0 else "pending",
                detail="Scoped embeddings are available."
                if counts["embedding_count"] > 0
                else "No scoped embeddings are stored yet.",
                count=counts["embedding_count"],
            ),
            "cluster": cluster_stage,
            "cluster_signals": cluster_signal_stage,
            "cluster_drift": _pipeline_stage(
                str(cluster_drift_bundle.get("status", "pending")),
                detail=str(cluster_drift_bundle.get("message", "")),
                count=len(((cluster_drift_bundle.get("data") or {}).get("clusters") or [])),
            ),
            "brief": _pipeline_stage(
                getattr(brief, "status", "insufficient_data"),
                detail=f"Resolved brief via {getattr(brief, 'generation_mode', 'unknown')} mode.",
                count=len(getattr(brief, "hypotheses", [])),
            ),
        },
    }


def build_output_bundle(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object]:
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    topic_resolution = resolve_topic(topic) if topic.strip() else None

    latest_cluster_run = None
    if topic.strip():
        latest_cluster_run = get_latest_cluster_run(
            topic=topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
        )

    run_id = str((latest_cluster_run or {}).get("run_id") or "") or None
    cluster_drift_payload = None
    cluster_drift_bundle: dict[str, object]
    if run_id is not None:
        try:
            cluster_drift_payload = get_cluster_drift(run_id)
            cluster_drift_payload["clusters"] = enrich_cluster_observations(
                run_id,
                list(cluster_drift_payload.get("clusters", [])),
            )
            cluster_drift_bundle = {
                "status": "ready",
                "message": "Loaded cluster drift for the latest scoped run.",
                "data": cluster_drift_payload,
            }
        except LookupError:
            cluster_drift_bundle = {
                "status": "pending",
                "message": "The latest cluster run exists, but cluster drift has not been computed yet.",
                "data": None,
            }
    else:
        cluster_drift_bundle = {
            "status": "missing",
            "message": "No cluster run exists for the requested scope yet.",
            "data": None,
        }

    if cluster_drift_payload is not None and run_id is not None:
        brief = generate_hierarchical_brief_cached(
            topic=topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            run_id=run_id,
        )
        brief_support = get_brief_support(
            topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            run_id=run_id,
            prefer_cluster=True,
            require_cluster=True,
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
        brief_support = get_brief_support(
            topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            prefer_cluster=False,
            require_cluster=False,
        )

    cached_pipeline = get_cached_pipeline_run_summary(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
    )
    pipeline = (
        dict(cached_pipeline["pipeline"])
        if cached_pipeline is not None and "pipeline" in cached_pipeline
        else _fallback_pipeline_summary(
            topic=topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
            latest_cluster_run=latest_cluster_run,
            cluster_drift_bundle=cluster_drift_bundle,
            brief=brief,
        )
    )

    digest = generate_digest(
        topic,
        country=normalized_country,
        source=normalized_source,
        limit=8,
    )
    signals = query_signals(
        topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
    )[:20]

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
            "country": normalized_country,
            "source": normalized_source,
            "language": normalized_language,
        },
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "pipeline": pipeline,
        "brief": _model_dump(brief),
        "digest": _model_dump(digest),
        "cluster_drift": cluster_drift_bundle,
        "clusters": list((latest_cluster_run or {}).get("clusters", [])),
        "brief_support": brief_support,
        "signals": signals,
    }
