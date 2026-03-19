from fastapi import APIRouter, HTTPException, Query

from api.models import (
    ClusterDriftStageResponse,
    ClusterRunResponse,
    ClusterSignalStageResponse,
    EmbeddingStageResponse,
    LatestClusterRunResponse,
)
from api.pipeline import run_collection_cycle
from brief.generator import (
    generate_hierarchical_brief_cached,
    get_brief_summary,
)
from brief.models import ResearchBrief
from clustering.clustering_service import get_latest_cluster_run, run_clustering
from delta.cluster_drift import run_cluster_drift
from extraction.cluster_extractor import run_cluster_extraction
from extraction.embedding_service import embed_pending_articles


router = APIRouter()


@router.post("/pipeline/run")
async def run_pipeline(
    topic: str = Query(..., min_length=1),
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object]:
    collect_result = await run_collection_cycle(topic, country=country, source=source)
    brief_summary = get_brief_summary(
        topic,
        country=country,
        source=source,
        language=language,
    )
    confidence_context = brief_summary.get("confidence_context")
    return {
        **collect_result,
        "brief_topic": brief_summary["topic"],
        "brief_status": brief_summary["status"],
        "brief_alert_level": brief_summary["alert_level"],
        "brief_confidence": (
            confidence_context.segment_confidence
            if confidence_context is not None
            else {}
        ),
    }


@router.post("/pipeline/embed", response_model=EmbeddingStageResponse)
def run_embedding_stage(
    topic: str | None = None,
    country: str | None = None,
    source: str | None = None,
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict[str, object]:
    return embed_pending_articles(
        limit=limit,
        topic=topic,
        country=country,
        source=source,
    )


@router.post("/pipeline/cluster", response_model=ClusterRunResponse)
def run_cluster_stage(
    topic: str = Query(..., min_length=1),
    country: str = "",
    source: str = "",
    language: str | None = None,
    window_hours: int = Query(default=24, ge=1),
    min_cluster_size: int = Query(default=3, ge=1),
) -> dict[str, object]:
    try:
        return run_clustering(
            topic=topic,
            country=country,
            source=source,
            language=language,
            window_hours=window_hours,
            min_cluster_size=min_cluster_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pipeline/cluster/signals", response_model=ClusterSignalStageResponse)
def run_cluster_signal_stage(
    run_id: str = Query(..., min_length=1),
    overwrite: bool = False,
    min_cluster_size: int = 3,
) -> dict[str, object]:
    try:
        return run_cluster_extraction(
            run_id=run_id,
            overwrite=overwrite,
            min_cluster_size=min_cluster_size,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pipeline/cluster/drift", response_model=ClusterDriftStageResponse)
def run_cluster_drift_stage(run_id: str = Query(..., min_length=1)) -> dict[str, object]:
    try:
        return run_cluster_drift(run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/pipeline/brief/hierarchical", response_model=ResearchBrief)
def run_hierarchical_brief_stage(
    topic: str | None = Query(default=None, min_length=1),
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = Query(default=None, min_length=1),
) -> ResearchBrief:
    if run_id is None and topic is None:
        raise HTTPException(
            status_code=400,
            detail="topic is required when run_id is not provided.",
        )
    try:
        return generate_hierarchical_brief_cached(
            topic=topic or "",
            country=country,
            source=source,
            language=language,
            run_id=run_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/pipeline/clusters/latest", response_model=LatestClusterRunResponse)
def get_latest_clusters(
    topic: str = Query(..., min_length=1),
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object]:
    result = get_latest_cluster_run(
        topic=topic,
        country=country,
        source=source,
        language=language,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No cluster run found for the requested scope.",
        )
    return result
