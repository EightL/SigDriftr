from fastapi import APIRouter, HTTPException

from api.models import (
    ClusterRunResponse,
    EmbeddingStageResponse,
    LatestClusterRunResponse,
)
from api.pipeline import run_collection_cycle
from brief.generator import generate_brief_cached
from brief.models import ResearchBrief
from clustering.clustering_service import get_latest_cluster_run, run_clustering
from extraction.embedding_service import embed_pending_articles


router = APIRouter()


@router.post("/pipeline/run")
async def run_pipeline(
    topic: str,
    country: str = "",
    source: str = "",
) -> dict[str, object]:
    collect_result = await run_collection_cycle(topic, country=country, source=source)
    brief: ResearchBrief = generate_brief_cached(topic)
    return {
        **collect_result,
        "brief_topic": brief.topic,
        "brief_status": brief.status,
        "brief_alert_level": brief.alert_level,
        "brief_confidence": (
            brief.confidence_context.segment_confidence
            if brief.confidence_context is not None
            else {}
        ),
    }


@router.post("/pipeline/embed", response_model=EmbeddingStageResponse)
def run_embedding_stage(
    topic: str | None = None,
    country: str | None = None,
    source: str | None = None,
    limit: int = 200,
) -> dict[str, object]:
    return embed_pending_articles(
        limit=limit,
        topic=topic,
        country=country,
        source=source,
    )


@router.post("/pipeline/cluster", response_model=ClusterRunResponse)
def run_cluster_stage(
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = None,
    window_hours: int = 24,
    min_cluster_size: int = 3,
) -> dict[str, object]:
    return run_clustering(
        topic=topic,
        country=country,
        source=source,
        language=language,
        window_hours=window_hours,
        min_cluster_size=min_cluster_size,
    )


@router.get("/pipeline/clusters/latest", response_model=LatestClusterRunResponse)
def get_latest_clusters(
    topic: str,
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
