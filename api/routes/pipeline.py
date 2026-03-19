from fastapi import APIRouter

from api.pipeline import run_collection_cycle
from brief.generator import generate_brief_cached
from brief.models import ResearchBrief


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
