from fastapi import APIRouter, Query

from brief.generator import generate_brief_cached
from brief.models import ResearchBrief


router = APIRouter()


@router.get("/brief/{topic}", response_model=ResearchBrief)
def get_brief(
    topic: str,
    country: str = Query(default=""),
    source: str = Query(default=""),
    language: str | None = Query(default=None),
) -> ResearchBrief:
    """Generate a research brief with status and confidence metadata."""
    return generate_brief_cached(
        topic,
        country=country,
        source=source,
        language=language,
    )
