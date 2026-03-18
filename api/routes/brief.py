from fastapi import APIRouter

from brief.generator import generate_brief_cached
from brief.models import ResearchBrief


router = APIRouter()


@router.get("/brief/{topic}", response_model=ResearchBrief)
def get_brief(topic: str) -> ResearchBrief:
    """Generate a research brief with status and confidence metadata."""
    return generate_brief_cached(topic)
