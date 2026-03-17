from fastapi import APIRouter

from brief.generator import generate_brief
from brief.models import ResearchBrief


router = APIRouter()


@router.get("/brief/{topic}", response_model=ResearchBrief)
def get_brief(topic: str) -> ResearchBrief:
    """Generate a Lakmoos-style research brief for a topic."""
    return generate_brief(topic)
