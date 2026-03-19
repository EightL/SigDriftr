from fastapi import APIRouter, Query

from api.models import DigestResponse
from brief.digest import generate_digest


router = APIRouter()


@router.get("/summaries", response_model=DigestResponse)
def get_summaries(
    topic: str,
    country: str = "",
    source: str = "",
    limit: int = Query(default=8, ge=1, le=20),
) -> DigestResponse:
    return generate_digest(topic, country=country, source=source, limit=limit)
