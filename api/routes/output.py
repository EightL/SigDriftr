from fastapi import APIRouter, Query

from api.models import OutputResponse
from api.output_service import build_output_bundle


router = APIRouter()


@router.get("/output/{topic}", response_model=OutputResponse)
def get_output_bundle(
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = Query(default=None),
) -> dict[str, object]:
    return build_output_bundle(
        topic,
        country=country,
        source=source,
        language=language,
    )
