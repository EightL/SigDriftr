from fastapi import APIRouter

from api.pipeline import run_collection_cycle


router = APIRouter()


@router.post("/collect")
async def collect(
    topic: str,
    country: str = "",
    source: str = "",
) -> dict[str, int | str]:
    return await run_collection_cycle(topic, country=country, source=source)
