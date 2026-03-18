from fastapi import APIRouter

from ingestion.crawler import _crawl_async


router = APIRouter()


@router.post("/collect")
async def collect(topic: str) -> dict[str, int | str]:
    inserted = await _crawl_async(topic)
    return {"inserted": inserted, "topic": topic}
