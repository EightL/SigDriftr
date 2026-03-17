from fastapi import APIRouter

from ingestion.crawler import crawl


router = APIRouter()


@router.post("/collect")
def collect(topic: str) -> dict[str, int | str]:
    inserted = crawl(topic)
    return {"inserted": inserted, "topic": topic}
