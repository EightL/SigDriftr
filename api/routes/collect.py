from fastapi import APIRouter

from api.pipeline import run_collection_cycle


router = APIRouter()


@router.post("/collect")
async def collect(
    topic: str,
    country: str = "",
    source: str = "",
    collection_mode: str | None = None,
    reward_mode: str | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"country": country, "source": source}
    if collection_mode is not None:
        kwargs["collection_mode"] = collection_mode
    if reward_mode is not None:
        kwargs["reward_mode"] = reward_mode
    return await run_collection_cycle(
        topic,
        **kwargs,
    )
