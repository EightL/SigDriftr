from fastapi import APIRouter, Query

from api.models import HistorySegmentResponse
from db.queries import get_profile_history


router = APIRouter()


@router.get("/history/{topic}", response_model=list[HistorySegmentResponse])
def get_history(topic: str, days_back: int = Query(default=30, ge=1)) -> list[dict]:
    real_topic = "" if topic == "_all" else topic
    return get_profile_history(real_topic, days_back=days_back)
