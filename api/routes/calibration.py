from fastapi import APIRouter, HTTPException

from delta.engine import compute_drift
from delta.mapper import SEGMENTS, compute_segment_profiles


router = APIRouter()


@router.get("/calibration/{topic}/{segment}")
def get_calibration(topic: str, segment: str) -> dict:
    if segment not in SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segment '{segment}'. Must be one of: {SEGMENTS}",
        )

    real_topic = "" if topic == "_all" else topic
    profiles = compute_segment_profiles(real_topic)
    match = next((profile for profile in profiles if profile["segment"] == segment), None)
    if match is None:
        raise HTTPException(status_code=404, detail="No data for this topic/segment.")
    return match


@router.get("/drift/{topic}")
def get_drift(topic: str, days_back: int = 7) -> dict:
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(real_topic, days_back=days_back)
    return {
        "topic": topic,
        "days_back": days_back,
        "segments": drift,
    }
