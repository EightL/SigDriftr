from fastapi import APIRouter, HTTPException

from api.models import CalibrationResponse, ClusterDriftResponse, DriftResponse
from delta.cluster_drift import get_latest_cluster_drift
from delta.engine import compute_drift
from delta.mapper import SEGMENTS, compute_segment_profiles


router = APIRouter()


@router.get("/calibration/{topic}/{segment}", response_model=CalibrationResponse)
def get_calibration(topic: str, segment: str) -> dict:
    if segment not in SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segment '{segment}'. Must be one of: {SEGMENTS}",
        )

    real_topic = "" if topic == "_all" else topic
    profiles = compute_segment_profiles(real_topic, learn_baseline=False)
    match = next((profile for profile in profiles if profile["segment"] == segment), None)
    if match is None:
        raise HTTPException(status_code=404, detail="No data for this topic/segment.")
    drift = compute_drift(real_topic)
    drift_match = next((item for item in drift if item["segment"] == segment), None)
    if drift_match is None:
        return match
    return {**match, **drift_match}


@router.get("/drift/clusters/{topic}", response_model=ClusterDriftResponse)
def get_cluster_drift_view(
    topic: str,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict:
    result = get_latest_cluster_drift(
        topic=topic,
        country=country,
        source=source,
        language=language,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No cluster drift run found for the requested scope.",
        )
    return result


@router.get("/drift/{topic}", response_model=DriftResponse)
def get_drift(
    topic: str,
    days_back: int = 7,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict:
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(
        real_topic,
        days_back=days_back,
        country=country,
        source=source,
        language=language,
    )
    source_mix = drift[0].get("source_mix") if drift else None
    return {
        "topic": topic,
        "days_back": days_back,
        "source_mix": source_mix,
        "segments": drift,
    }
