from pydantic import BaseModel, Field


class SegmentConfidence(BaseModel):
    confidence: float | None = None
    baseline_is_learned: bool | None = None
    baseline_sample_count: int | None = None
    baseline_age_days: int | None = None


class SignalRecord(BaseModel):
    article_id: str
    topic: str | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None
    concern_level: float | None = None
    purchase_intent: float | None = None
    avoidance_signals: float | None = None
    dominant_frame: str | None = None
    seg_young_urban: float | None = None
    seg_family: float | None = None
    seg_senior: float | None = None
    seg_b2b: float | None = None
    raw_json: dict = Field(default_factory=dict)
    extracted_at: str
    segment_confidence: dict[str, SegmentConfidence] | None = None


class CalibrationResponse(BaseModel):
    segment: str
    topic: str
    window_start: str
    window_days: int
    article_count: int
    concern_level: float
    purchase_intent: float
    avoidance_signals: float
    dominant_frame: str
    confidence: float | None = None
    baseline_is_learned: bool | None = None
    baseline_sample_count: int | None = None
    baseline_age_days: int | None = None
    baseline_frame: str | None = None
    drift_magnitude: float | None = None
    frame_shift: bool | None = None
    alert_level: str | None = None
    deltas: dict[str, float] | None = None
    baseline: dict[str, float] | None = None
    current: dict[str, float] | None = None
    has_data: bool | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None


class DriftSegmentResponse(BaseModel):
    segment: str
    topic: str
    article_count: int
    has_data: bool
    current: dict[str, float]
    baseline: dict[str, float] | None = None
    deltas: dict[str, float]
    drift_magnitude: float
    frame_shift: bool
    alert_level: str
    dominant_frame: str
    baseline_frame: str | None = None
    confidence: float | None = None
    baseline_is_learned: bool | None = None
    baseline_sample_count: int | None = None
    baseline_age_days: int | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None


class DriftResponse(BaseModel):
    topic: str
    days_back: int
    segments: list[DriftSegmentResponse]
