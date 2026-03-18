from typing import Literal

from pydantic import BaseModel, Field

SegmentKey = Literal["young_urban", "family", "senior", "b2b"]


class SurveyHypothesis(BaseModel):
    segment: SegmentKey = Field(description="Target audience segment")
    hypothesis: str = Field(
        description="Falsifiable behavioral hypothesis (1-2 sentences)"
    )
    signal_basis: str = Field(
        description="Which signal drove this hypothesis (e.g. concern_level +0.18)"
    )
    suggested_question: str = Field(
        description="One Likert-scale survey question to test this hypothesis"
    )


class BriefConfidenceContext(BaseModel):
    segment_confidence: dict[SegmentKey, float] = Field(default_factory=dict)
    baseline_is_learned: dict[SegmentKey, bool] = Field(default_factory=dict)
    baseline_sample_count: dict[SegmentKey, int] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    topic: str
    status: Literal["insufficient_data", "warming", "ready"] = Field(
        description="Whether the brief is a data notice, an exploratory result, or ready for normal use."
    )
    headline: str = Field(
        description="8-12 word summary of the dominant behavioral shift"
    )
    narrative: str = Field(
        description="2-3 sentence plain-language brief for a Lakmoos analyst"
    )
    most_affected_segment: SegmentKey
    drift_type: Literal[
        "concern_spike",
        "purchase_surge",
        "avoidance_rise",
        "frame_shift",
        "mixed",
        "stable",
    ]
    hypotheses: list[SurveyHypothesis] = Field(min_length=3, max_length=3)
    alert_level: Literal["none", "mild", "strong"]
    confidence_context: BriefConfidenceContext | None = Field(
        default=None,
        description="Per-segment calibration metadata; None when not available.",
    )
    generated_at: str
    model_used: str
