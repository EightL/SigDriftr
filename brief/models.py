from typing import Literal

from pydantic import BaseModel, Field


class SurveyHypothesis(BaseModel):
    segment: str = Field(description="Target audience segment")
    hypothesis: str = Field(
        description="Falsifiable behavioral hypothesis (1-2 sentences)"
    )
    signal_basis: str = Field(
        description="Which signal drove this hypothesis (e.g. concern_level +0.18)"
    )
    suggested_question: str = Field(
        description="One Likert-scale survey question to test this hypothesis"
    )


class ResearchBrief(BaseModel):
    topic: str
    headline: str = Field(
        description="8-12 word summary of the dominant behavioral shift"
    )
    narrative: str = Field(
        description="2-3 sentence plain-language brief for a Lakmoos analyst"
    )
    most_affected_segment: str
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
    generated_at: str
    model_used: str

