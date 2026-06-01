from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from brief.models import (
    BriefCalibrationWeights,
    BriefConfidenceContext,
    ResearchBrief,
)


class AnalystArtifact(BaseModel):
    facts: list[str] = Field(default_factory=list, min_length=1)
    numeric_changes: list[str] = Field(default_factory=list)
    cited_clusters: list[str] = Field(default_factory=list)
    cited_articles: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)


class ExplainerArtifact(BaseModel):
    what_changed: str
    for_whom: str
    uncertainty_and_caveats: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class BriefSourceResolution:
    display_topic: str
    real_topic: str
    canonical_topic_id: str
    canonical_display_name: str
    country: str
    source: str
    language: str | None
    source_mode: Literal["cluster_drift", "legacy_drift"]
    generation_mode: Literal["hierarchical_cluster", "hierarchical_legacy"]
    run_id: str | None = None
    cluster_snapshot: dict[str, object] | None = None


@dataclass(frozen=True)
class BriefBundle:
    resolution: BriefSourceResolution
    generated_at: str
    status: str
    segment_rollups: list[dict[str, object]]
    selected_observations: list[dict[str, object]]
    confidence_context: BriefConfidenceContext
    calibration_weights: BriefCalibrationWeights


@dataclass(frozen=True)
class BriefArtifacts:
    brief: ResearchBrief
    support: dict[str, object]


def _copy_brief(brief: ResearchBrief, **updates: object) -> ResearchBrief:
    if hasattr(brief, "model_copy"):
        return brief.model_copy(update=updates)
    return brief.copy(update=updates)


def _model_validate(model_cls, data: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _model_dump(model: BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)
