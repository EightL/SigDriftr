from typing import Literal

from pydantic import BaseModel, Field


SegmentStatus = Literal["no_data", "warming", "ready"]


class SegmentConfidence(BaseModel):
    confidence: float | None = None
    baseline_is_learned: bool | None = None
    baseline_sample_count: int | None = None
    baseline_age_days: int | None = None


class SignalRecord(BaseModel):
    article_id: str
    topic: str | None = None
    canonical_topic_id: str | None = None
    title: str | None = None
    outlet: str | None = None
    country: str | None = None
    language: str | None = None
    url: str | None = None
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
    seg_young_urban_relevance: float | None = None
    seg_family_relevance: float | None = None
    seg_senior_relevance: float | None = None
    seg_b2b_relevance: float | None = None
    topic_relevance_score: float | None = None
    topic_relevance: str | None = None
    raw_json: dict = Field(default_factory=dict)
    extracted_at: str
    segment_confidence: dict[str, SegmentConfidence] | None = None


class CalibrationResponse(BaseModel):
    segment: str
    topic: str
    canonical_topic_id: str | None = None
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
    status: SegmentStatus | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None


class DriftSegmentResponse(BaseModel):
    segment: str
    topic: str
    canonical_topic_id: str | None = None
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
    status: SegmentStatus | None = None
    source_mix: dict[str, object] | None = None
    source_normalized: dict[str, object] | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None


class DriftResponse(BaseModel):
    topic: str
    requested_topic: str | None = None
    canonical_topic_id: str | None = None
    canonical_display_name: str | None = None
    days_back: int
    source_mix: dict[str, object] | None = None
    segments: list[DriftSegmentResponse]


class HistorySegmentResponse(BaseModel):
    segment: str
    dates: list[str]
    concern: list[float]
    purchase_intent: list[float]
    avoidance: list[float]
    drift_magnitude: list[float]


class DigestArticle(BaseModel):
    article_id: str
    title: str
    url: str
    outlet: str
    country: str
    published_at: str | None = None
    relevance_score: float


class EvidenceArticle(BaseModel):
    article_id: str
    title: str
    summary: str = ""
    body_excerpt: str = ""
    outlet: str = ""
    country: str = ""
    language: str | None = None
    url: str = ""
    published_at: str | None = None


class DigestResponse(BaseModel):
    topic: str
    requested_topic: str | None = None
    canonical_topic_id: str | None = None
    canonical_display_name: str | None = None
    country: str
    source: str
    article_count: int
    generated_at: str
    sources_used: list[str]
    summary_headline: str
    summary_text: str
    key_points: list[str]
    articles: list[DigestArticle]


class EmbeddingStageResponse(BaseModel):
    model_name: str
    embedding_dim: int
    selected: int
    embedded: int
    already_current: int
    retried_failed: int
    stale_reembedded: int
    failed: int
    duration_s: float


ClusterRunStatus = Literal["completed", "all_noise", "skipped_small_sample"]


class ClusterSignalRecord(BaseModel):
    topic_label: str
    concern_level: float
    purchase_intent: float
    avoidance_signals: float
    sentiment: float
    dominant_frame: str
    frame_detail: str
    seg_young_urban: float
    seg_family: float
    seg_senior: float
    seg_b2b: float
    evidence: list[str] = Field(default_factory=list)
    raw_json: dict = Field(default_factory=dict)
    member_count: int
    membership_fingerprint: str
    exemplar_article_ids: list[str] = Field(default_factory=list)
    exemplar_articles: list[EvidenceArticle] = Field(default_factory=list)
    extractor_provider: str
    extractor_model: str
    schema_version: str
    extracted_at: str


class ClusterSignalStageResponse(BaseModel):
    run_id: str
    selected_clusters: int
    processed: int
    skipped_existing: int
    failed: int
    provider: str
    model_name: str
    duration_s: float


ClusterDriftMatchType = Literal["matched", "new", "missing"]


class ClusterMember(BaseModel):
    article_id: str
    embedding_id: int
    membership_strength: float | None = None
    is_noise: bool


class ClusterResponse(BaseModel):
    cluster_id: int
    cluster_label: int
    size: int
    centroid_vector: list[float]
    centroid_dim: int
    members: list[ClusterMember] = Field(default_factory=list)
    signal: ClusterSignalRecord | None = None


class ClusterRunMetadata(BaseModel):
    run_id: str
    topic: str
    canonical_topic_id: str | None = None
    country: str
    source: str
    language: str | None = None
    window_start: str
    window_end: str
    status: ClusterRunStatus
    n_articles: int
    n_clusters: int
    n_noise: int
    model_name: str
    model_version: str | None = None
    umap_n_components: int
    umap_n_neighbors: int
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int


class ClusterRunResponse(ClusterRunMetadata):
    duration_s: float


class LatestClusterRunResponse(ClusterRunMetadata):
    created_at: str
    clusters: list[ClusterResponse] = Field(default_factory=list)
    noise_members: list[ClusterMember] = Field(default_factory=list)


class ClusterDriftObservationResponse(BaseModel):
    track_id: str
    cluster_id: int | None = None
    cluster_label: int | None = None
    topic_label: str
    baseline_topic_label: str
    match_type: ClusterDriftMatchType
    direction: str
    centroid_distance: float
    segment_vector_distance: float
    signal_drift: float
    drift_magnitude: float
    alert_level: str
    confidence: float
    member_count: int
    mean_membership_strength: float
    current: dict[str, float]
    baseline: dict[str, float] | None = None
    deltas: dict[str, float]
    dominant_frame: str
    baseline_frame: str | None = None
    frame_shift: bool
    evidence_json: list[str] = Field(default_factory=list)
    exemplar_article_ids: list[str] = Field(default_factory=list)
    exemplar_articles: list[EvidenceArticle] = Field(default_factory=list)


class ClusterDriftSegmentResponse(DriftSegmentResponse):
    direction: str
    centroid_shift: float
    new_cluster_weight: float
    tracked_cluster_count: int
    matched_cluster_count: int
    new_cluster_count: int
    missing_cluster_count: int


class ClusterDriftResponse(BaseModel):
    topic: str
    canonical_topic_id: str | None = None
    country: str
    source: str
    language: str | None = None
    run_id: str
    computed_at: str
    segments: list[ClusterDriftSegmentResponse] = Field(default_factory=list)
    clusters: list[ClusterDriftObservationResponse] = Field(default_factory=list)


class ClusterDriftStageResponse(BaseModel):
    run_id: str
    observed_clusters: int
    matched_tracks: int
    new_tracks: int
    missing_tracks: int
    segments: int
    computed_at: str
    duration_s: float


class ScopeResponse(BaseModel):
    topic: str
    requested_topic: str | None = None
    canonical_topic_id: str | None = None
    canonical_display_name: str | None = None
    country: str = ""
    source: str = ""
    language: str | None = None
    collection_mode: str | None = None
    reward_mode: str | None = None


class PipelineStageStatus(BaseModel):
    status: str
    detail: str | None = None
    count: int | None = None
    duration_s: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class PipelineStageSummary(BaseModel):
    collect: PipelineStageStatus
    extract: PipelineStageStatus
    embed: PipelineStageStatus
    cluster: PipelineStageStatus
    cluster_signals: PipelineStageStatus
    cluster_drift: PipelineStageStatus
    brief: PipelineStageStatus


class PipelineSummaryResponse(BaseModel):
    article_count: int = 0
    signal_count: int = 0
    embedding_count: int = 0
    cluster_count: int = 0
    noise_count: int = 0
    cluster_status: str = "not_run"
    brief_status: str = "insufficient_data"
    strongest_segment: str | None = None
    stages: PipelineStageSummary


class PipelineRunResponse(BaseModel):
    scope: ScopeResponse
    run_id: str | None = None
    generated_at: str
    duration_s: float
    cluster_status: str
    brief_status: str
    pipeline: PipelineSummaryResponse
    brief: dict[str, object] | None = None


class BriefSupportResponse(BaseModel):
    status: str
    source_mode: str
    generation_mode: str | None = None
    cited_track_ids: list[str] = Field(default_factory=list)
    cited_article_ids: list[str] = Field(default_factory=list)
    selected_observation_ids: list[str] = Field(default_factory=list)
    fallback_note: str | None = None


class OutputClusterDriftResponse(BaseModel):
    status: str
    message: str
    data: ClusterDriftResponse | None = None


class OutputResponse(BaseModel):
    scope: ScopeResponse
    run_id: str | None = None
    generated_at: str
    pipeline: PipelineSummaryResponse
    brief: dict[str, object]
    digest: DigestResponse
    cluster_drift: OutputClusterDriftResponse
    clusters: list[ClusterResponse] = Field(default_factory=list)
    brief_support: BriefSupportResponse
    signals: list[SignalRecord] = Field(default_factory=list)
