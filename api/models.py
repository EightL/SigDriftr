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
    status: SegmentStatus | None = None
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
    status: SegmentStatus | None = None
    domain: str | None = "generic"
    relevant_fields: list[str] | None = None


class DriftResponse(BaseModel):
    topic: str
    days_back: int
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


class DigestResponse(BaseModel):
    topic: str
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
