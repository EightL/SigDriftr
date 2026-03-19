import json
import logging
import math
import re
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from brief.models import (
    BriefCalibrationWeights,
    BriefConfidenceContext,
    BriefSourceScope,
    ClusterPriorityWeight,
    ResearchBrief,
)
from brief.prompt import (
    ANALYST_TEMPLATE,
    BRIEF_TEMPLATE,
    EXPLAINER_TEMPLATE,
    LOW_CONFIDENCE_WARNING,
    WRITER_TEMPLATE,
    build_context_block,
    build_json_input_block,
    confidence_label,
)
from config.settings import MIN_BRIEF_CONFIDENCE
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from delta.cluster_drift import get_cluster_drift, get_latest_cluster_drift
from delta.engine import compute_drift


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 60
BRIEF_CACHE_TTL = timedelta(minutes=30)
logger = logging.getLogger(__name__)

CacheKey = tuple[str, str, str, str | None, str, str | None]

# This cache is intentionally process-local for the single-worker demo setup.
_brief_cache: dict[CacheKey, tuple[ResearchBrief, datetime]] = {}
_cache_lock = threading.Lock()
_SEGMENT_ALIASES = {
    "young_urban": "young_urban",
    "young_urban_adults": "young_urban",
    "young_urban_adult": "young_urban",
    "young_adults": "young_urban",
    "family": "family",
    "families": "family",
    "family_household": "family",
    "family_households": "family",
    "senior": "senior",
    "seniors": "senior",
    "senior_citizen": "senior",
    "senior_citizens": "senior",
    "b2b": "b2b",
    "business": "b2b",
    "business_decisionmaker": "b2b",
    "business_decisionmakers": "b2b",
    "business_decision_maker": "b2b",
    "business_decision_makers": "b2b",
}
_DRIFT_TYPE_ALIASES = {
    "concern_spike": "concern_spike",
    "concern": "concern_spike",
    "purchase_surge": "purchase_surge",
    "purchase": "purchase_surge",
    "avoidance_rise": "avoidance_rise",
    "avoidance": "avoidance_rise",
    "frame_shift": "frame_shift",
    "frame": "frame_shift",
    "mixed": "mixed",
    "stable": "stable",
}
_SEGMENT_DISPLAY_NAMES = {
    "young_urban": "Young Urban",
    "family": "Family",
    "senior": "Senior",
    "b2b": "B2B",
}
_INSUFFICIENT_DATA_SEGMENTS = ("young_urban", "family", "senior")
_ALERT_LEVEL_RANK = {
    "no_data": -1,
    "none": 0,
    "mild": 1,
    "strong": 2,
}


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


def _call_ollama_json(prompt: str) -> dict:
    """Call local Ollama in JSON mode and return the parsed response object."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 1024},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        raw = json.loads(response.read().decode("utf-8")).get("response", "").strip()

    if not raw:
        raise RuntimeError(f"Empty response from Ollama model {OLLAMA_MODEL}.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[brief] Ollama JSON decode failed. raw=%s", raw[:1000])
        raise RuntimeError("Ollama returned invalid JSON in JSON mode.") from exc


def _normalize_country(country: str | None) -> str:
    return (country or "").strip().upper()


def _normalize_source(source: str | None) -> str:
    return (source or "").strip().lower()


def _normalize_language(language: str | None) -> str | None:
    normalized = (language or "").strip().lower()
    return normalized or None


def _source_scope(resolution: BriefSourceResolution) -> BriefSourceScope:
    return BriefSourceScope(
        country=resolution.country,
        source=resolution.source,
        language=resolution.language,
    )


def _cache_key(
    topic: str,
    country: str,
    source: str,
    language: str | None,
    mode_key: str,
    run_id: str | None = None,
) -> CacheKey:
    return (topic, country, source, language, mode_key, run_id)


def _cache_mode_keys(source_mode: str) -> list[str]:
    if source_mode == "cluster_drift":
        return ["hierarchical_cluster", "cluster_drift"]
    return ["hierarchical_legacy", "legacy_drift"]


def _cache_mode_for_brief(
    brief: ResearchBrief,
    *,
    source_mode: str,
) -> str:
    if brief.generation_mode == "fallback" or brief.generation_mode is None:
        return source_mode
    return brief.generation_mode


def _get_cached_brief_for_resolution(
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> ResearchBrief | None:
    now = datetime.now(timezone.utc)
    with _cache_lock:
        for mode_key in _cache_mode_keys(resolution.source_mode):
            cached = _brief_cache.get(
                _cache_key(
                    resolution.display_topic,
                    resolution.country,
                    resolution.source,
                    resolution.language,
                    mode_key,
                    run_id,
                )
            )
            if cached and now - cached[1] < BRIEF_CACHE_TTL:
                return cached[0]
    return None


def _store_cached_brief(
    brief: ResearchBrief,
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> None:
    mode_key = _cache_mode_for_brief(brief, source_mode=resolution.source_mode)
    with _cache_lock:
        _brief_cache[
            _cache_key(
                resolution.display_topic,
                resolution.country,
                resolution.source,
                resolution.language,
                mode_key,
                run_id,
            )
        ] = (brief, datetime.now(timezone.utc))


def _safe_json_list(raw: str | None) -> list[object]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _load_article_snapshots(article_ids: list[str]) -> list[dict[str, str]]:
    if not article_ids:
        return []

    conn = get_conn()
    placeholders = ", ".join("?" for _ in article_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, summary
        FROM articles
        WHERE id IN ({placeholders})
        """,
        article_ids,
    ).fetchall()
    article_map = {
        str(row[0]): {
            "article_id": str(row[0]),
            "title": row[1] or "[no title]",
            "summary": row[2] or "",
        }
        for row in rows
    }
    return [article_map[article_id] for article_id in article_ids if article_id in article_map]


def _get_top_articles(
    topic: str,
    segment: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    limit: int = 2,
) -> list[dict]:
    """Fetch recent high-signal articles for a topic and segment."""
    conn = get_conn()
    seg_col = f"s.seg_{segment}"
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT s.article_id, a.title, a.summary, {seg_col} AS relevance, s.extracted_at
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE {seg_col} > 0.3
        {topic_sql}
    """
    params: list[object] = list(topic_params)
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    if normalized_country:
        query += " AND a.country = ?"
        params.append(normalized_country)
    if normalized_source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(normalized_source)
    if normalized_language is not None:
        query += " AND LOWER(a.language) = ?"
        params.append(normalized_language)
    query += " ORDER BY s.extracted_at DESC, relevance DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [
        {
            "article_id": row[0],
            "title": row[1],
            "summary": row[2],
            "relevance": row[3],
            "extracted_at": row[4],
            "segment": segment,
        }
        for row in rows
    ]


def _load_entities_for_articles(article_ids: list[str]) -> dict[str, list[dict[str, str]]]:
    if not article_ids:
        return {}

    conn = get_conn()
    placeholders = ", ".join("?" for _ in article_ids)
    rows = conn.execute(
        f"""
        SELECT article_id, entity_text, entity_label
        FROM article_entities
        WHERE article_id IN ({placeholders})
        ORDER BY article_id ASC, entity_label ASC, entity_text ASC
        """,
        article_ids,
    ).fetchall()

    entities_by_article: dict[str, list[dict[str, str]]] = {}
    for article_id, entity_text, entity_label in rows:
        entities_by_article.setdefault(article_id, []).append(
            {"text": entity_text, "label": entity_label}
        )
    return entities_by_article


def _canonicalize_segment(value: str | None) -> str:
    if not value:
        return ""

    normalized = re.sub(r"\([^)]*\)", "", value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    return _SEGMENT_ALIASES.get(normalized, normalized)


def _canonicalize_drift_type(value: str | None) -> str:
    if not value:
        return ""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_|/]+", "", normalized)
    if normalized in _DRIFT_TYPE_ALIASES:
        return _DRIFT_TYPE_ALIASES[normalized]

    for part in re.split(r"[|/]+", normalized):
        part = part.strip("_")
        if part in _DRIFT_TYPE_ALIASES:
            return _DRIFT_TYPE_ALIASES[part]
    return normalized


def _normalize_brief_payload(data: dict) -> dict:
    normalized = dict(data)
    normalized["most_affected_segment"] = _canonicalize_segment(
        normalized.get("most_affected_segment")
    )
    normalized["drift_type"] = _canonicalize_drift_type(normalized.get("drift_type"))

    hypotheses = []
    for item in normalized.get("hypotheses", []):
        if not isinstance(item, dict):
            hypotheses.append(item)
            continue
        hypothesis = dict(item)
        hypothesis["segment"] = _canonicalize_segment(hypothesis.get("segment"))
        hypotheses.append(hypothesis)
    normalized["hypotheses"] = hypotheses
    return normalized


def _topic_label(topic: str) -> str:
    if topic == "_all":
        return "all monitored topics"
    if topic:
        return topic
    return "this topic"


def _normalize_alert_level(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"strong", "mild", "none"}:
        return normalized
    return "none"


def _rank_segments(drift: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        drift,
        key=lambda item: (
            -_ALERT_LEVEL_RANK.get(str(item.get("alert_level", "none")), 0),
            -float(item.get("drift_magnitude", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
            -int(item.get("article_count", 0) or 0),
            str(item.get("segment", "")),
        ),
    )


def _brief_status_from_drift(drift: list[dict]) -> str:
    total_articles = sum(int(entry.get("article_count", 0) or 0) for entry in drift)
    if not drift or total_articles == 0:
        return "insufficient_data"

    ready_segments = [
        entry
        for entry in drift
        if entry.get("status") == "ready"
        or (
            bool(entry.get("has_data", entry.get("article_count", 0) > 0))
            and bool(entry.get("baseline_is_learned", False))
            and entry.get("confidence", 0.0) >= MIN_BRIEF_CONFIDENCE
        )
    ]
    if ready_segments:
        return "ready"
    return "warming"


def _confidence_context_from_segments(
    segments: list[dict[str, object]],
) -> BriefConfidenceContext:
    return BriefConfidenceContext(
        segment_confidence={
            str(entry["segment"]): float(entry.get("confidence", 0.0) or 0.0)
            for entry in segments
            if entry.get("segment") in _SEGMENT_DISPLAY_NAMES
        },
        baseline_is_learned={
            str(entry["segment"]): bool(entry.get("baseline_is_learned", False))
            for entry in segments
            if entry.get("segment") in _SEGMENT_DISPLAY_NAMES
        },
        baseline_sample_count={
            str(entry["segment"]): int(entry.get("baseline_sample_count", 0) or 0)
            for entry in segments
            if entry.get("segment") in _SEGMENT_DISPLAY_NAMES
        },
    )


def _insufficient_data_brief(topic: str, generated_at: str) -> ResearchBrief:
    topic_label = _topic_label(topic)
    return ResearchBrief(
        topic=topic,
        status="insufficient_data",
        headline=f"Insufficient recent coverage for {topic_label}",
        narrative=(
            f"No recent articles were collected for {topic_label}. "
            "This is a data-availability notice rather than a drift finding; "
            "rerun the pipeline later or widen the monitoring window."
        ),
        most_affected_segment="young_urban",
        drift_type="stable",
        alert_level="none",
        hypotheses=[
            {
                "segment": _INSUFFICIENT_DATA_SEGMENTS[0],
                "hypothesis": "Behavior change may become measurable once recent coverage volume increases.",
                "signal_basis": "Insufficient recent evidence to estimate segment drift yet.",
                "suggested_question": "Recent coverage on this topic has been noticeable to me.",
            },
            {
                "segment": _INSUFFICIENT_DATA_SEGMENTS[1],
                "hypothesis": "Audience reactions may diverge once the topic appears more consistently in coverage.",
                "signal_basis": "Additional article volume is needed before segment comparisons are reliable.",
                "suggested_question": "I have seen enough recent coverage on this topic to form an opinion.",
            },
            {
                "segment": _INSUFFICIENT_DATA_SEGMENTS[2],
                "hypothesis": "Awareness and concern may remain flat until the topic reaches sustained visibility.",
                "signal_basis": "Current window contains too little evidence for a grounded behavioral summary.",
                "suggested_question": "If coverage on this topic increases, it would affect my expectations or behavior.",
            },
        ],
        generated_at=generated_at,
        model_used=OLLAMA_MODEL,
    )


def _derive_drift_type(top_entry: dict[str, object]) -> tuple[str, str]:
    deltas = top_entry.get("deltas", {})
    signal_key = max(
        ("concern_level", "purchase_intent", "avoidance_signals"),
        key=lambda key: abs(float((deltas or {}).get(key, 0.0) or 0.0)),
        default="concern_level",
    )
    signal_value = float((deltas or {}).get(signal_key, 0.0) or 0.0)

    if signal_key == "concern_level" and signal_value > 0:
        return "concern_spike", "Concern Spike"
    if signal_key == "purchase_intent" and signal_value > 0:
        return "purchase_surge", "Purchase Intent Surge"
    if signal_key == "avoidance_signals" and signal_value > 0:
        return "avoidance_rise", "Avoidance Rise"
    if bool(top_entry.get("frame_shift")):
        return "frame_shift", "Frame Shift"
    if float(top_entry.get("drift_magnitude", 0.0) or 0.0) <= 0:
        return "stable", "Stable Signals"
    return "mixed", "Behavioral Drift"


def _fallback_hypothesis_segments(drift: list[dict[str, object]]) -> list[str]:
    hypothesis_segments: list[str] = []
    for entry in _rank_segments(drift):
        segment = entry.get("segment")
        if segment in _SEGMENT_DISPLAY_NAMES and segment not in hypothesis_segments:
            hypothesis_segments.append(str(segment))
        if len(hypothesis_segments) == 3:
            break
    for segment in ("young_urban", "family", "senior", "b2b"):
        if segment not in hypothesis_segments:
            hypothesis_segments.append(segment)
        if len(hypothesis_segments) == 3:
            break
    return hypothesis_segments


def _fallback_brief(
    topic: str,
    generated_at: str,
    drift: list[dict[str, object]],
    status: str,
    *,
    narrative_hint: str | None = None,
) -> ResearchBrief:
    ranked_segments = _rank_segments(drift)
    top_entry = ranked_segments[0] if ranked_segments else {}
    top_segment = str(top_entry.get("segment", "young_urban"))
    alert_level = _normalize_alert_level(str(top_entry.get("alert_level", "none")))
    drift_type, signal_phrase = _derive_drift_type(top_entry)
    display_name = _SEGMENT_DISPLAY_NAMES.get(
        top_segment, top_segment.replace("_", " ").title()
    )
    hypothesis_segments = _fallback_hypothesis_segments(drift)
    narrative = narrative_hint or (
        f"Computed drift results show the strongest movement in the {display_name.lower()} segment, "
        f"with {signal_phrase.lower()} standing out in the current calibration window. "
        "This brief uses deterministic synthesis from the validated evidence snapshot."
    )

    return ResearchBrief(
        topic=topic,
        status=status,
        headline=f"{display_name} Segment Shows {signal_phrase}",
        narrative=narrative,
        most_affected_segment=top_segment,
        drift_type=drift_type,
        alert_level=alert_level,
        hypotheses=[
            {
                "segment": hypothesis_segments[0],
                "hypothesis": "The most shifted segment will show measurable behavior change in follow-up research.",
                "signal_basis": "Derived from the highest calibrated drift in the current evidence window.",
                "suggested_question": "How much has recent coverage changed your near-term behavior?",
            },
            {
                "segment": hypothesis_segments[1],
                "hypothesis": "The second-most affected segment will show a measurable shift if current coverage patterns persist.",
                "signal_basis": "Derived from the next strongest validated segment signal.",
                "suggested_question": "How much has recent coverage changed your expected behavior this month?",
            },
            {
                "segment": hypothesis_segments[2],
                "hypothesis": "The third-most affected segment will show directional movement if uncertainty persists.",
                "signal_basis": "Derived from the remaining top-ranked segment in the evidence snapshot.",
                "suggested_question": "How likely are you to change a planned decision because of recent coverage?",
            },
        ],
        generated_at=generated_at,
        model_used=OLLAMA_MODEL,
    )


def _apply_confidence_language(
    brief: ResearchBrief,
    drift: list[dict],
) -> ResearchBrief:
    drift_by_segment = {entry["segment"]: entry for entry in drift}
    segment_entry = drift_by_segment.get(brief.most_affected_segment)
    if segment_entry is None and drift:
        segment_entry = _rank_segments(drift)[0]

    narrative = brief.narrative.strip()
    if segment_entry is not None:
        qualifier = confidence_label(float(segment_entry.get("confidence", 0.0) or 0.0))
        prefix = (
            "Exploratory finding."
            if qualifier == "exploratory finding"
            else f"{qualifier.capitalize()} finding."
        )
        if qualifier not in narrative.lower():
            narrative = f"{prefix} {narrative}".strip()

    if drift and all(
        float(entry.get("confidence", 0.0) or 0.0) < MIN_BRIEF_CONFIDENCE
        for entry in drift
    ) and LOW_CONFIDENCE_WARNING not in narrative:
        narrative = f"{LOW_CONFIDENCE_WARNING} {narrative}".strip()

    return _copy_brief(brief, narrative=narrative)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    normalized_scores = {
        key: max(0.0, float(value or 0.0))
        for key, value in scores.items()
    }
    total = sum(normalized_scores.values())
    if total <= 0:
        return {key: 0.0 for key in normalized_scores}
    return {
        key: round(value / total, 4)
        for key, value in normalized_scores.items()
    }


def _segment_priority(
    segments: list[dict[str, object]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for entry in segments:
        segment = str(entry.get("segment", ""))
        if segment not in _SEGMENT_DISPLAY_NAMES:
            continue
        scores[segment] = (
            float(entry.get("confidence", 0.0) or 0.0)
            * float(entry.get("drift_magnitude", 0.0) or 0.0)
            * math.log1p(int(entry.get("article_count", 0) or 0))
        )
    return _normalize_scores(scores)


def _cluster_priority_weights(
    observations: list[dict[str, object]],
) -> list[ClusterPriorityWeight]:
    raw_scores: dict[str, float] = {}
    lookup: dict[str, dict[str, object]] = {}
    for observation in observations:
        track_id = str(observation.get("track_id", observation.get("observation_id", "")))
        if not track_id:
            continue
        lookup[track_id] = observation
        raw_scores[track_id] = (
            float(observation.get("confidence", 0.0) or 0.0)
            * float(observation.get("drift_magnitude", 0.0) or 0.0)
            * math.log1p(
                max(
                    int(observation.get("member_count", 0) or 0),
                    int(observation.get("baseline_member_count", 0) or 0),
                )
            )
        )
    normalized = _normalize_scores(raw_scores)
    ranked_track_ids = sorted(
        normalized,
        key=lambda track_id: (
            -normalized[track_id],
            track_id,
        ),
    )
    return [
        ClusterPriorityWeight(
            track_id=track_id,
            cluster_id=(
                int(lookup[track_id]["cluster_id"])
                if lookup[track_id].get("cluster_id") is not None
                else None
            ),
            match_type=str(lookup[track_id].get("match_type", "")),
            priority=normalized[track_id],
        )
        for track_id in ranked_track_ids[:4]
    ]


def _build_calibration_weights(
    *,
    source_mode: Literal["cluster_drift", "legacy_drift"],
    segments: list[dict[str, object]],
    observations: list[dict[str, object]],
) -> BriefCalibrationWeights:
    return BriefCalibrationWeights(
        source_mode=source_mode,
        segment_priority=_segment_priority(segments),
        top_cluster_priorities=_cluster_priority_weights(observations)
        if source_mode == "cluster_drift"
        else [],
    )


def _fading_story_is_dominant(segments: list[dict[str, object]]) -> bool:
    if not segments:
        return False
    top_segment = _rank_segments(segments)[0]
    return str(top_segment.get("direction", "")).lower() == "fading"


def _rank_cluster_observations(
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        observations,
        key=lambda item: (
            -_ALERT_LEVEL_RANK.get(str(item.get("alert_level", "none")), 0),
            -float(item.get("drift_magnitude", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
            -int(item.get("member_count", 0) or 0),
            str(item.get("track_id", "")),
        ),
    )


def _select_cluster_observations(
    segments: list[dict[str, object]],
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    preferred = [
        dict(item)
        for item in observations
        if item.get("match_type") in {"matched", "new"}
    ]
    missing = [
        dict(item)
        for item in observations
        if item.get("match_type") == "missing"
    ]

    selected = _rank_cluster_observations(preferred)[:4]
    if _fading_story_is_dominant(segments) and missing:
        top_missing = _rank_cluster_observations(missing)[0]
        if len(selected) >= 4:
            selected = selected[:3] + [top_missing]
        else:
            selected.append(top_missing)
    return selected


def _load_previous_observation_context(
    track_ids: list[str],
    *,
    current_run_id: str,
) -> dict[str, dict[str, object]]:
    if not track_ids:
        return {}

    conn = get_conn()
    placeholders = ", ".join("?" for _ in track_ids)
    rows = conn.execute(
        f"""
        SELECT track_id, run_id, cluster_id, member_count
        FROM cluster_drift_observations
        WHERE track_id IN ({placeholders})
          AND run_id != ?
          AND cluster_id IS NOT NULL
        ORDER BY track_id ASC, computed_at DESC, id DESC
        """,
        [*track_ids, current_run_id],
    ).fetchall()

    previous: dict[str, dict[str, object]] = {}
    for track_id, run_id, cluster_id, member_count in rows:
        if track_id in previous:
            continue
        previous[str(track_id)] = {
            "run_id": str(run_id),
            "cluster_id": int(cluster_id),
            "member_count": int(member_count or 0),
        }
    return previous


def _load_cluster_signal_context(
    *,
    run_id: str,
    cluster_id: int,
) -> dict[str, object] | None:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT evidence_json, exemplar_article_ids
        FROM cluster_signals
        WHERE run_id = ? AND cluster_id = ?
        """,
        (run_id, cluster_id),
    ).fetchone()
    if row is None:
        return None
    article_ids = [str(item) for item in _safe_json_list(row[1]) if isinstance(item, str)]
    return {
        "evidence_json": [str(item) for item in _safe_json_list(row[0]) if isinstance(item, str)],
        "exemplar_article_ids": article_ids,
        "exemplar_articles": _load_article_snapshots(article_ids),
    }


def _enrich_cluster_observations(
    run_id: str,
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    previous_context = _load_previous_observation_context(
        [str(item.get("track_id", "")) for item in observations],
        current_run_id=run_id,
    )
    enriched: list[dict[str, object]] = []
    for item in observations:
        observation = dict(item)
        track_id = str(observation.get("track_id", ""))
        previous = previous_context.get(track_id, {})
        observation["observation_id"] = track_id or f"cluster:{observation.get('cluster_id')}"
        observation["baseline_member_count"] = int(previous.get("member_count", 0) or 0)

        signal_run_id = run_id
        signal_cluster_id = observation.get("cluster_id")
        if observation.get("match_type") == "missing":
            signal_run_id = str(previous.get("run_id", ""))
            signal_cluster_id = previous.get("cluster_id")

        signal_context = None
        if signal_run_id and signal_cluster_id is not None:
            signal_context = _load_cluster_signal_context(
                run_id=signal_run_id,
                cluster_id=int(signal_cluster_id),
            )

        observation["evidence_json"] = (
            list(signal_context["evidence_json"]) if signal_context is not None else []
        )
        observation["exemplar_article_ids"] = (
            list(signal_context["exemplar_article_ids"]) if signal_context is not None else []
        )
        observation["exemplar_articles"] = (
            list(signal_context["exemplar_articles"]) if signal_context is not None else []
        )
        enriched.append(observation)
    return enriched


def _build_legacy_observations(
    resolution: BriefSourceResolution,
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for entry in _rank_segments(segments)[:4]:
        segment = str(entry.get("segment", ""))
        articles = _get_top_articles(
            resolution.real_topic,
            segment,
            country=resolution.country,
            source=resolution.source,
            language=resolution.language,
            limit=2,
        )
        observations.append(
            {
                "observation_id": f"legacy:{segment}",
                "track_id": f"legacy:{segment}",
                "cluster_id": None,
                "match_type": "legacy",
                "direction": entry.get("status", "warming"),
                "topic_label": _SEGMENT_DISPLAY_NAMES.get(segment, segment),
                "baseline_topic_label": _SEGMENT_DISPLAY_NAMES.get(segment, segment),
                "drift_magnitude": float(entry.get("drift_magnitude", 0.0) or 0.0),
                "alert_level": _normalize_alert_level(str(entry.get("alert_level", "none"))),
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
                "member_count": int(entry.get("article_count", 0) or 0),
                "baseline_member_count": int(entry.get("baseline_sample_count", 0) or 0),
                "dominant_frame": entry.get("dominant_frame"),
                "baseline_frame": entry.get("baseline_frame"),
                "deltas": dict(entry.get("deltas", {})),
                "evidence_json": [],
                "exemplar_article_ids": [article["article_id"] for article in articles],
                "exemplar_articles": [
                    {
                        "article_id": article["article_id"],
                        "title": article.get("title") or "[no title]",
                        "summary": article.get("summary") or "",
                    }
                    for article in articles
                ],
            }
        )
    return observations


def _resolve_source_reference(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
    prefer_cluster: bool = True,
    require_cluster: bool = False,
) -> BriefSourceResolution:
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)

    if run_id is not None:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id is required.")
        snapshot = get_cluster_drift(normalized_run_id)
        snapshot_topic = str(snapshot["topic"])
        return BriefSourceResolution(
            display_topic=snapshot_topic,
            real_topic=snapshot_topic,
            country=_normalize_country(str(snapshot.get("country", ""))),
            source=_normalize_source(str(snapshot.get("source", ""))),
            language=_normalize_language(snapshot.get("language")),
            source_mode="cluster_drift",
            generation_mode="hierarchical_cluster",
            run_id=str(snapshot["run_id"]),
            cluster_snapshot=snapshot,
        )

    if not topic:
        raise ValueError("topic is required.")

    display_topic = topic
    real_topic = "" if topic == "_all" else topic
    if prefer_cluster and real_topic:
        snapshot = get_latest_cluster_drift(
            topic=real_topic,
            country=normalized_country,
            source=normalized_source,
            language=normalized_language,
        )
        if snapshot is not None:
            return BriefSourceResolution(
                display_topic=display_topic,
                real_topic=real_topic,
                country=_normalize_country(str(snapshot.get("country", normalized_country))),
                source=_normalize_source(str(snapshot.get("source", normalized_source))),
                language=_normalize_language(snapshot.get("language")),
                source_mode="cluster_drift",
                generation_mode="hierarchical_cluster",
                run_id=str(snapshot["run_id"]),
                cluster_snapshot=snapshot,
            )

    if require_cluster:
        raise RuntimeError("No cluster drift run found for the requested scope.")

    return BriefSourceResolution(
        display_topic=display_topic,
        real_topic=real_topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        source_mode="legacy_drift",
        generation_mode="hierarchical_legacy",
    )


def _build_bundle(resolution: BriefSourceResolution) -> BriefBundle:
    generated_at = datetime.now(timezone.utc).isoformat()
    if resolution.source_mode == "cluster_drift":
        snapshot = resolution.cluster_snapshot
        if snapshot is None:
            if resolution.run_id is None:
                raise RuntimeError("Cluster snapshot was not resolved.")
            snapshot = get_cluster_drift(resolution.run_id)
        segments = [dict(item) for item in snapshot.get("segments", [])]
        selected = _enrich_cluster_observations(
            str(snapshot["run_id"]),
            _select_cluster_observations(segments, list(snapshot.get("clusters", []))),
        )
    else:
        segments = compute_drift(
            resolution.real_topic,
            country=resolution.country,
            source=resolution.source,
            language=resolution.language,
        )
        selected = _build_legacy_observations(resolution, segments)

    return BriefBundle(
        resolution=resolution,
        generated_at=generated_at,
        status=_brief_status_from_drift(segments),
        segment_rollups=segments,
        selected_observations=selected,
        confidence_context=_confidence_context_from_segments(segments),
        calibration_weights=_build_calibration_weights(
            source_mode=resolution.source_mode,
            segments=segments,
            observations=selected,
        ),
    )


def _domain_for_bundle(bundle: BriefBundle) -> str:
    if bundle.segment_rollups:
        return str(bundle.segment_rollups[0].get("domain", "generic"))
    return "generic"


def _relevant_fields_for_bundle(bundle: BriefBundle) -> list[str]:
    if bundle.segment_rollups:
        return [
            str(item)
            for item in bundle.segment_rollups[0].get("relevant_fields", [])
            if isinstance(item, str)
        ]
    return []


def _segment_prompt_view(segment: dict[str, object]) -> dict[str, object]:
    return {
        "segment": segment.get("segment"),
        "direction": segment.get("direction"),
        "status": segment.get("status"),
        "alert_level": _normalize_alert_level(str(segment.get("alert_level", "none"))),
        "article_count": int(segment.get("article_count", 0) or 0),
        "drift_magnitude": float(segment.get("drift_magnitude", 0.0) or 0.0),
        "confidence": float(segment.get("confidence", 0.0) or 0.0),
        "dominant_frame": segment.get("dominant_frame"),
        "baseline_frame": segment.get("baseline_frame"),
        "deltas": dict(segment.get("deltas", {})),
    }


def _observation_prompt_view(observation: dict[str, object]) -> dict[str, object]:
    return {
        "observation_id": observation.get("observation_id"),
        "track_id": observation.get("track_id"),
        "cluster_id": observation.get("cluster_id"),
        "match_type": observation.get("match_type"),
        "direction": observation.get("direction"),
        "topic_label": observation.get("topic_label"),
        "baseline_topic_label": observation.get("baseline_topic_label"),
        "alert_level": _normalize_alert_level(str(observation.get("alert_level", "none"))),
        "drift_magnitude": float(observation.get("drift_magnitude", 0.0) or 0.0),
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "member_count": int(observation.get("member_count", 0) or 0),
        "baseline_member_count": int(observation.get("baseline_member_count", 0) or 0),
        "dominant_frame": observation.get("dominant_frame"),
        "baseline_frame": observation.get("baseline_frame"),
        "deltas": dict(observation.get("deltas", {})),
        "evidence_json": list(observation.get("evidence_json", [])),
        "exemplar_article_ids": list(observation.get("exemplar_article_ids", [])),
        "exemplar_articles": list(observation.get("exemplar_articles", [])),
    }


def _analyst_input(bundle: BriefBundle) -> dict[str, object]:
    return {
        "topic": bundle.resolution.display_topic,
        "source_mode": bundle.resolution.source_mode,
        "source_scope": _model_dump(_source_scope(bundle.resolution)),
        "status": bundle.status,
        "domain": _domain_for_bundle(bundle),
        "relevant_fields": _relevant_fields_for_bundle(bundle),
        "run_id": bundle.resolution.run_id,
        "generated_at": bundle.generated_at,
        "segment_rollups": [
            _segment_prompt_view(item)
            for item in _rank_segments(bundle.segment_rollups)
        ],
        "selected_observations": [
            _observation_prompt_view(item)
            for item in bundle.selected_observations
        ],
        "calibration_weights": _model_dump(bundle.calibration_weights),
    }


def _run_analyst(bundle: BriefBundle) -> AnalystArtifact:
    data = _call_ollama_json(
        ANALYST_TEMPLATE.format(
            input_json=build_json_input_block(_analyst_input(bundle)),
        )
    )
    return _model_validate(AnalystArtifact, data)


def _explainer_input(bundle: BriefBundle, analyst: AnalystArtifact) -> dict[str, object]:
    return {
        "topic": bundle.resolution.display_topic,
        "source_mode": bundle.resolution.source_mode,
        "status": bundle.status,
        "segment_rollups": [
            _segment_prompt_view(item)
            for item in _rank_segments(bundle.segment_rollups)
        ],
        "analyst_output": _model_dump(analyst),
    }


def _run_explainer(bundle: BriefBundle, analyst: AnalystArtifact) -> ExplainerArtifact:
    data = _call_ollama_json(
        EXPLAINER_TEMPLATE.format(
            input_json=build_json_input_block(_explainer_input(bundle, analyst)),
        )
    )
    return _model_validate(ExplainerArtifact, data)


def _writer_input(
    bundle: BriefBundle,
    analyst: AnalystArtifact,
    explainer: ExplainerArtifact,
) -> dict[str, object]:
    return {
        "topic": bundle.resolution.display_topic,
        "status": bundle.status,
        "domain": _domain_for_bundle(bundle),
        "relevant_fields": _relevant_fields_for_bundle(bundle),
        "source_scope": _model_dump(_source_scope(bundle.resolution)),
        "source_mode": bundle.resolution.source_mode,
        "segment_rollups": [
            _segment_prompt_view(item)
            for item in _rank_segments(bundle.segment_rollups)
        ],
        "analyst_output": _model_dump(analyst),
        "explainer_output": _model_dump(explainer),
        "calibration_weights": _model_dump(bundle.calibration_weights),
    }


def _finalize_brief(
    brief: ResearchBrief,
    bundle: BriefBundle,
    *,
    generation_mode: Literal["hierarchical_cluster", "hierarchical_legacy", "fallback"],
) -> ResearchBrief:
    finalized = brief
    if finalized.status != "insufficient_data":
        finalized = _apply_confidence_language(finalized, bundle.segment_rollups)
    return _copy_brief(
        finalized,
        confidence_context=bundle.confidence_context,
        generation_mode=generation_mode,
        calibration_weights=bundle.calibration_weights,
        source_scope=_source_scope(bundle.resolution),
    )


def _run_writer(
    bundle: BriefBundle,
    analyst: AnalystArtifact,
    explainer: ExplainerArtifact,
) -> ResearchBrief:
    data = _call_ollama_json(
        WRITER_TEMPLATE.format(
            input_json=build_json_input_block(_writer_input(bundle, analyst, explainer)),
        )
    )
    data = _normalize_brief_payload(data)
    data["topic"] = bundle.resolution.display_topic
    data["status"] = bundle.status
    data["generated_at"] = bundle.generated_at
    data["model_used"] = OLLAMA_MODEL
    brief = ResearchBrief(**data)
    return _finalize_brief(
        brief,
        bundle,
        generation_mode=bundle.resolution.generation_mode,
    )


def _artifact_narrative_hint(
    analyst: AnalystArtifact | None,
    explainer: ExplainerArtifact | None,
) -> str | None:
    if explainer is not None:
        parts = [explainer.what_changed.strip(), explainer.for_whom.strip()]
        caveats = [item.strip() for item in explainer.uncertainty_and_caveats if item.strip()]
        if caveats:
            parts.append(f"Uncertainty remains around {'; '.join(caveats[:2])}.")
        narrative = " ".join(part for part in parts if part)
        return narrative or None

    if analyst is not None:
        parts = [item.strip() for item in analyst.facts[:2] if item.strip()]
        changes = [item.strip() for item in analyst.numeric_changes[:2] if item.strip()]
        gaps = [item.strip() for item in analyst.evidence_gaps[:1] if item.strip()]
        if changes:
            parts.append(f"Key numeric changes: {'; '.join(changes)}.")
        if gaps:
            parts.append(f"Evidence gaps: {'; '.join(gaps)}.")
        narrative = " ".join(parts)
        return narrative or None

    return None


def _deterministic_fallback_brief(
    bundle: BriefBundle,
    *,
    analyst: AnalystArtifact | None = None,
    explainer: ExplainerArtifact | None = None,
) -> ResearchBrief:
    brief = _fallback_brief(
        bundle.resolution.display_topic,
        bundle.generated_at,
        bundle.segment_rollups,
        bundle.status,
        narrative_hint=_artifact_narrative_hint(analyst, explainer),
    )
    return _finalize_brief(brief, bundle, generation_mode="fallback")


def _generate_hierarchical_brief(bundle: BriefBundle) -> ResearchBrief:
    if bundle.status == "insufficient_data":
        return _finalize_brief(
            _insufficient_data_brief(bundle.resolution.display_topic, bundle.generated_at),
            bundle,
            generation_mode=bundle.resolution.generation_mode,
        )

    analyst: AnalystArtifact | None = None
    explainer: ExplainerArtifact | None = None

    try:
        analyst = _run_analyst(bundle)
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Analyst failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        return _deterministic_fallback_brief(bundle)

    try:
        explainer = _run_explainer(bundle, analyst)
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Explainer failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        return _deterministic_fallback_brief(bundle, analyst=analyst)

    try:
        return _run_writer(bundle, analyst, explainer)
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Writer failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        return _deterministic_fallback_brief(
            bundle,
            analyst=analyst,
            explainer=explainer,
        )


def _generate_legacy_single_pass_brief(
    resolution: BriefSourceResolution,
) -> ResearchBrief:
    """Retained for comparison while the hierarchical path stabilizes."""
    drift = compute_drift(
        resolution.real_topic,
        country=resolution.country,
        source=resolution.source,
        language=resolution.language,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    brief_status = _brief_status_from_drift(drift)
    confidence_context = _confidence_context_from_segments(drift)

    if brief_status == "insufficient_data":
        return _copy_brief(
            _insufficient_data_brief(resolution.display_topic, generated_at),
            confidence_context=confidence_context,
            generation_mode="hierarchical_legacy",
            calibration_weights=BriefCalibrationWeights(
                source_mode="legacy_drift",
                segment_priority=_segment_priority(drift),
                top_cluster_priorities=[],
            ),
            source_scope=_source_scope(resolution),
        )

    top_articles: list[dict] = []
    for entry in _rank_segments(drift):
        top_articles.extend(
            _get_top_articles(
                resolution.real_topic,
                str(entry["segment"]),
                country=resolution.country,
                source=resolution.source,
                language=resolution.language,
                limit=2,
            )
        )
    entities_by_article = _load_entities_for_articles(
        [article["article_id"] for article in top_articles]
    )
    for article in top_articles:
        article["entities"] = entities_by_article.get(article["article_id"], [])

    context_block = build_context_block(drift, top_articles, confidence_context)
    domain = drift[0].get("domain", "generic") if drift else "generic"
    relevant_fields = drift[0].get("relevant_fields", []) if drift else []

    data = _call_ollama_json(
        BRIEF_TEMPLATE.format(
            topic=resolution.display_topic,
            date=generated_at,
            context_block=context_block,
            domain=domain,
            relevant_fields=", ".join(relevant_fields) if relevant_fields else "none",
            model=OLLAMA_MODEL,
        )
    )
    data = _normalize_brief_payload(data)
    data["generated_at"] = generated_at
    data["model_used"] = OLLAMA_MODEL
    data["topic"] = resolution.display_topic
    data["status"] = brief_status
    brief = _apply_confidence_language(ResearchBrief(**data), drift)
    return _copy_brief(
        brief,
        confidence_context=confidence_context,
        generation_mode="hierarchical_legacy",
        calibration_weights=BriefCalibrationWeights(
            source_mode="legacy_drift",
            segment_priority=_segment_priority(drift),
            top_cluster_priorities=[],
        ),
        source_scope=_source_scope(resolution),
    )


def clear_brief_cache() -> None:
    with _cache_lock:
        _brief_cache.clear()


def generate_brief(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
    prefer_cluster: bool = True,
    require_cluster: bool = False,
) -> ResearchBrief:
    """Generate a hierarchical brief, preferring cluster drift when it exists for the scope."""
    resolution = _resolve_source_reference(
        topic,
        country=country,
        source=source,
        language=language,
        run_id=run_id,
        prefer_cluster=prefer_cluster,
        require_cluster=require_cluster,
    )
    return _generate_hierarchical_brief(_build_bundle(resolution))


def generate_brief_cached(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
    prefer_cluster: bool = True,
    require_cluster: bool = False,
) -> ResearchBrief:
    """Return a cached scoped brief when it is still fresh enough for demo use."""
    resolution = _resolve_source_reference(
        topic,
        country=country,
        source=source,
        language=language,
        run_id=run_id,
        prefer_cluster=prefer_cluster,
        require_cluster=require_cluster,
    )
    cache_run_id = resolution.run_id if run_id is not None else None
    cached = _get_cached_brief_for_resolution(resolution, run_id=cache_run_id)
    if cached is not None:
        return cached

    brief = _generate_hierarchical_brief(_build_bundle(resolution))
    _store_cached_brief(brief, resolution, run_id=cache_run_id)
    return brief


def generate_hierarchical_brief(
    *,
    topic: str = "",
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
) -> ResearchBrief:
    return generate_brief(
        topic,
        country=country,
        source=source,
        language=language,
        run_id=run_id,
        prefer_cluster=True,
        require_cluster=True,
    )


def generate_hierarchical_brief_cached(
    *,
    topic: str = "",
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
) -> ResearchBrief:
    return generate_brief_cached(
        topic,
        country=country,
        source=source,
        language=language,
        run_id=run_id,
        prefer_cluster=True,
        require_cluster=True,
    )


def peek_cached_brief(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> ResearchBrief | None:
    resolution = _resolve_source_reference(
        topic,
        country=country,
        source=source,
        language=language,
        prefer_cluster=True,
        require_cluster=False,
    )
    return _get_cached_brief_for_resolution(resolution)


def get_brief_summary(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> dict[str, object]:
    cached = peek_cached_brief(
        topic,
        country=country,
        source=source,
        language=language,
    )
    if cached is not None:
        return {
            "topic": cached.topic,
            "status": cached.status,
            "alert_level": cached.alert_level,
            "confidence_context": cached.confidence_context,
        }

    resolution = _resolve_source_reference(
        topic,
        country=country,
        source=source,
        language=language,
        prefer_cluster=True,
        require_cluster=False,
    )
    if resolution.source_mode == "cluster_drift":
        segments = list((resolution.cluster_snapshot or {}).get("segments", []))
    else:
        segments = compute_drift(
            resolution.real_topic,
            country=resolution.country,
            source=resolution.source,
            language=resolution.language,
        )
    ranked = _rank_segments(segments)
    alert_level = (
        _normalize_alert_level(str(ranked[0].get("alert_level", "none")))
        if ranked
        else "none"
    )
    return {
        "topic": resolution.display_topic,
        "status": _brief_status_from_drift(segments),
        "alert_level": alert_level,
        "confidence_context": _confidence_context_from_segments(segments),
    }
