import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import ValidationError

from brief.artifacts import (
    AnalystArtifact,
    BriefArtifacts,
    BriefBundle,
    BriefSourceResolution,
    ExplainerArtifact,
    _copy_brief,
    _model_dump,
    _model_validate,
)
from brief.cache import (
    BRIEF_CACHE_TTL,
    CacheKey,
    _brief_cache,
    _brief_support_cache,
    _cache_key,
    _cache_lock,
    _cache_mode_for_brief,
    _cache_mode_keys,
    _get_cached_brief_for_resolution,
    _get_cached_brief_support_for_resolution,
    _store_cached_brief,
    _store_cached_brief_support,
    clear_brief_cache,
)
from brief.llm import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    call_ollama_json as _call_ollama_json,
)
from brief.models import (
    BriefCalibrationWeights,
    BriefConfidenceContext,
    ClusterPriorityWeight,
    ResearchBrief,
)
from brief.prompt import (
    ANALYST_TEMPLATE,
    EXPLAINER_TEMPLATE,
    LOW_CONFIDENCE_WARNING,
    WRITER_TEMPLATE,
    build_json_input_block,
    confidence_label,
)
from brief.validator import validate_brief, validation_issues_as_dicts
from config.settings import MIN_BRIEF_CONFIDENCE
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from db.topic_resolver import resolve_topic
from delta.cluster_drift import get_cluster_drift, get_latest_cluster_drift
from delta.engine import compute_drift
from brief.source_resolution import (
    _normalize_country,
    _normalize_language,
    _normalize_source,
    _source_scope,
)


logger = logging.getLogger(__name__)

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
                "hypothesis": "Coverage framing may become measurable once recent article volume increases.",
                "signal_basis": "Insufficient recent evidence to estimate segment drift yet.",
                "suggested_question": "Recent coverage on this topic has been noticeable to me.",
            },
            {
                "segment": _INSUFFICIENT_DATA_SEGMENTS[1],
                "hypothesis": "Segment-relevant coverage may diverge once the topic appears more consistently.",
                "signal_basis": "Additional article volume is needed before segment comparisons are reliable.",
                "suggested_question": "I have seen enough recent coverage on this topic to form an opinion.",
            },
            {
                "segment": _INSUFFICIENT_DATA_SEGMENTS[2],
                "hypothesis": "Awareness and concern may remain flat until the topic reaches sustained visibility.",
                "signal_basis": "Current window contains too little evidence for a grounded coverage summary.",
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
    return "mixed", "Mixed Coverage Drift"


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
                "hypothesis": "The most shifted segment will show measurable coverage recall in follow-up research.",
                "signal_basis": "Derived from the highest calibrated drift in the current evidence window.",
                "suggested_question": "How noticeable has recent coverage on this topic been?",
            },
            {
                "segment": hypothesis_segments[1],
                "hypothesis": "The second-most affected segment will show a measurable shift if current coverage patterns persist.",
                "signal_basis": "Derived from the next strongest validated segment signal.",
                "suggested_question": "How strongly does recent coverage shape your expectations this month?",
            },
            {
                "segment": hypothesis_segments[2],
                "hypothesis": "The third-most affected segment will show directional movement if uncertainty persists.",
                "signal_basis": "Derived from the remaining top-ranked segment in the evidence snapshot.",
                "suggested_question": "How relevant is recent coverage to decisions you are already considering?",
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


def enrich_cluster_observations(
    run_id: str,
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    return _enrich_cluster_observations(run_id, observations)


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
        snapshot_resolution = resolve_topic(snapshot_topic)
        canonical_topic_id = str(
            snapshot.get("canonical_topic_id") or snapshot_resolution.canonical_topic_id
        )
        canonical_display_name = str(
            snapshot.get("canonical_display_name") or snapshot_resolution.display_name
        )
        return BriefSourceResolution(
            display_topic=snapshot_topic,
            real_topic=snapshot_topic,
            canonical_topic_id=canonical_topic_id,
            canonical_display_name=canonical_display_name,
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
    topic_resolution = resolve_topic(real_topic) if real_topic else None
    canonical_topic_id = topic_resolution.canonical_topic_id if topic_resolution else ""
    canonical_display_name = topic_resolution.display_name if topic_resolution else display_topic
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
                canonical_topic_id=canonical_topic_id,
                canonical_display_name=canonical_display_name,
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
        canonical_topic_id=canonical_topic_id,
        canonical_display_name=canonical_display_name,
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


def _selected_observation_ids(bundle: BriefBundle) -> list[str]:
    return [
        str(item.get("observation_id", item.get("track_id", "")))
        for item in bundle.selected_observations
        if str(item.get("observation_id", item.get("track_id", ""))).strip()
    ]


def _brief_support_payload(
    bundle: BriefBundle,
    *,
    brief: ResearchBrief | None = None,
    generation_mode: str | None,
    cited_track_ids: list[str] | None = None,
    cited_article_ids: list[str] | None = None,
    fallback_note: str | None = None,
) -> dict[str, object]:
    cited_tracks = list(dict.fromkeys(cited_track_ids or []))
    cited_articles = list(dict.fromkeys(cited_article_ids or []))
    validation_issues = (
        validation_issues_as_dicts(
            validate_brief(
                brief,
                bundle=bundle,
                cited_track_ids=cited_tracks,
                cited_article_ids=cited_articles,
            )
        )
        if brief is not None
        else []
    )
    return {
        "status": bundle.status,
        "source_mode": bundle.resolution.source_mode,
        "generation_mode": generation_mode,
        "cited_track_ids": cited_tracks,
        "cited_article_ids": cited_articles,
        "selected_observation_ids": _selected_observation_ids(bundle),
        "fallback_note": fallback_note,
        "validation_issues": validation_issues,
    }


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
    source_mix = segment.get("source_mix")
    source_mix_view = None
    if isinstance(source_mix, dict):
        source_mix_view = {
            "warning": source_mix.get("warning"),
            "jensen_shannon_divergence": source_mix.get("jensen_shannon_divergence"),
            "current_article_count_by_outlet": (
                source_mix.get("current", {}).get("article_count_by_outlet", {})
                if isinstance(source_mix.get("current"), dict)
                else {}
            ),
            "reference_article_count_by_outlet": (
                source_mix.get("reference", {}).get("article_count_by_outlet", {})
                if isinstance(source_mix.get("reference"), dict)
                else {}
            ),
        }
    source_normalized = segment.get("source_normalized")
    normalized_view = None
    if isinstance(source_normalized, dict):
        normalized_view = {
            "status": source_normalized.get("status"),
            "panel_outlets": source_normalized.get("panel_outlets", []),
            "observed_outlets": source_normalized.get("observed_outlets", []),
            "missing_outlets": source_normalized.get("missing_outlets", []),
            "drift_magnitude": source_normalized.get("drift_magnitude"),
            "normalization_effect": source_normalized.get("normalization_effect"),
            "interpretation": source_normalized.get("interpretation"),
        }
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
        "source_mix": source_mix_view,
        "source_normalized": normalized_view,
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
        requested_topic=bundle.resolution.display_topic,
        canonical_topic_id=bundle.resolution.canonical_topic_id,
        canonical_display_name=bundle.resolution.canonical_display_name,
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


def _generate_hierarchical_brief_artifacts(bundle: BriefBundle) -> BriefArtifacts:
    if bundle.status == "insufficient_data":
        brief = _finalize_brief(
            _insufficient_data_brief(
                bundle.resolution.display_topic,
                bundle.generated_at,
            ),
            bundle,
            generation_mode=bundle.resolution.generation_mode,
        )
        return BriefArtifacts(
            brief=brief,
            support=_brief_support_payload(
                bundle,
                brief=brief,
                generation_mode=brief.generation_mode,
                fallback_note="Insufficient data brief did not emit analyst citations.",
            ),
        )

    analyst: AnalystArtifact | None = None
    explainer: ExplainerArtifact | None = None

    try:
        analyst = _run_analyst(bundle)
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Analyst failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        brief = _deterministic_fallback_brief(bundle)
        return BriefArtifacts(
            brief=brief,
            support=_brief_support_payload(
                bundle,
                brief=brief,
                generation_mode=brief.generation_mode,
                fallback_note="Analyst stage failed; citations unavailable in fallback brief.",
            ),
        )

    try:
        explainer = _run_explainer(bundle, analyst)
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Explainer failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        brief = _deterministic_fallback_brief(bundle, analyst=analyst)
        return BriefArtifacts(
            brief=brief,
            support=_brief_support_payload(
                bundle,
                brief=brief,
                generation_mode=brief.generation_mode,
                fallback_note="Explainer stage failed; citations unavailable in fallback brief.",
            ),
        )

    try:
        brief = _run_writer(bundle, analyst, explainer)
        return BriefArtifacts(
            brief=brief,
            support=_brief_support_payload(
                bundle,
                brief=brief,
                generation_mode=brief.generation_mode,
                cited_track_ids=list(analyst.cited_clusters),
                cited_article_ids=list(analyst.cited_articles),
            ),
        )
    except (ValidationError, RuntimeError, ValueError) as exc:
        logger.warning("[brief] Writer failed topic=%s error=%s", bundle.resolution.display_topic, exc)
        brief = _deterministic_fallback_brief(
            bundle,
            analyst=analyst,
            explainer=explainer,
        )
        return BriefArtifacts(
            brief=brief,
            support=_brief_support_payload(
                bundle,
                brief=brief,
                generation_mode=brief.generation_mode,
                fallback_note="Writer stage failed; citations unavailable in fallback brief.",
            ),
        )


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
    return _generate_hierarchical_brief_artifacts(_build_bundle(resolution)).brief


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

    artifacts = _generate_hierarchical_brief_artifacts(_build_bundle(resolution))
    _store_cached_brief(artifacts.brief, resolution, run_id=cache_run_id)
    _store_cached_brief_support(
        artifacts.support,
        artifacts.brief,
        resolution,
        run_id=cache_run_id,
    )
    return artifacts.brief


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


def get_brief_support(
    topic: str,
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
    run_id: str | None = None,
    prefer_cluster: bool = True,
    require_cluster: bool = False,
) -> dict[str, object]:
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
    cached_support = _get_cached_brief_support_for_resolution(
        resolution,
        run_id=cache_run_id,
    )
    if cached_support is not None:
        return cached_support

    artifacts = _generate_hierarchical_brief_artifacts(_build_bundle(resolution))
    _store_cached_brief(artifacts.brief, resolution, run_id=cache_run_id)
    _store_cached_brief_support(
        artifacts.support,
        artifacts.brief,
        resolution,
        run_id=cache_run_id,
    )
    return dict(artifacts.support)
