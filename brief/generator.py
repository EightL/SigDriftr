import json
import logging
import re
import threading
import urllib.request
from datetime import datetime, timedelta, timezone

from brief.models import BriefConfidenceContext, ResearchBrief
from brief.prompt import (
    BRIEF_TEMPLATE,
    LOW_CONFIDENCE_WARNING,
    build_context_block,
    confidence_label,
)
from config.settings import MIN_BRIEF_CONFIDENCE
from db.init import get_conn
from delta.engine import compute_drift
from pydantic import ValidationError


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 60
BRIEF_CACHE_TTL = timedelta(minutes=30)
logger = logging.getLogger(__name__)

# This cache is intentionally process-local for the single-worker demo setup.
_brief_cache: dict[str, tuple[ResearchBrief, datetime]] = {}
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


def _get_top_articles(topic: str, segment: str, limit: int = 2) -> list[dict]:
    """Fetch recent high-signal articles for a topic and segment."""
    conn = get_conn()
    seg_col = f"s.seg_{segment}"
    query = f"""
        SELECT a.title, a.summary, {seg_col} AS relevance, s.extracted_at
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE {seg_col} > 0.3
    """
    params: list[object] = []
    if topic:
        query += " AND a.topic = ?"
        params.append(topic)
    query += " ORDER BY s.extracted_at DESC, relevance DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [
        {
            "title": row[0],
            "summary": row[1],
            "relevance": row[2],
            "extracted_at": row[3],
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


def _fallback_brief(
    topic: str, generated_at: str, drift: list[dict], error: Exception
) -> ResearchBrief:
    ranked_segments = sorted(
        drift,
        key=lambda item: (
            item.get("drift_magnitude", 0.0),
            item.get("article_count", 0),
        ),
        reverse=True,
    )
    top_segment = ranked_segments[0]["segment"] if ranked_segments else "young_urban"
    top_entry = ranked_segments[0] if ranked_segments else {}
    strongest_alert = top_entry.get("alert_level", "none")
    if strongest_alert not in ("none", "mild", "strong"):
        strongest_alert = "none"

    deltas = top_entry.get("deltas", {})
    signal_key = max(
        ("concern_level", "purchase_intent", "avoidance_signals"),
        key=lambda key: abs(deltas.get(key, 0.0)),
        default="concern_level",
    )
    signal_value = deltas.get(signal_key, 0.0)
    display_name = _SEGMENT_DISPLAY_NAMES.get(top_segment, top_segment.replace("_", " ").title())

    if signal_key == "concern_level" and signal_value > 0:
        drift_type = "concern_spike"
        signal_phrase = "Concern Spike"
    elif signal_key == "purchase_intent" and signal_value > 0:
        drift_type = "purchase_surge"
        signal_phrase = "Purchase Intent Surge"
    elif signal_key == "avoidance_signals" and signal_value > 0:
        drift_type = "avoidance_rise"
        signal_phrase = "Avoidance Rise"
    elif top_entry.get("frame_shift"):
        drift_type = "frame_shift"
        signal_phrase = "Frame Shift"
    else:
        drift_type = "mixed"
        signal_phrase = "Behavioral Drift"

    hypothesis_segments = []
    for entry in ranked_segments:
        segment = entry.get("segment")
        if segment in ("young_urban", "family", "senior", "b2b") and segment not in hypothesis_segments:
            hypothesis_segments.append(segment)
        if len(hypothesis_segments) == 3:
            break
    for segment in ("young_urban", "family", "senior", "b2b"):
        if segment not in hypothesis_segments:
            hypothesis_segments.append(segment)
        if len(hypothesis_segments) == 3:
            break

    hint = str(error).splitlines()[0][:120]
    return ResearchBrief(
        topic=topic,
        headline=f"{display_name} Segment Shows {signal_phrase}",
        narrative=(
            f"Computed drift results show the strongest movement in the {display_name.lower()} segment, "
            f"with {signal_phrase.lower()} standing out in the current calibration window. "
            f"This summary was generated from validated drift data after the model output failed validation: {hint}"
        ),
        most_affected_segment=top_segment,
        drift_type=drift_type,
        alert_level=strongest_alert,
        hypotheses=[
            {
                "segment": hypothesis_segments[0],
                "hypothesis": "The most shifted segment will show measurable behavior change in follow-up research.",
                "signal_basis": "Derived from highest drift_magnitude in current calibration window.",
                "suggested_question": "How much has recent coverage changed your near-term behavior?",
            },
            {
                "segment": hypothesis_segments[1],
                "hypothesis": "The second-most affected segment will show a measurable shift if current coverage patterns persist.",
                "signal_basis": "Fallback hypothesis used because the model response was invalid.",
                "suggested_question": "How much has recent coverage changed your expected behavior this month?",
            },
            {
                "segment": hypothesis_segments[2],
                "hypothesis": "The third-most affected segment will show directional movement if uncertainty persists.",
                "signal_basis": "Fallback hypothesis used because the model response was invalid.",
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
        segment_entry = max(
            drift,
            key=lambda item: (
                item.get("drift_magnitude", 0.0),
                item.get("article_count", 0),
            ),
        )

    narrative = brief.narrative.strip()
    if segment_entry is not None:
        qualifier = confidence_label(segment_entry.get("confidence", 0.0))
        prefix = (
            "Exploratory finding."
            if qualifier == "exploratory finding"
            else f"{qualifier.capitalize()} finding."
        )
        if qualifier not in narrative.lower():
            narrative = f"{prefix} {narrative}".strip()

    if drift and all(
        entry.get("confidence", 0.0) < MIN_BRIEF_CONFIDENCE for entry in drift
    ) and LOW_CONFIDENCE_WARNING not in narrative:
        narrative = f"{LOW_CONFIDENCE_WARNING} {narrative}".strip()

    if hasattr(brief, "model_copy"):
        return brief.model_copy(update={"narrative": narrative})
    return brief.copy(update={"narrative": narrative})


def clear_brief_cache() -> None:
    with _cache_lock:
        _brief_cache.clear()


def generate_brief(topic: str) -> ResearchBrief:
    """Run the drift-to-brief pipeline and return a validated brief."""
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(real_topic)
    generated_at = datetime.now(timezone.utc).isoformat()

    ranked_segments = sorted(
        drift,
        key=lambda item: (
            item.get("drift_magnitude", 0.0),
            item.get("article_count", 0),
        ),
        reverse=True,
    )

    top_articles: list[dict] = []
    for entry in ranked_segments:
        top_articles.extend(_get_top_articles(real_topic, entry["segment"], limit=2))

    confidence_context = BriefConfidenceContext(
        segment_confidence={
            entry["segment"]: entry.get("confidence", 0.0) for entry in drift
        },
        baseline_is_learned={
            entry["segment"]: entry.get("baseline_is_learned", False) for entry in drift
        },
        baseline_sample_count={
            entry["segment"]: entry.get("baseline_sample_count", 0) for entry in drift
        },
    )
    context_block = build_context_block(drift, top_articles, confidence_context)
    data: dict | None = None
    try:
        data = _call_ollama_json(
            BRIEF_TEMPLATE.format(
                topic=topic,
                date=generated_at,
                context_block=context_block,
                model=OLLAMA_MODEL,
            )
        )
        data = _normalize_brief_payload(data)
        data["generated_at"] = generated_at
        data["model_used"] = OLLAMA_MODEL
        data["topic"] = topic
        logger.debug("[brief] normalized payload topic=%s data=%s", topic, data)
        return _apply_confidence_language(ResearchBrief(**data), drift)
    except ValidationError as exc:
        logger.warning("[brief] Validation failed topic=%s error=%s data=%s", topic, exc, data)
        return _apply_confidence_language(
            _fallback_brief(topic, generated_at, drift, exc),
            drift,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("[brief] Model output unusable topic=%s error=%s", topic, exc)
        return _apply_confidence_language(
            _fallback_brief(topic, generated_at, drift, exc),
            drift,
        )


def generate_brief_cached(topic: str) -> ResearchBrief:
    """Return a cached brief when it is still fresh enough for demo use."""
    now = datetime.now(timezone.utc)
    with _cache_lock:
        cached = _brief_cache.get(topic)
        if cached and now - cached[1] < BRIEF_CACHE_TTL:
            return cached[0]

    brief = generate_brief(topic)

    with _cache_lock:
        _brief_cache[topic] = (brief, datetime.now(timezone.utc))
    return brief
