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
from db.queries import topic_filter_sql
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
_INSUFFICIENT_DATA_SEGMENTS = ("young_urban", "family", "senior")


def _copy_brief(brief: ResearchBrief, **updates: object) -> ResearchBrief:
    if hasattr(brief, "model_copy"):
        return brief.model_copy(update=updates)
    return brief.copy(update=updates)


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
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT s.article_id, a.title, a.summary, {seg_col} AS relevance, s.extracted_at
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE {seg_col} > 0.3
        {topic_sql}
    """
    params: list[object] = list(topic_params)
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


def _topic_label(topic: str) -> str:
    if topic == "_all":
        return "all monitored topics"
    if topic:
        return topic
    return "this topic"


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


def _fallback_brief(
    topic: str,
    generated_at: str,
    drift: list[dict],
    error: Exception,
    status: str,
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
        status=status,
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

    return _copy_brief(brief, narrative=narrative)


def clear_brief_cache() -> None:
    with _cache_lock:
        _brief_cache.clear()


def generate_brief(topic: str) -> ResearchBrief:
    """Run the drift-to-brief pipeline and return a validated brief."""
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(real_topic)
    generated_at = datetime.now(timezone.utc).isoformat()
    brief_status = _brief_status_from_drift(drift)

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
    if brief_status == "insufficient_data":
        return _copy_brief(
            _insufficient_data_brief(topic, generated_at),
            confidence_context=confidence_context,
        )

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
    entities_by_article = _load_entities_for_articles(
        [article["article_id"] for article in top_articles]
    )
    for article in top_articles:
        article["entities"] = entities_by_article.get(article["article_id"], [])

    context_block = build_context_block(drift, top_articles, confidence_context)
    domain = drift[0].get("domain", "generic") if drift else "generic"
    relevant_fields = drift[0].get("relevant_fields", []) if drift else []
    data: dict | None = None
    try:
        data = _call_ollama_json(
            BRIEF_TEMPLATE.format(
                topic=topic,
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
        data["topic"] = topic
        data["status"] = brief_status
        logger.debug("[brief] normalized payload topic=%s data=%s", topic, data)
        brief = _apply_confidence_language(ResearchBrief(**data), drift)
        return _copy_brief(brief, confidence_context=confidence_context)
    except ValidationError as exc:
        logger.warning("[brief] Validation failed topic=%s error=%s data=%s", topic, exc, data)
        brief = _apply_confidence_language(
            _fallback_brief(topic, generated_at, drift, exc, brief_status),
            drift,
        )
        return _copy_brief(brief, confidence_context=confidence_context)
    except (RuntimeError, ValueError) as exc:
        logger.warning("[brief] Model output unusable topic=%s error=%s", topic, exc)
        brief = _apply_confidence_language(
            _fallback_brief(topic, generated_at, drift, exc, brief_status),
            drift,
        )
        return _copy_brief(brief, confidence_context=confidence_context)


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
