from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request

from config.settings import (
    CLUSTER_SIGNAL_MODEL,
    CLUSTER_SIGNAL_OLLAMA_URL,
    CLUSTER_SIGNAL_PROVIDER,
    CLUSTER_SIGNAL_TIMEOUT_SECONDS,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


VALID_STABLE_FRAMES = {"fear", "opportunity", "conflict", "neutral"}
VALID_FRAME_DETAILS = {
    "fear",
    "opportunity",
    "conflict",
    "neutral",
    "human_interest",
    "economic_impact",
    "policy",
    "moral_concern",
    "uncertainty",
}
FRAME_DETAIL_ALIASES = {
    "alert": "fear",
    "anxiety": "fear",
    "concern": "fear",
    "fear": "fear",
    "risk": "fear",
    "threat": "fear",
    "benefit": "opportunity",
    "growth": "opportunity",
    "opportunity": "opportunity",
    "solution": "opportunity",
    "controversy": "conflict",
    "conflict": "conflict",
    "debate": "conflict",
    "dispute": "conflict",
    "economic": "economic_impact",
    "economic_impact": "economic_impact",
    "explanatory": "neutral",
    "human": "human_interest",
    "human_interest": "human_interest",
    "informational": "neutral",
    "mixed": "neutral",
    "moral": "moral_concern",
    "moral_concern": "moral_concern",
    "neutral": "neutral",
    "policy": "policy",
    "regulation": "policy",
    "regulatory": "policy",
    "uncertainty": "uncertainty",
    "unclear": "uncertainty",
    "unknown": "uncertainty",
}
FRAME_DETAIL_TO_STABLE = {
    "fear": "fear",
    "opportunity": "opportunity",
    "conflict": "conflict",
    "neutral": "neutral",
    "human_interest": "neutral",
    "economic_impact": "neutral",
    "policy": "neutral",
    "moral_concern": "fear",
    "uncertainty": "fear",
}
DEFAULT_CLUSTER_SIGNAL = {
    "topic_label": "",
    "concern_level": 0.5,
    "purchase_intent": 0.0,
    "avoidance_signals": 0.0,
    "sentiment": 0.0,
    "dominant_frame": "neutral",
    "frame_detail": "neutral",
    "seg_young_urban": 0.25,
    "seg_family": 0.25,
    "seg_senior": 0.25,
    "seg_b2b": 0.25,
    "evidence": [],
}
SEGMENT_KEYS = ["seg_young_urban", "seg_family", "seg_senior", "seg_b2b"]
PROMPT_TEMPLATE = """
You are a media intelligence analyst. You receive representative articles from the same story cluster.
Extract one conservative, cluster-level behavioral signal. Only use evidence that is explicitly present.

Return ONLY a JSON object with this shape:
{
  "topic_label": "<short descriptive label, max 8 words>",
  "concern_level": 0.0,
  "purchase_intent": 0.0,
  "avoidance_signals": 0.0,
  "sentiment": 0.0,
  "dominant_frame": "<one of: fear|opportunity|conflict|neutral>",
  "frame_detail": "<one of: fear|opportunity|conflict|neutral|human_interest|economic_impact|policy|moral_concern|uncertainty>",
  "seg_young_urban": 0.0,
  "seg_family": 0.0,
  "seg_senior": 0.0,
  "seg_b2b": 0.0,
  "evidence": [
    "<short quote or tight paraphrase grounded in the supplied articles>",
    "<second grounded quote or paraphrase if useful>"
  ]
}

Rules:
- concern_level, purchase_intent, avoidance_signals must be between 0.0 and 1.0.
- sentiment must be between -1.0 and 1.0.
- seg_* fields should sum to approximately 1.0.
- If uncertain, use dominant_frame="neutral" and frame_detail="neutral".
- Keep evidence short and grounded in the supplied text.

Cluster exemplars:
{articles}
""".strip()


def get_cluster_signal_provider() -> str:
    return CLUSTER_SIGNAL_PROVIDER


def get_cluster_signal_model() -> str:
    return CLUSTER_SIGNAL_MODEL


def _clamp_score(
    value: object,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


def _clean_topic_label(value: object, fallback: str) -> str:
    if isinstance(value, str):
        collapsed = " ".join(value.split()).strip()
    else:
        collapsed = ""
    if not collapsed:
        collapsed = fallback
    words = collapsed.split()
    if len(words) > 8:
        collapsed = " ".join(words[:8])
    return collapsed[:80].strip() or fallback


def _canonicalize_frame_detail(value: object) -> str:
    if not value:
        return "neutral"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    candidate = FRAME_DETAIL_ALIASES.get(normalized, normalized)
    if candidate in VALID_FRAME_DETAILS:
        return candidate
    return "neutral"


def _stable_frame(value: object, detail: str) -> str:
    if value:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
        if normalized in VALID_STABLE_FRAMES:
            return normalized
    return FRAME_DETAIL_TO_STABLE.get(detail, "neutral")


def _clean_evidence(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value]
    else:
        items = []

    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        collapsed = " ".join(item.split()).strip()
        if not collapsed:
            continue
        cleaned.append(collapsed[:280])
        if len(cleaned) == 2:
            break
    return cleaned


def _softmax_segments(signals: dict[str, object]) -> dict[str, object]:
    raw = [_clamp_score(signals.get(key), 0.25) for key in SEGMENT_KEYS]
    exps = [math.exp(value * 5) for value in raw]
    total = sum(exps)
    if total <= 0:
        normalized = [0.25, 0.25, 0.25, 0.25]
    else:
        normalized = [value / total for value in exps]
    for key, value in zip(SEGMENT_KEYS, normalized):
        signals[key] = round(value, 4)
    return signals


def _normalize_cluster_signal(
    result: dict | None,
    *,
    fallback_topic_label: str,
) -> dict[str, object]:
    normalized = dict(DEFAULT_CLUSTER_SIGNAL)
    detail = "neutral"
    if isinstance(result, dict):
        detail = _canonicalize_frame_detail(
            result.get("frame_detail", result.get("dominant_frame"))
        )
        normalized.update(
            {
                "topic_label": _clean_topic_label(
                    result.get("topic_label"),
                    fallback_topic_label,
                ),
                "concern_level": _clamp_score(result.get("concern_level"), 0.5),
                "purchase_intent": _clamp_score(result.get("purchase_intent"), 0.0),
                "avoidance_signals": _clamp_score(
                    result.get("avoidance_signals"),
                    0.0,
                ),
                "sentiment": _clamp_score(
                    result.get("sentiment"),
                    0.0,
                    minimum=-1.0,
                    maximum=1.0,
                ),
                "dominant_frame": _stable_frame(result.get("dominant_frame"), detail),
                "frame_detail": detail,
                "evidence": _clean_evidence(result.get("evidence")),
            }
        )
        for key in SEGMENT_KEYS:
            normalized[key] = _clamp_score(result.get(key), 0.25)
    else:
        normalized["topic_label"] = fallback_topic_label

    _softmax_segments(normalized)
    return normalized


def _format_articles(exemplars: list[dict[str, object]]) -> str:
    blocks: list[str] = []
    for index, exemplar in enumerate(exemplars, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] ARTICLE_ID: {exemplar.get('article_id', '')}",
                    f"OUTLET: {exemplar.get('outlet', '')}",
                    f"TITLE: {exemplar.get('title', '')}",
                    f"SUMMARY: {exemplar.get('summary', '')}",
                    f"BODY_EXCERPT: {exemplar.get('body_excerpt', '')}",
                ]
            ).strip()
        )
    return "\n---\n".join(blocks)


def _fallback_topic_label(exemplars: list[dict[str, object]]) -> str:
    for exemplar in exemplars:
        title = _clean_topic_label(exemplar.get("title"), "")
        if title:
            return title
    return "Story cluster"


@retry(
    retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _ollama_request(prompt: str) -> dict:
    payload = json.dumps(
        {
            "model": CLUSTER_SIGNAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 1024},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        CLUSTER_SIGNAL_OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=CLUSTER_SIGNAL_TIMEOUT_SECONDS) as response:
        raw = json.loads(response.read().decode("utf-8")).get("response", "").strip()

    if not raw:
        raise RuntimeError(f"Empty response from Ollama model {CLUSTER_SIGNAL_MODEL}.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON in JSON mode.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Cluster signal provider returned a non-object JSON payload.")
    return parsed


def extract_cluster_signals(exemplars: list[dict[str, object]]) -> dict[str, object]:
    if not exemplars:
        raise ValueError("At least one exemplar article is required.")

    fallback_topic_label = _fallback_topic_label(exemplars)
    prompt = PROMPT_TEMPLATE.format(articles=_format_articles(exemplars))

    if CLUSTER_SIGNAL_PROVIDER != "ollama":
        raise ValueError(
            f"Unsupported cluster signal provider '{CLUSTER_SIGNAL_PROVIDER}'."
        )

    raw_result = _ollama_request(prompt)
    normalized = _normalize_cluster_signal(
        raw_result,
        fallback_topic_label=fallback_topic_label,
    )
    normalized["raw_json"] = raw_result
    return normalized
