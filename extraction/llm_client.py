import json
import math
import urllib.error
import urllib.request

from config.domains import DOMAIN_SIGNAL_KEYS, get_domain_config, topic_to_domain
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_FALLBACK_MODEL = "gemma3:1b"
OLLAMA_TIMEOUT_SECONDS = 120
VALID_FRAMES = {"fear", "opportunity", "conflict", "neutral"}
AFFINITY_PRIORS = {
    "mainstream": {
        "seg_young_urban": 0.30,
        "seg_family": 0.30,
        "seg_senior": 0.25,
        "seg_b2b": 0.15,
    },
    "young_urban": {
        "seg_young_urban": 0.60,
        "seg_family": 0.15,
        "seg_senior": 0.10,
        "seg_b2b": 0.15,
    },
    "family": {
        "seg_young_urban": 0.10,
        "seg_family": 0.60,
        "seg_senior": 0.20,
        "seg_b2b": 0.10,
    },
    "b2b": {
        "seg_young_urban": 0.15,
        "seg_family": 0.10,
        "seg_senior": 0.10,
        "seg_b2b": 0.65,
    },
    "senior": {
        "seg_young_urban": 0.10,
        "seg_family": 0.20,
        "seg_senior": 0.60,
        "seg_b2b": 0.10,
    },
}
PROMPT_TEMPLATE = """
You are a behavioral analyst specializing in Czech media. Analyze the article below.

Topic: {topic}
Article title: {title}
Article summary: {summary}

Domain guidance: {domain_hint}

Output ONLY this JSON object (nothing before or after it):
{{
  "concern_level": 0.0,
  "purchase_intent": 0.0,
  "avoidance_signals": 0.0,
  "dominant_frame": "<one of: fear|opportunity|conflict|neutral>",
  "seg_young_urban": 0.0,
  "seg_family": 0.0,
  "seg_senior": 0.0,
  "seg_b2b": 0.0
}}

Rules:
- All numeric fields are floats between 0.0 and 1.0.
- seg_* fields represent the relative relevance to each audience segment and should sum to approximately 1.0.
- dominant_frame must be exactly one of: fear, opportunity, conflict, neutral.
""".strip()

DEFAULT_SIGNALS = {
    "concern_level": 0.5,
    "purchase_intent": 0.0,
    "avoidance_signals": 0.0,
    "dominant_frame": "neutral",
    "seg_young_urban": 0.25,
    "seg_family": 0.25,
    "seg_senior": 0.25,
    "seg_b2b": 0.25,
}


def _softmax_segments(signals: dict) -> dict:
    """Normalize seg_* fields so they sum to 1.0 using softmax."""
    seg_keys = ["seg_young_urban", "seg_family", "seg_senior", "seg_b2b"]
    raw = [signals.get(key, 0.25) for key in seg_keys]
    exps = [math.exp(value * 5) for value in raw]
    total = sum(exps)
    normed = [value / total for value in exps]
    for key, value in zip(seg_keys, normed):
        signals[key] = round(value, 4)
    return signals


def _apply_affinity_prior(signals: dict, affinity_tag: str) -> dict:
    """Blend LLM segment scores with the feed's affinity prior."""
    prior = AFFINITY_PRIORS.get(affinity_tag)
    if prior is None:
        return signals

    for key in ["seg_young_urban", "seg_family", "seg_senior", "seg_b2b"]:
        signals[key] = round(0.7 * signals.get(key, 0.25) + 0.3 * prior[key], 4)
    return signals


def _clamp_score(value: object, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_signals(result: dict | None) -> dict:
    normalized = dict(DEFAULT_SIGNALS)
    if isinstance(result, dict):
        for key, default in DEFAULT_SIGNALS.items():
            if key == "dominant_frame":
                frame = str(result.get(key, default)).strip().lower()
                normalized[key] = frame if frame in VALID_FRAMES else default
            else:
                normalized[key] = _clamp_score(result.get(key), default)
    return normalized


def _apply_domain_mask(signals: dict, domain: str) -> dict:
    """Zero out signal fields that are not relevant to the resolved domain."""
    config = get_domain_config(domain)
    relevant = set(config["relevant_fields"])
    irrelevant: list[str] = []
    for key in DOMAIN_SIGNAL_KEYS:
        if key not in relevant:
            signals[key] = 0.0
            irrelevant.append(key)
    signals["domain"] = domain
    signals["irrelevant_fields"] = irrelevant
    return signals


@retry(
    retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False,
)
def _ollama_request(payload: bytes) -> dict | None:
    """Single Ollama HTTP call retried on transient transport failures."""
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        raw = json.loads(response.read().decode("utf-8")).get("response", "").strip()

    if not raw:
        return None
    return json.loads(raw)


def _try_ollama(
    title: str,
    summary: str,
    model: str,
    topic: str,
    domain_hint: str,
) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        topic=topic,
        title=title,
        summary=summary,
        domain_hint=domain_hint,
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    try:
        return _ollama_request(payload)
    except Exception:
        return None


def _ollama_fallback(title: str, summary: str, topic: str, domain_hint: str) -> dict | None:
    """Fallback to a second Ollama model to keep inference GPU-backed."""
    return _try_ollama(title, summary, OLLAMA_FALLBACK_MODEL, topic, domain_hint)


def extract_signals(
    title: str,
    summary: str,
    affinity_tag: str = "mainstream",
    topic: str = "",
) -> dict:
    domain = topic_to_domain(topic)
    domain_hint = str(get_domain_config(domain)["prompt_hint"])
    result = _try_ollama(title, summary, OLLAMA_MODEL, topic, domain_hint)
    if result is None:
        result = _ollama_fallback(title, summary, topic, domain_hint)
    signals = _normalize_signals(result)
    signals = _softmax_segments(signals)
    signals = _apply_affinity_prior(signals, affinity_tag)
    signals = _apply_domain_mask(signals, domain)
    return signals
