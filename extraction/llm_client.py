import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from config.domains import DOMAIN_SIGNAL_KEYS, get_domain_config
from config.topics import domain_for_topic

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ModuleNotFoundError:

    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def retry_if_exception_type(*args, **kwargs):
        return None

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(*args, **kwargs):
        return False


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GOOGLE_GEMMA_API_KEY = (
    os.environ.get("GOOGLE_GEMMA_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or ""
).strip()
GOOGLE_GEMMA_MODEL = os.environ.get("GOOGLE_GEMMA_MODEL", "gemma-4-31b-it").strip()
GOOGLE_GEMMA_URL_TEMPLATE = os.environ.get(
    "GOOGLE_GEMMA_URL_TEMPLATE",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
).strip()
GOOGLE_GEMMA_TIMEOUT_SECONDS = int(os.environ.get("GOOGLE_GEMMA_TIMEOUT_SECONDS", "120"))
LLM_PROVIDER = os.environ.get(
    "SIGDRIFTR_LLM_PROVIDER",
    "google" if GOOGLE_GEMMA_API_KEY else "ollama",
).strip().lower()
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_FALLBACK_MODEL = "gemma3:1b"
OLLAMA_TIMEOUT_SECONDS = 120
VALID_FRAMES = {"fear", "opportunity", "conflict", "neutral"}
ARTICLE_BODY_PROMPT_CHARS = 3000
SEGMENT_KEYS = ["seg_young_urban", "seg_family", "seg_senior", "seg_b2b"]
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
Article body excerpt: {body}

Domain guidance: {domain_hint}

Output ONLY this JSON object (nothing before or after it):
{{
  "concern_level": 0.0,
  "purchase_intent": 0.0,
  "avoidance_signals": 0.0,
  "dominant_frame": "<one of: fear|opportunity|conflict|neutral>",
  "seg_young_urban_relevance": 0.0,
  "seg_family_relevance": 0.0,
  "seg_senior_relevance": 0.0,
  "seg_b2b_relevance": 0.0
}}

Rules:
- All numeric fields are floats between 0.0 and 1.0.
- seg_*_relevance fields are independent relevance scores. They do not need to sum to 1.0.
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
    "seg_young_urban_relevance": 0.25,
    "seg_family_relevance": 0.25,
    "seg_senior_relevance": 0.25,
    "seg_b2b_relevance": 0.25,
}


def _softmax_segments(signals: dict) -> dict:
    """Derive aggregation shares from independent segment relevance scores."""
    raw = [signals.get(f"{key}_relevance", signals.get(key, 0.25)) for key in SEGMENT_KEYS]
    exps = [math.exp(value * 5) for value in raw]
    total = sum(exps)
    normed = [value / total for value in exps]
    for key, value in zip(SEGMENT_KEYS, normed):
        signals[key] = round(value, 4)
        signals[f"{key}_share"] = signals[key]
    return signals


def _apply_affinity_prior(signals: dict, affinity_tag: str) -> dict:
    """Blend derived segment shares with the feed's affinity prior."""
    prior = AFFINITY_PRIORS.get(affinity_tag)
    if prior is None:
        return signals

    for key in SEGMENT_KEYS:
        signals[key] = round(0.7 * signals.get(key, 0.25) + 0.3 * prior[key], 4)
        signals[f"{key}_share"] = signals[key]
    return signals


def _clamp_score(value: object, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_signals(result: dict | None) -> dict:
    normalized = dict(DEFAULT_SIGNALS)
    if isinstance(result, dict):
        for key in ["concern_level", "purchase_intent", "avoidance_signals"]:
            normalized[key] = _clamp_score(result.get(key), DEFAULT_SIGNALS[key])

        frame = str(result.get("dominant_frame", DEFAULT_SIGNALS["dominant_frame"])).strip().lower()
        normalized["dominant_frame"] = (
            frame if frame in VALID_FRAMES else DEFAULT_SIGNALS["dominant_frame"]
        )

        for key in SEGMENT_KEYS:
            relevance_key = f"{key}_relevance"
            normalized[relevance_key] = _clamp_score(
                result.get(relevance_key, result.get(key)),
                DEFAULT_SIGNALS[relevance_key],
            )
    return normalized


def _body_excerpt(body: str) -> str:
    collapsed = " ".join((body or "").split())
    if not collapsed:
        return ""
    return collapsed[:ARTICLE_BODY_PROMPT_CHARS]


def _build_prompt(
    title: str,
    summary: str,
    topic: str,
    domain_hint: str,
    body: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        topic=topic,
        title=title,
        summary=summary,
        body=_body_excerpt(body),
        domain_hint=domain_hint,
    )


def _parse_json_text(raw: str) -> dict | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


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
def _google_gemma_request(prompt: str) -> dict | None:
    """Single Google Gemma generateContent call retried on transient failures."""
    if not GOOGLE_GEMMA_API_KEY:
        return None

    base_url = GOOGLE_GEMMA_URL_TEMPLATE.format(
        model=urllib.parse.quote(GOOGLE_GEMMA_MODEL, safe="")
    )
    url = f"{base_url}?{urllib.parse.urlencode({'key': GOOGLE_GEMMA_API_KEY})}"
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=GOOGLE_GEMMA_TIMEOUT_SECONDS) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    parts = (
        response_payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    raw = "".join(
        str(part.get("text", ""))
        for part in parts
        if not part.get("thought")
    ).strip()
    if not raw:
        raw = "".join(str(part.get("text", "")) for part in parts).strip()
    return _parse_json_text(raw)


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
    return _parse_json_text(raw)


def _try_google_gemma(
    title: str,
    summary: str,
    topic: str,
    domain_hint: str,
    body: str = "",
) -> dict | None:
    prompt = _build_prompt(title, summary, topic, domain_hint, body=body)
    try:
        return _google_gemma_request(prompt)
    except Exception:
        return None


def _try_ollama(
    title: str,
    summary: str,
    model: str,
    topic: str,
    domain_hint: str,
    body: str = "",
) -> dict | None:
    prompt = _build_prompt(title, summary, topic, domain_hint, body=body)
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    try:
        return _ollama_request(payload)
    except Exception:
        return None


def _ollama_fallback(
    title: str,
    summary: str,
    topic: str,
    domain_hint: str,
    body: str = "",
) -> dict | None:
    """Fallback to a second Ollama model to keep inference GPU-backed."""
    return _try_ollama(title, summary, OLLAMA_FALLBACK_MODEL, topic, domain_hint, body=body)


def extract_signals(
    title: str,
    summary: str,
    affinity_tag: str = "mainstream",
    topic: str = "",
    body: str = "",
) -> dict:
    domain = domain_for_topic(topic)
    domain_hint = str(get_domain_config(domain)["prompt_hint"])
    result = None
    provider = LLM_PROVIDER
    model = ""

    if provider in {"google", "google_gemma", "gemma"}:
        result = _try_google_gemma(title, summary, topic, domain_hint, body=body)
        if result is not None:
            model = GOOGLE_GEMMA_MODEL
        else:
            provider = "ollama"

    if result is None:
        result = _try_ollama(title, summary, OLLAMA_MODEL, topic, domain_hint, body=body)
        model = OLLAMA_MODEL
    if result is None:
        result = _ollama_fallback(title, summary, topic, domain_hint, body=body)
        model = OLLAMA_FALLBACK_MODEL
    signals = _normalize_signals(result)
    signals = _softmax_segments(signals)
    signals = _apply_affinity_prior(signals, affinity_tag)
    signals = _apply_domain_mask(signals, domain)
    signals["extractor_provider"] = provider
    signals["extractor_model"] = model
    signals["schema_version"] = "article-signal-v2"
    signals["input_text"] = {
        "title_chars": len(title or ""),
        "summary_chars": len(summary or ""),
        "body_chars": len(body or ""),
        "body_prompt_chars": len(_body_excerpt(body)),
        "body_truncated": len(" ".join((body or "").split())) > ARTICLE_BODY_PROMPT_CHARS,
    }
    return signals
