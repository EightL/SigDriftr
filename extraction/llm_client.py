import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 120
VALID_FRAMES = {"fear", "opportunity", "conflict", "neutral"}
PROMPT_TEMPLATE = """
You are a behavioral analyst. Given a Czech news article, output ONLY valid JSON.

Article title: {title}
Article summary: {summary}

Output JSON with exactly these keys (all floats 0.0-1.0 unless noted):
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


def _try_ollama(title: str, summary: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=OLLAMA_TIMEOUT_SECONDS
        ) as response:
            raw = json.loads(response.read().decode("utf-8")).get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _try_hf_fallback(title: str, summary: str) -> dict:
    raise RuntimeError(
        "Ollama extraction failed and the Hugging Face fallback is disabled."
    )


def extract_signals(title: str, summary: str) -> dict:
    result = _try_ollama(title, summary)
    if result is None:
        result = _try_hf_fallback(title, summary)
    return _normalize_signals(result)
