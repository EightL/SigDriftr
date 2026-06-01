from __future__ import annotations

import json
import logging
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 60

logger = logging.getLogger(__name__)


def call_ollama_json(prompt: str) -> dict:
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
