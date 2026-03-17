import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone

from brief.models import ResearchBrief
from brief.prompt import BRIEF_TEMPLATE, build_context_block
from db.init import get_conn
from delta.engine import compute_drift


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 60
BRIEF_CACHE_TTL = timedelta(minutes=30)

_brief_cache: dict[str, tuple[ResearchBrief, datetime]] = {}
_cache_lock = threading.Lock()


def _call_ollama_json(prompt: str) -> dict:
    """Call local Ollama in JSON mode and return the parsed response object."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 512},
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

    context_block = build_context_block(drift, top_articles)
    data = _call_ollama_json(
        BRIEF_TEMPLATE.format(
            topic=topic,
            date=generated_at,
            context_block=context_block,
            model=OLLAMA_MODEL,
        )
    )
    data["generated_at"] = generated_at
    data["model_used"] = OLLAMA_MODEL
    data["topic"] = topic
    return ResearchBrief(**data)


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
