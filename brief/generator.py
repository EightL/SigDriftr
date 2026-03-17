import json
import re
import urllib.request
from datetime import datetime, timezone

from brief.models import ResearchBrief
from brief.prompt import PASS1_TEMPLATE, PASS2_TEMPLATE, build_context_block
from db.init import get_conn
from delta.engine import compute_drift


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 60


def _call_ollama(prompt: str) -> str:
    """Call local Ollama and return the raw text response."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
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
    return raw


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


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from model output, stripping code fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON in LLM output")

def generate_brief(topic: str) -> ResearchBrief:
    """Run the two-pass drift-to-brief pipeline and return a validated brief."""
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

    reasoning = _call_ollama(
        PASS1_TEMPLATE.format(
            topic=topic,
            date=generated_at,
            context_block=context_block,
        )
    )
    raw_json_text = _call_ollama(
        PASS2_TEMPLATE.format(
            topic=topic,
            context_block=context_block,
            reasoning=reasoning[:1500],
            date=generated_at,
            model=OLLAMA_MODEL,
        )
    )
    data = _extract_json(raw_json_text)
    data["generated_at"] = generated_at
    data["model_used"] = OLLAMA_MODEL
    data["topic"] = topic
    return ResearchBrief(**data)
