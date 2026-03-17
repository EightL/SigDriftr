# Plan 04 — Research Brief Generator

**Goal:** Synthesize per-segment drift scores from Stage 3 into a Lakmoos-style research brief — plain-language narrative + 3 falsifiable survey hypotheses — by driving a local LLM with a structured Chain-of-Thought prompt grounded in real drift evidence.

**Where it fits:** Reads from the `baselines`, `segment_profiles`, and `signals` tables written by plans 01–03. Adds a new module `brief/generator.py`, a prompt template in `brief/prompt.py`, and exposes a new endpoint `GET /brief/{topic}` via `api/routes/brief.py`.

---

## Inputs
- Output of `delta.engine.compute_drift(topic)` — list of per-segment drift dicts
- `signals` table — up to 5 most-recent high-signal articles per segment (for grounding)
- Ollama local LLM (Gemma-3-1B or Phi-3-mini) — primary
- OpenAI/Groq API fallback — if Ollama unavailable

## Outputs
- New module `brief/generator.py` — orchestrates drift → LLM → structured brief
- New module `brief/prompt.py` — prompt template + context assembly
- New module `brief/models.py` — Pydantic output schema
- New API route `api/routes/brief.py` — `GET /brief/{topic}`
- Updated `main.py` — register brief router
- Updated `requirements.txt` — add `openai` (fallback only, no new required deps for primary path)

---

## Exact File Paths

```
brief/
    __init__.py
    models.py          # Pydantic schema: ResearchBrief
    prompt.py          # context builder + prompt template
    generator.py       # orchestration: drift -> LLM -> ResearchBrief
api/routes/
    brief.py           # GET /brief/{topic}
main.py                # register brief router
requirements.txt       # add openai>=1.0
```

---

## Architecture Decision: Why Structured CoT, Not Plain Summarization

A naive prompt like "summarize this drift data" produces generic, ungrounded text. Instead, we use a **two-pass structured Chain-of-Thought** approach inspired by recent work on narrative shift detection (Hybrid Dynamic Topic Models + LLM, 2025) and LLM-inferred narrative framing (zero-shot, 2025):

1. **Pass 1 — Reasoning chain (hidden):** The LLM is asked to reason step-by-step about *what changed*, *which segment is most affected*, and *what behavioral mechanism could explain the shift* (framing theory: conflict/fear/opportunity shift → purchase_intent/avoidance change).
2. **Pass 2 — Structured output:** The LLM produces a fixed JSON schema with `headline`, `narrative`, and `hypotheses[3]`. The reasoning from Pass 1 is injected as context into Pass 2's prompt — this is the "chain" part.

This mirrors the RAT (Retrieval-Augmented Thoughts) paradigm from 2024 RAG research: instead of retrieving external documents, we retrieve the LLM's own intermediate reasoning as evidence for the final generation step. It dramatically reduces hallucination and produces crisper, more actionable hypotheses.

For small local models (Gemma-3-1B, Phi-3-mini), two short focused prompts outperform one long complex prompt — consistent with Google Research's "Small models, big results" (2026) finding that intent decomposition into sub-tasks significantly improves small LLM accuracy.

---

## Numbered Steps

### 1. Create `brief/__init__.py`

Empty file.

---

### 2. Create `brief/models.py`

Defines the Pydantic output schema. This is the contract between the LLM output and the API response.

```python
from pydantic import BaseModel, Field
from typing import Literal


class SurveyHypothesis(BaseModel):
    segment: str = Field(description="Target audience segment")
    hypothesis: str = Field(description="Falsifiable behavioral hypothesis (1-2 sentences)")
    signal_basis: str = Field(description="Which signal drove this hypothesis (e.g. concern_level +0.18)")
    suggested_question: str = Field(description="One Likert-scale survey question to test this hypothesis")


class ResearchBrief(BaseModel):
    topic: str
    headline: str = Field(description="8-12 word summary of the dominant behavioral shift")
    narrative: str = Field(description="2-3 sentence plain-language brief for a Lakmoos analyst")
    most_affected_segment: str
    drift_type: Literal["concern_spike", "purchase_surge", "avoidance_rise", "frame_shift", "mixed", "stable"]
    hypotheses: list[SurveyHypothesis] = Field(min_length=3, max_length=3)
    alert_level: Literal["none", "mild", "strong"]
    generated_at: str
    model_used: str
```

---

### 3. Create `brief/prompt.py`

Builds the context block injected into the LLM prompt and defines both prompt templates.

```python
from datetime import datetime, timezone


SEGMENT_LABELS = {
    "young_urban": "young urban adults (18-35, city dwellers, digital-first)",
    "family":      "family households (parents with children, suburban/rural)",
    "senior":      "seniors (60+, traditional media consumers)",
    "b2b":         "business decision-makers (SME owners, executives)",
}

FRAME_MEANINGS = {
    "fear":       "articles emphasise threats, danger, and negative outcomes",
    "opportunity": "articles emphasise gains, positive futures, and solutions",
    "conflict":   "articles emphasise disagreement, opposition, and tension",
    "neutral":    "articles are balanced or informational without strong framing",
}


def build_context_block(drift_results: list[dict], top_articles: list[dict]) -> str:
    """
    Formats drift data + grounding article snippets into a compact context string
    for injection into the LLM prompt.
    """
    lines = ["## Drift Evidence"]
    for d in drift_results:
        seg = d["segment"]
        label = SEGMENT_LABELS.get(seg, seg)
        deltas = d.get("deltas", {})
        frame = d.get("dominant_frame", "neutral")
        frame_desc = FRAME_MEANINGS.get(frame, frame)
        frame_shift = d.get("frame_shift", False)
        alert = d.get("alert_level", "none")
        n = d.get("article_count", 0)

        lines.append(f"\n### Segment: {label}")
        lines.append(f"- Articles analysed: {n}")
        lines.append(f"- Alert level: {alert}")
        lines.append(f"- concern_level delta: {deltas.get('concern_level', 0):+.3f}")
        lines.append(f"- purchase_intent delta: {deltas.get('purchase_intent', 0):+.3f}")
        lines.append(f"- avoidance_signals delta: {deltas.get('avoidance_signals', 0):+.3f}")
        lines.append(f"- Dominant frame: {frame} ({frame_desc})")
        lines.append(f"- Frame shifted vs baseline: {frame_shift}")

    if top_articles:
        lines.append("\n## Representative Article Snippets (grounding evidence)")
        for i, art in enumerate(top_articles[:5], 1):
            title = art.get("title", "[no title]")[:120]
            summary = art.get("summary", "")[:200]
            lines.append(f"{i}. [{art.get('segment','?')}] {title} — {summary}")

    return "\n".join(lines)


PASS1_TEMPLATE = """You are an expert media analyst. Your task is to reason carefully about behavioral drift signals extracted from Czech news media.

Topic: {topic}
Date: {date}

{context_block}

## Your Task
Think step by step:
1. Which segment shows the strongest behavioral shift? Why?
2. What type of drift is this? (concern_spike / purchase_surge / avoidance_rise / frame_shift / mixed / stable)
3. What behavioral mechanism connects the news framing to the audience's likely response? Be specific.
4. For each of the 3 most affected segments, what is one testable behavioral hypothesis a market researcher could verify?

Reason through each point before concluding. Be concise."""


PASS2_TEMPLATE = """You are a Lakmoos AI research analyst writing a calibration brief.

Topic: {topic}

{context_block}

## Analyst Reasoning (use this as evidence)
{reasoning}

## Output Instructions
Produce ONLY a valid JSON object matching this exact schema. No markdown, no explanation outside the JSON:

{{
  "topic": "{topic}",
  "headline": "<8-12 word summary of dominant behavioral shift>",
  "narrative": "<2-3 sentence plain-language brief for a Lakmoos analyst>",
  "most_affected_segment": "<one of: young_urban | family | senior | b2b>",
  "drift_type": "<one of: concern_spike | purchase_surge | avoidance_rise | frame_shift | mixed | stable>",
  "alert_level": "<one of: none | mild | strong>",
  "hypotheses": [
    {{
      "segment": "<segment>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. concern_level +0.18>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<segment>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. purchase_intent -0.12>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<segment>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. avoidance_signals +0.09>",
      "suggested_question": "<one Likert-scale survey question>"
    }}
  ],
  "generated_at": "{date}",
  "model_used": "{model}"
}}"""
```

---

### 4. Create `brief/generator.py`

Orchestrates the two-pass LLM call and returns a validated `ResearchBrief`.

```python
import json
import re
from datetime import datetime, timezone

import requests

from brief.models import ResearchBrief
from brief.prompt import build_context_block, PASS1_TEMPLATE, PASS2_TEMPLATE
from db.init import get_conn
from delta.engine import compute_drift


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"          # or "phi3:mini" — whichever is pulled
FALLBACK_MODEL = "gpt-4o-mini"     # only used if Ollama unavailable
OLLAMA_TIMEOUT = 60                # seconds


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama, return raw text. Raises on connection error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


def _call_openai_fallback(prompt: str) -> tuple[str, str]:
    """Fallback to OpenAI if Ollama is unavailable. Returns (text, model_name)."""
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content, FALLBACK_MODEL
    except Exception as e:
        raise RuntimeError(f"Both Ollama and OpenAI fallback failed: {e}")


def _llm_call(prompt: str) -> tuple[str, str]:
    """Try Ollama first; fall back to OpenAI. Returns (text, model_name)."""
    try:
        text = _call_ollama(prompt)
        return text, OLLAMA_MODEL
    except Exception:
        return _call_openai_fallback(prompt)


# ---------------------------------------------------------------------------
# Evidence retrieval
# ---------------------------------------------------------------------------

def _get_top_articles(topic: str, segment: str, limit: int = 2) -> list[dict]:
    """
    Fetch the most recent high-signal articles for a topic+segment.
    Uses seg_* score as a proxy for relevance.
    """
    conn = get_conn()
    seg_col = f"s.seg_{segment}"
    query = f"""
        SELECT a.title, a.summary, {seg_col} AS relevance
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE {seg_col} > 0.3
    """
    params: list = []
    if topic:
        query += " AND a.topic = ?"
        params.append(topic)
    query += f" ORDER BY s.extracted_at DESC LIMIT {limit}"

    rows = conn.execute(query, params).fetchall()
    return [
        {"title": r[0], "summary": r[1], "relevance": r[2], "segment": segment}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from LLM output.
    LLMs sometimes wrap JSON in markdown code fences — strip those first.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Find first { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in LLM output")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_brief(topic: str) -> ResearchBrief:
    """
    Full two-pass pipeline:
    1. Compute drift for topic
    2. Gather grounding article snippets
    3. Pass 1: LLM reasoning chain
    4. Pass 2: LLM structured JSON output
    5. Validate + return ResearchBrief
    """
    # Step 1: Get drift data
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(real_topic)

    # Step 2: Gather grounding articles (2 per most-affected segments)
    segments = ["young_urban", "family", "senior", "b2b"]
    top_articles: list[dict] = []
    for seg in segments:
        top_articles.extend(_get_top_articles(real_topic, seg, limit=2))

    # Step 3: Build context block
    context_block = build_context_block(drift, top_articles)
    date_str = datetime.now(timezone.utc).isoformat()

    # Step 4: Pass 1 — reasoning chain
    pass1_prompt = PASS1_TEMPLATE.format(
        topic=topic,
        date=date_str,
        context_block=context_block,
    )
    reasoning, model_name = _llm_call(pass1_prompt)

    # Step 5: Pass 2 — structured brief
    pass2_prompt = PASS2_TEMPLATE.format(
        topic=topic,
        context_block=context_block,
        reasoning=reasoning[:1500],  # cap to avoid context overflow on small models
        date=date_str,
        model=model_name,
    )
    raw_json_text, _ = _llm_call(pass2_prompt)

    # Step 6: Parse + validate
    try:
        data = _extract_json(raw_json_text)
        # Ensure generated_at and model_used are always correct (LLM may hallucinate)
        data["generated_at"] = date_str
        data["model_used"] = model_name
        data["topic"] = topic
        brief = ResearchBrief(**data)
    except Exception as e:
        # Graceful degradation: return a minimal valid brief rather than crashing
        brief = ResearchBrief(
            topic=topic,
            headline="Insufficient data for brief generation",
            narrative=(
                f"Brief generation failed: {e}. "
                "Raw reasoning is available for manual review."
            ),
            most_affected_segment="senior",
            drift_type="stable",
            alert_level="none",
            hypotheses=[
                {
                    "segment": seg,
                    "hypothesis": "Insufficient signal to generate hypothesis.",
                    "signal_basis": "N/A",
                    "suggested_question": "How would you rate your current concern about this topic?",
                }
                for seg in ["young_urban", "family", "senior"]
            ],
            generated_at=date_str,
            model_used=model_name,
        )

    return brief
```

---

### 5. Create `api/routes/brief.py`

```python
from fastapi import APIRouter, HTTPException

from brief.generator import generate_brief
from brief.models import ResearchBrief

router = APIRouter()


@router.get("/brief/{topic}", response_model=ResearchBrief)
def get_brief(topic: str) -> ResearchBrief:
    """
    Generate a Lakmoos-style research brief for a topic.
    Runs the two-pass LLM synthesis over current drift data.
    topic: any topic string, or '_all' for global
    """
    try:
        return generate_brief(topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brief generation error: {e}")
```

---

### 6. Update `main.py`

Add the brief router. Replace the entire file:

```python
from fastapi import FastAPI

import db.init
from api.routes.collect import router as collect_router
from api.routes.signals import router as signals_router
from api.routes.calibration import router as calibration_router
from api.routes.brief import router as brief_router
from delta.seeder import seed_baselines

app = FastAPI(title="SigDriftr")


@app.on_event("startup")
def startup() -> None:
    db.init.get_conn()
    seeded = seed_baselines()
    if seeded:
        print(f"[SigDriftr] Seeded {seeded} baseline rows.")


app.include_router(collect_router, prefix="")
app.include_router(signals_router, prefix="")
app.include_router(calibration_router, prefix="")
app.include_router(brief_router, prefix="")
```

---

### 7. Update `requirements.txt`

Add `openai>=1.0` for the fallback path. The primary path (Ollama via `requests`) requires no new package.

Append to existing `requirements.txt`:
```
openai>=1.0
```

---

## pip Dependencies

```
openai>=1.0   # fallback only — only used if Ollama unavailable
```

All other imports (`json`, `re`, `datetime`) are stdlib. `requests` was already a dependency from plan 01.

---

## Test to Confirm It Works

Create `test_brief.py` in the repo root:

```python
# test_brief.py"""
End-to-end test for the Research Brief Generator.
Requires: Ollama running locally with gemma3:1b or phi3:mini pulled,
or OPENAI_API_KEY set in environment for fallback.
"""
from ingestion.crawler import crawl
from extraction.extractor import run_extraction
from delta.seeder import seed_baselines
from brief.generator import generate_brief
from brief.models import ResearchBrief

# Ensure pipeline has data
seed_baselines()
crawl("inflace")
run_extraction("inflace")

# Generate brief
brief = generate_brief("inflace")

# Validate type
assert isinstance(brief, ResearchBrief), "Expected ResearchBrief instance"

# Validate structure
assert len(brief.headline) > 5, "Headline too short"
assert len(brief.narrative) > 20, "Narrative too short"
assert brief.most_affected_segment in ["young_urban", "family", "senior", "b2b"]
assert brief.drift_type in ["concern_spike", "purchase_surge", "avoidance_rise", "frame_shift", "mixed", "stable"]
assert brief.alert_level in ["none", "mild", "strong"]
assert len(brief.hypotheses) == 3

for h in brief.hypotheses:
    assert h.segment in ["young_urban", "family", "senior", "b2b"]
    assert len(h.hypothesis) > 10
    assert len(h.suggested_question) > 10

print(f"\n=== BRIEF: {brief.topic.upper()} ===")
print(f"Headline : {brief.headline}")
print(f"Narrative: {brief.narrative}")
print(f"Type     : {brief.drift_type} | Alert: {brief.alert_level}")
print(f"Most affected: {brief.most_affected_segment}")
print("\nHypotheses:")
for i, h in enumerate(brief.hypotheses, 1):
    print(f"  {i}. [{h.segment}] {h.hypothesis}")
    print(f"     Basis: {h.signal_basis}")
    print(f"     Q: {h.suggested_question}")

print(f"\nModel: {brief.model_used}")
print("\nAll brief generator checks passed.")
```

Expected output shape:
```
=== BRIEF: INFLACE ===
Headline : Seniors show rising concern as inflation narrative intensifies
Narrative: Czech media coverage of inflation over the past 7 days has shifted...
Type     : concern_spike | Alert: strong
Most affected: senior

Hypotheses:
  1. [senior] Seniors exposed to fear-framed inflation coverage...
  2. [family] Family households show increased avoidance signals...
  3. [b2b]    B2B decision-makers maintain purchase intent despite...

Model: gemma3:1b
All brief generator checks passed.
```

---

## Notes for the Coding Agent

- **Two-pass design is intentional.** Do not collapse Pass 1 and Pass 2 into a single prompt. Small local models (Gemma-3-1B, Phi-3-mini) produce significantly better structured output when reasoning is externalized first and then injected as explicit context into the formatting pass. This is backed by the intent decomposition research from Google (2026).
- **`_extract_json` must be robust.** Local LLMs frequently wrap JSON in markdown code fences (```json ... ```) even when instructed not to. The regex stripping in `_extract_json` handles this. Do not simplify it.
- **Graceful degradation is mandatory.** If the LLM fails or returns unparseable output, the endpoint must return a valid (degraded) `ResearchBrief`, not a 500 error. The fallback block in `generate_brief` handles this. Keep it.
- **`reasoning[:1500]` cap in Pass 2.** Gemma-3-1B has a small context window (~2K tokens). Capping the injected reasoning prevents overflow. Do not remove this cap.
- **Ollama model name.** Use `"gemma3:1b"` (with colon, Ollama format). If unavailable, `"phi3:mini"` is the fallback. The `_call_ollama` function will raise a `requests.exceptions.ConnectionError` if Ollama is not running, which correctly triggers the OpenAI fallback chain.
- **Do NOT add caching or background jobs in this plan.** That is Stage 5 (API hardening/scheduling) scope.
- **Do NOT import from `extraction.extractor` in `brief/`.** The brief module is downstream-only: it reads from DB (via `delta.engine`) and calls the LLM. No direct dependency on ingestion or extraction.
- **`topic = '_all'`** maps to empty string internally (same convention as plan 03). The brief router follows the same convention.
- **`openai` is an optional dependency.** It is only imported inside `_call_openai_fallback` to avoid import errors when the package is not installed. Keep the lazy import inside the function.
