# Plan 02 — LLM Behavioral Signal Extraction

**Goal:** For every article stored in SQLite by plan 01, call an LLM and extract a structured behavioral signal JSON that later stages (segment mapper, delta engine) consume.

**Where it fits:** Sits directly after ingestion. Reads unprocessed articles from `articles` table, writes results to a new `signals` table.

---

## Inputs
- SQLite `articles` table (from plan 01) with columns: `id, outlet, title, summary, url, topic, published_at, fetched_at`
- Ollama running locally (`gemma3:1b` or `phi3:mini`) OR fallback to `distilbart-cnn-12-6` via HuggingFace `transformers` pipeline

## Outputs
- New SQLite table `signals` with columns: `article_id, concern_level, purchase_intent, avoidance_signals, dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b, raw_json, extracted_at`
- New FastAPI endpoint `GET /signals?topic=<topic>` returning list of signal objects

---

## Exact File Paths

```
extraction/
    __init__.py          # empty
    llm_client.py        # Ollama call + HuggingFace fallback
    extractor.py         # reads unprocessed articles, calls LLM, writes signals
api/routes/signals.py    # GET /signals?topic= endpoint
db/init.py               # ADD signals table (edit existing file)
requirements.txt         # ADD transformers, torch (cpu), ollama
```

---

## Numbered Steps

### 1. Extend the database schema — `db/init.py`
Add a second `CREATE TABLE IF NOT EXISTS` block right after the existing `articles` table creation:

```python
_CONN.execute(
    """
    CREATE TABLE IF NOT EXISTS signals (
        article_id       TEXT PRIMARY KEY,
        concern_level    REAL,
        purchase_intent  REAL,
        avoidance_signals REAL,
        dominant_frame   TEXT,
        seg_young_urban  REAL,
        seg_family       REAL,
        seg_senior       REAL,
        seg_b2b          REAL,
        raw_json         TEXT,
        extracted_at     TEXT NOT NULL,
        FOREIGN KEY (article_id) REFERENCES articles(id)
    );
    """
)
_CONN.commit()
```

---

### 2. Create `extraction/llm_client.py`

This module exposes one function: `extract_signals(title: str, summary: str) -> dict`.

**Strategy — try Ollama first, fallback to HuggingFace:**

```python
import json, re, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"  # or phi3:mini — whichever is pulled

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
"""

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

def _try_ollama(title: str, summary: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        # extract first JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

def _try_hf_fallback(title: str, summary: str) -> dict:
    """
    HuggingFace fallback: use zero-shot-classification with a small multilingual model
    to score dominant_frame, then fill numeric fields with heuristics.
    Uses facebook/bart-large-mnli (or a smaller model if available).
    """
    try:
        from transformers import pipeline
        classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        text = f"{title}. {summary}"[:512]
        labels = ["fear", "opportunity", "conflict", "neutral"]
        result = classifier(text, candidate_labels=labels)
        dominant_frame = result["labels"][0]
        scores = dict(zip(result["labels"], result["scores"]))
        concern = round(scores.get("fear", 0) + scores.get("conflict", 0), 2)
        return {
            **DEFAULT_SIGNALS,
            "dominant_frame": dominant_frame,
            "concern_level": min(concern, 1.0),
        }
    except Exception:
        return DEFAULT_SIGNALS

def extract_signals(title: str, summary: str) -> dict:
    result = _try_ollama(title, summary)
    if result is None:
        result = _try_hf_fallback(title, summary)
    # Ensure all required keys exist
    for key, default in DEFAULT_SIGNALS.items():
        result.setdefault(key, default)
    return result
```

---

### 3. Create `extraction/extractor.py`

This module exposes one function: `run_extraction(topic: str) -> int` (returns count of newly extracted articles).

```python
import json
from datetime import datetime, timezone
from db.init import get_conn
from extraction.llm_client import extract_signals

def run_extraction(topic: str) -> int:
    conn = get_conn()
    # fetch articles that have NOT been extracted yet
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.summary
        FROM articles a
        LEFT JOIN signals s ON a.id = s.article_id
        WHERE s.article_id IS NULL
          AND (a.topic = ? OR ? = '')
        """,
        (topic, topic),
    ).fetchall()

    processed = 0
    for article_id, title, summary in rows:
        signals = extract_signals(title, summary or "")
        conn.execute(
            """
            INSERT OR IGNORE INTO signals
            (article_id, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
             raw_json, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                signals["concern_level"],
                signals["purchase_intent"],
                signals["avoidance_signals"],
                signals["dominant_frame"],
                signals["seg_young_urban"],
                signals["seg_family"],
                signals["seg_senior"],
                signals["seg_b2b"],
                json.dumps(signals),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        processed += 1

    conn.commit()
    return processed
```

---

### 4. Create `extraction/__init__.py`
Empty file:
```python
```

---

### 5. Create `api/routes/signals.py`

```python
from fastapi import APIRouter
from db.init import get_conn
from extraction.extractor import run_extraction

router = APIRouter()

@router.post("/extract")
def extract(topic: str = ""):
    count = run_extraction(topic)
    return {"extracted": count, "topic": topic}

@router.get("/signals")
def get_signals(topic: str = ""):
    conn = get_conn()
    query = """
        SELECT s.*, a.outlet, a.title, a.url, a.published_at
        FROM signals s
        JOIN articles a ON s.article_id = a.id
    """
    params = []
    if topic:
        query += " WHERE a.topic = ?"
        params.append(topic)
    rows = conn.execute(query, params).fetchall()
    cols = ["article_id","concern_level","purchase_intent","avoidance_signals",
            "dominant_frame","seg_young_urban","seg_family","seg_senior","seg_b2b",
            "raw_json","extracted_at","outlet","title","url","published_at"]
    return [{c: row[i] for i, c in enumerate(cols)} for row in rows]
```

---

### 6. Register new routes in `main.py`
In `main.py`, add after the existing `collect` router include:

```python
from api.routes.signals import router as signals_router
app.include_router(signals_router)
```

---

### 7. Update `requirements.txt`
Add these lines (keep existing ones):
```
transformers>=4.40
torch  # CPU-only is fine; no CUDA needed
ollama  # optional, only needed if using Ollama Python SDK instead of raw requests
```
Note: The code uses raw `requests` for Ollama, so `ollama` package is optional. `torch` is needed for HuggingFace transformers inference.

---

## pip Dependencies
```
feedparser        # already installed (plan 01)
requests          # already installed (plan 01)
fastapi           # already installed (plan 01)
transformers>=4.40
torch
```

---

## Test to Confirm It Works

Create `test_extraction.py` in the repo root and run it manually:

```python
from ingestion.crawler import crawl
from extraction.extractor import run_extraction
from db.init import get_conn

# Step 1: ingest some articles
crawled = crawl("energie")
print(f"Crawled: {crawled} new articles")

# Step 2: extract signals
extracted = run_extraction("energie")
print(f"Extracted: {extracted} signal rows")

# Step 3: verify at least one signal row has valid data
conn = get_conn()
row = conn.execute(
    "SELECT concern_level, dominant_frame FROM signals LIMIT 1"
).fetchone()
assert row is not None, "No signals found — extraction failed"
assert row[1] in ("fear", "opportunity", "conflict", "neutral"), f"Bad frame: {row[1]}"
assert 0.0 <= row[0] <= 1.0, f"Bad concern_level: {row[0]}"
print(f"OK — concern={row[0]}, frame={row[1]}")
```

Expected output:
```
Crawled: N new articles
Extracted: M signal rows
OK — concern=0.X, frame=<fear|opportunity|conflict|neutral>
```

---

## Notes for the coding agent
- Do NOT fine-tune any model. Inference only.
- Ollama must be running locally (`ollama serve`) for the primary path. The HuggingFace fallback must work even without Ollama.
- Keep `DEFAULT_SIGNALS` as the final safety net — the extraction must never crash the API.
- The `dominant_frame` field is a string enum: `fear | opportunity | conflict | neutral` — validate/clamp it after LLM response.
- All float scores must be clamped to `[0.0, 1.0]` after parsing.
