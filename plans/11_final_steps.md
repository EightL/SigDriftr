
# 11_final_product_completion

**Goal:** Close every remaining gap between the current codebase and a fledged-out,
demo-ready product — wiring the bandit reward loop, hardening the brief output
contract, adding a scheduled background pipeline, and producing a polished
end-to-end integration test.

Fits in pipeline: This is the capstone plan. It touches `ingestion/` (reward
feedback), `brief/` (output contract), `api/` (scheduler + collect flow),
`db/` (crawl-history table), and `scripts/` (warm-start CLI + e2e test).

Inputs:
- All modules from plans 01–10 as implemented.
- `ingestion/bandit.py` — `record_signal_reward`, `update_feed_reward`,
  `warm_start_from_history` already present.
- `brief/models.py` — `BriefConfidenceContext` exists but is **not** embedded
  in `ResearchBrief`.
- `api/routes/collect.py` — calls `_crawl_async` but does **not** call
  `run_extraction` or feed bandit rewards.
- No scheduler exists; the pipeline must be triggered manually via HTTP.
- No end-to-end integration test exists.

Outputs:
- Bandit reward loop closed: every crawl → extract → reward cycle updates bandit
  state automatically.
- `ResearchBrief` carries `confidence_context` so API consumers always get
  calibration metadata.
- A `/pipeline/run` endpoint that chains collect → extract → brief in one call.
- An APScheduler background job that runs the pipeline every N minutes per
  configured topic list.
- A `scripts/warm_start.py` CLI for bandit cold-start from DB history.
- A `test_e2e.py` integration test that validates the whole stack from empty DB
  to a produced brief.

---

## Design Rationale

### Why close the reward loop now?
The bandit (`ingestion/bandit.py`) implements the full disjoint LinUCB from
Li et al. (2010, "A Contextual-Bandit Approach to Personalized News Article
Recommendation") with UCB score `p = θᵀx + α√The push failed (likely a branch protection or permission issue), but here is the full plan in `.md` format as requested. You can push it manually or tell me to try again.

---

```md
# 11_final_product_completion

**Goal:** Close every remaining gap between the current codebase and a fledged-out,
demo-ready product — wiring the bandit reward loop, hardening the brief output contract,
adding a scheduled background pipeline, and producing a polished end-to-end integration test.

Fits in pipeline: Capstone plan. Touches `ingestion/` (reward feedback), `brief/` (output
contract), `api/` (scheduler + collect flow), and `scripts/` (warm-start CLI + e2e test).

Inputs:
- All modules from plans 01–10 as implemented.
- `ingestion/bandit.py` — `record_signal_reward`, `update_feed_reward`,
  `warm_start_from_history` already present but never called from the HTTP layer.
- `brief/models.py` — `BriefConfidenceContext` exists but is NOT a field on `ResearchBrief`.
- `api/routes/collect.py` — calls `_crawl_async` but does NOT call `run_extraction`
  or feed bandit rewards afterwards.
- No scheduler exists; pipeline must be triggered manually via HTTP.
- No end-to-end integration test exists.

Outputs:
- Bandit reward loop closed: every crawl → extract → reward cycle updates bandit state.
- `ResearchBrief` carries `confidence_context` so API consumers get calibration metadata.
- A `/pipeline/run` endpoint chains collect → extract → brief in one call.
- An APScheduler background job runs the pipeline every N minutes per configured topics.
- `scripts/warm_start.py` CLI initialises the bandit from any existing DB.
- `test_e2e.py` validates the whole stack from empty DB to produced brief.

***

## Design Rationale

### Why close the reward loop now?
The bandit implements disjoint LinUCB (Li et al., 2010 — "A Contextual-Bandit Approach to
Personalized News Article Recommendation") with UCB score `p = θᵀx + α√(xᵀA⁻¹x)`.
The A and b matrices update via `update_feed_reward`, and `record_signal_reward` correctly
computes domain-weighted rewards from extraction output. But `/collect` never calls
extraction, and extraction never calls the bandit reward update. The bandit is a no-op
after cold start until this loop is closed.

### Why embed confidence in ResearchBrief?
`BriefConfidenceContext` is built inside `generate_brief` but thrown away at the response
boundary. Research on LLM confidence (Xiong et al., ICLR 2024 — "Can LLMs Express Their
Uncertainty?") shows that verbalising confidence without surfacing it in a structured field
leads downstream consumers to ignore it. The fix is one optional field on `ResearchBrief`.

### Why a scheduler?
For an overnight demo, feeds must refresh automatically. APScheduler's
`BackgroundScheduler` runs inside the same FastAPI process — no Celery, no Redis — which
matches the single-worker profile the project already targets.

### Why an e2e test?
Plan 10 §6 specified this test but it was never written. Without it, a refactor can silently
break the ingestion → extraction → drift → brief chain with no fast signal.

***

## 1. Close the bandit reward loop in /collect

**File: `api/routes/collect.py`** — replace the thin wrapper:

1. Store `crawl_start = datetime.now(timezone.utc).isoformat()` before calling `_crawl_async`.
2. Call `await _crawl_async(topic)` → `inserted`.
3. If `inserted > 0`, call `run_extraction(topic)` → `processed`.
4. Query signals for articles with `fetched_at >= crawl_start`:
   ```sql
   SELECT s.article_id, a.outlet, a.published_at,
          s.concern_level, s.purchase_intent, s.avoidance_signals, s.raw_json
   FROM signals s JOIN articles a ON a.id = s.article_id
   WHERE a.topic = ? AND a.fetched_at >= ?
   ```
5. For each row call `record_signal_reward(outlet, topic, signals, when=published_at)`.
6. Return `{"inserted": inserted, "extracted": processed, "rewards_recorded": N, "topic": topic}`.

The `crawl_start` filter is critical — it prevents re-rewarding historical articles on
every crawl tick, which would bias the bandit toward outlets that appeared early in history.

***

## 2. Embed confidence_context in ResearchBrief

**File: `brief/models.py`** — add one optional field:

```python
from typing import Optional

class ResearchBrief(BaseModel):
    # ... all existing fields unchanged ...
    confidence_context: Optional[BriefConfidenceContext] = Field(
        default=None,
        description="Per-segment calibration metadata; None when not available."
    )
```

**File: `brief/generator.py`** — after building the brief, attach context:

```python
brief = _apply_confidence_language(ResearchBrief(**data), drift)
return brief.model_copy(update={"confidence_context": confidence_context})
```

Apply the same `model_copy` line to both `_fallback_brief` call sites.

***

## 3. Add /pipeline/run endpoint

**File: `api/routes/pipeline.py`** (new file):

```python
from fastapi import APIRouter
from api.routes.collect import collect
from brief.generator import generate_brief_cached
from brief.models import ResearchBrief

router = APIRouter()

@router.post("/pipeline/run")
async def run_pipeline(topic: str) -> dict:
    collect_result = await collect(topic)        # crawl + extract + reward
    brief: ResearchBrief = generate_brief_cached(topic)
    return {
        **collect_result,
        "brief_topic": brief.topic,
        "brief_alert_level": brief.alert_level,
        "brief_confidence": (
            brief.confidence_context.segment_confidence
            if brief.confidence_context else {}
        ),
    }
```

Register in `main.py`:
```python
from api.routes.pipeline import router as pipeline_router
app.include_router(pipeline_router)
```

***

## 4. APScheduler background job

**File: `config/settings.py`** — add, with env override:

```python
import os, json
SCHEDULED_TOPICS: list[str] = json.loads(
    os.environ.get("SCHEDULED_TOPICS", '["inflace", "energie", "zdravi"]')
)
PIPELINE_INTERVAL_MINUTES: int = int(
    os.environ.get("PIPELINE_INTERVAL_MINUTES", "30")
)
```

**File: `api/scheduler.py`** (new file):

```python
import asyncio, logging
from apscheduler.schedulers.background import BackgroundScheduler
from config.settings import SCHEDULED_TOPICS, PIPELINE_INTERVAL_MINUTES
from ingestion.crawler import _crawl_async
from extraction.extractor import run_extraction
from delta.mapper import compute_segment_profiles

logger = logging.getLogger(__name__)
_scheduler = None

def _run_topic_pipeline(topic: str) -> None:
    try:
        inserted = asyncio.run(_crawl_async(topic))
        if inserted > 0:
            run_extraction(topic)
            compute_segment_profiles(topic, learn_baseline=True)
        logger.info("[scheduler] topic=%s inserted=%d", topic, inserted)
    except Exception as exc:
        logger.warning("[scheduler] topic=%s error=%s", topic, exc)

def start_scheduler() -> None:
    global _scheduler
    if _scheduler: return
    _scheduler = BackgroundScheduler(daemon=True)
    for topic in SCHEDULED_TOPICS:
        _scheduler.add_job(
            _run_topic_pipeline, "interval",
            minutes=PIPELINE_INTERVAL_MINUTES,
            args=[topic], id=f"pipeline_{topic}", max_instances=1,
        )
    _scheduler.start()
    logger.info("[scheduler] started %d jobs every %d min",
                len(SCHEDULED_TOPICS), PIPELINE_INTERVAL_MINUTES)

def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
```

**File: `main.py`** — hook into FastAPI lifespan:

```python
from contextlib import asynccontextmanager
from api.scheduler import start_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app):
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(lifespan=lifespan)
```

***

## 5. Warm-start CLI

**File: `scripts/warm_start.py`** (new file):

```python
#!/usr/bin/env python
"""Replay historical signal data into the LinUCB bandit.

Usage:
    python scripts/warm_start.py
    python scripts/warm_start.py --topic inflace
    python scripts/warm_start.py --topic inflace --limit 500
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.bandit import warm_start_from_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(f"Warm-starting bandit (topic={args.topic or 'all'}, limit={args.limit or 'unlimited'})...")
    n = warm_start_from_history(topic=args.topic, limit=args.limit)
    print(f"Done. Replayed {n} signal records.")

if __name__ == "__main__":
    main()
```

***

## 6. End-to-end integration test

**File: `test_e2e.py`** (new file at repo root).

Patch three things so the test runs without internet or Ollama:
- `ingestion.crawler._fetch_feed_bytes` → return a minimal RSS XML bytes string
  containing 3 fake articles about "inflace".
- The Ollama call inside `extraction/llm_client.py` → return fixed signals
  `{concern_level: 0.7, purchase_intent: 0.1, avoidance_signals: 0.4, dominant_frame: "fear"}`.
- `brief.generator._call_ollama_json` → return a valid brief dict.

Test scenario (all assertions must pass, must complete < 30 s):

```
1. Monkeypatch DB_PATH to a tmp file (pytest tmp_path fixture).
2. crawl("inflace") → assert inserted >= 1.
3. run_extraction("inflace") → assert processed >= 1.
4. compute_segment_profiles("inflace", learn_baseline=True).
5. drift = compute_drift("inflace")
   For each entry assert: keys confidence, baseline_is_learned,
   baseline_sample_count, domain, relevant_fields all present.
6. brief = generate_brief("inflace")
   assert brief.topic == "inflace"
   assert brief.confidence_context is not None
   assert set(brief.confidence_context.segment_confidence.keys()) == {"young_urban","family","senior","b2b"}
   assert brief.alert_level in ("none","mild","strong")
   assert len(brief.hypotheses) == 3
7. snapshot = get_bandit_snapshot(outlet)
   assert snapshot["pulls"] >= 1
```

***

## File change summary

| File | Action | What changes |
|---|---|---|
| `api/routes/collect.py` | Update | Chain extract + reward after crawl |
| `brief/models.py` | Update | Add `confidence_context` to `ResearchBrief` |
| `brief/generator.py` | Update | Attach `confidence_context` to returned brief |
| `api/routes/pipeline.py` | Create | `/pipeline/run` endpoint |
| `api/scheduler.py` | Create | APScheduler per-topic background job |
| `main.py` | Update | Register pipeline router + lifespan hooks |
| `config/settings.py` | Update | `SCHEDULED_TOPICS`, `PIPELINE_INTERVAL_MINUTES` |
| `scripts/warm_start.py` | Create | Bandit warm-start CLI |
| `test_e2e.py` | Create | End-to-end integration test |
| `requirements.txt` | Update | Add `apscheduler>=3.10,<4.0` |

***

## Confirm it works

```bash
# Run all tests
python -m pytest test_e2e.py test_bandit.py test_baseline_confidence.py \
               test_brief.py test_delta.py -v

# Start server and run the full pipeline
uvicorn main:app --reload
curl -X POST "http://localhost:8000/pipeline/run?topic=inflace"
# Expected: {"inserted":N,"extracted":N,"rewards_recorded":N,"brief_alert_level":"...","brief_confidence":{...}}
```
