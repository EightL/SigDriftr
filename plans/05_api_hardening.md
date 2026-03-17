# 05 — API Hardening: Async Brief Jobs, Caching & Health

## Goal
Make the SigDriftr FastAPI server demo-safe: the slow `GET /brief/{topic}` call must never block or time out during a live demo, briefs must be cached so repeat calls are instant, and a health endpoint must confirm the full pipeline is alive.

## Where It Fits
This is the final production-readiness layer. All pipeline stages (ingestion → extraction → delta → brief) already exist. This stage makes the API surface robust and presentable.

## Inputs
- `main.py` — existing FastAPI app with 4 routers registered
- `api/routes/brief.py` — currently a synchronous blocking `GET /brief/{topic}` that calls Ollama twice
- `brief/generator.py` — `generate_brief(topic)` — the slow two-pass LLM function
- `db/init.py` — SQLite connection helper

## Outputs
- `api/routes/brief.py` — rewritten with async job pattern (POST to start, GET to poll)
- `db/init.py` or `db/schema.py` — new `brief_cache` table added to schema
- `api/routes/health.py` — new health check route
- `main.py` — updated to include health router

---

## Steps

### 1. Add `brief_cache` table to the database schema

Open `db/init.py` (or wherever `CREATE TABLE` statements live). Add the following table if it does not already exist:

```sql
CREATE TABLE IF NOT EXISTS brief_cache (
    id          TEXT PRIMARY KEY,  -- sha256(topic)
    topic       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | error
    result_json TEXT,              -- serialised ResearchBrief JSON, NULL until done
    error_msg   TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
```

The `id` is `hashlib.sha256(topic.encode()).hexdigest()` — same pattern already used in `delta/engine.py`.

Call this `CREATE TABLE` in the same `get_conn()` / init block so it runs on startup automatically.

---

### 2. Rewrite `api/routes/brief.py` with an async job pattern

Replace the current single blocking GET with three endpoints:

#### `POST /brief/{topic}` — enqueue a brief generation job

```python
@router.post("/brief/{topic}", status_code=202)
def enqueue_brief(topic: str, background_tasks: BackgroundTasks) -> dict:
```

- Compute `job_id = sha256(topic)`
- Check `brief_cache` for an existing row:
  - If `status == 'done'` and `updated_at` is less than 30 minutes ago → return `{"job_id": job_id, "status": "done"}` immediately (cache hit, no re-generation)
  - If `status == 'running'` → return `{"job_id": job_id, "status": "running"}` (already in flight)
  - Otherwise → upsert row with `status='pending'`, add background task
- Add `background_tasks.add_task(_run_brief_job, topic, job_id)`
- Return `{"job_id": job_id, "status": "pending"}`

#### `GET /brief/{topic}/status` — poll job status

```python
@router.get("/brief/{topic}/status")
def get_brief_status(topic: str) -> dict:
```

- Look up `job_id = sha256(topic)` in `brief_cache`
- If not found: return `{"status": "not_started"}`
- Return `{"job_id": job_id, "status": row.status, "updated_at": row.updated_at}`

#### `GET /brief/{topic}/result` — fetch the completed brief

```python
@router.get("/brief/{topic}/result", response_model=ResearchBrief)
def get_brief_result(topic: str) -> ResearchBrief:
```

- Look up in `brief_cache`
- If `status != 'done'`: raise `HTTPException(409, "Brief not ready yet")`
- Parse `result_json` and return as `ResearchBrief`

#### Background worker function `_run_brief_job(topic, job_id)`

This is a plain (non-async) function that FastAPI runs in a thread via `BackgroundTasks`:

```python
def _run_brief_job(topic: str, job_id: str) -> None:
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    # Mark as running
    conn.execute("UPDATE brief_cache SET status='running', updated_at=? WHERE id=?", (now, job_id))
    conn.commit()
    try:
        brief = generate_brief(topic)
        result_json = brief.model_dump_json()
        now2 = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE brief_cache SET status='done', result_json=?, updated_at=? WHERE id=?",
            (result_json, now2, job_id)
        )
    except Exception as exc:
        now3 = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE brief_cache SET status='error', error_msg=?, updated_at=? WHERE id=?",
            (str(exc)[:500], now3, job_id)
        )
    finally:
        conn.commit()
```

---

### 3. Create `api/routes/health.py`

Create a new file `api/routes/health.py` with a single endpoint:

```python
from fastapi import APIRouter
from db.init import get_conn

router = APIRouter()

@router.get("/health")
def health() -> dict:
    checks = {}
    # DB check
    try:
        conn = get_conn()
        article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        checks["db"] = "ok"
        checks["articles"] = article_count
        checks["signals"] = signal_count
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Ollama reachability check (fast, no generation)
    import urllib.request, json
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            models = json.loads(resp.read()).get("models", [])
            checks["ollama"] = "ok"
            checks["ollama_models"] = [m["name"] for m in models]
    except Exception as exc:
        checks["ollama"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" or isinstance(v, (int, list)) for v in checks.values()) else "degraded"
    return {"status": overall, **checks}
```

---

### 4. Register health router in `main.py`

Add to `main.py`:

```python
from api.routes.health import router as health_router
# ...
app.include_router(health_router, prefix="")
```

---

### 5. Add `sse_starlette` streaming endpoint (optional but recommended for demo)

If time allows, add a streaming variant to `api/routes/brief.py` using `sse-starlette`:

```python
from sse_starlette.sse import EventSourceResponse

@router.get("/brief/{topic}/stream")
async def stream_brief_status(topic: str):
    """SSE endpoint — streams status updates until brief is done."""
    async def generator():
        import asyncio
        job_id = hashlib.sha256(topic.encode()).hexdigest()
        for _ in range(120):  # max 2 minutes polling
            row = get_conn().execute(
                "SELECT status, updated_at FROM brief_cache WHERE id=?", (job_id,)
            ).fetchone()
            status = row[0] if row else "not_started"
            yield {"data": json.dumps({"status": status, "topic": topic})}
            if status in ("done", "error"):
                break
            await asyncio.sleep(1)
    return EventSourceResponse(generator())
```

This lets a frontend (or `curl -N`) watch the brief being generated in real time — very compelling in a demo context.

---

## pip Dependencies

No new mandatory dependencies. If implementing the SSE stream endpoint:

```
sse-starlette>=1.8.2
```

Add to `requirements.txt`.

---

## File Summary

| File | Action |
|---|---|
| `db/init.py` | Add `brief_cache` CREATE TABLE |
| `api/routes/brief.py` | Full rewrite — POST enqueue, GET status, GET result, optional GET stream |
| `api/routes/health.py` | New file |
| `main.py` | Add health router import + include |
| `requirements.txt` | Add `sse-starlette` if SSE step is done |

---

## Test to Confirm It Works

Run the following sequence:

```bash
# 1. Start server
uvicorn main:app --reload

# 2. Confirm health
curl http://localhost:8000/health
# Expected: {"status": "ok", "db": "ok", "articles": N, "signals": N, "ollama": "ok"}

# 3. Enqueue a brief
curl -X POST http://localhost:8000/brief/inflace
# Expected: {"job_id": "...", "status": "pending"}

# 4. Poll until done
curl http://localhost:8000/brief/inflace/status
# Expected: {"status": "running"} then {"status": "done"}

# 5. Fetch result
curl http://localhost:8000/brief/inflace/result
# Expected: full ResearchBrief JSON with headline, narrative, hypotheses

# 6. Enqueue same topic again — should return status=done immediately (cache hit)
curl -X POST http://localhost:8000/brief/inflace
# Expected: {"job_id": "...", "status": "done"}
```

All 6 steps passing = stage 5 complete.
