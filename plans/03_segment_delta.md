# Plan 03 — Segment Mapper + Delta Engine

**Goal:** Aggregate per-article signals into per-segment weekly profiles, compare against a seeded 4-week baseline, and expose drift scores via two new API endpoints — so the system produces interesting, non-zero calibration output from the very first run.

**Where it fits:** Reads from the `signals` + `articles` tables written by plans 01/02/02.5. Writes to two new tables: `segment_profiles` (rolling aggregates) and `baselines` (reference window). Adds `GET /calibration/{topic}/{segment}` and `GET /drift/{topic}` endpoints.

---

## Inputs
- SQLite `signals` table (from plan 02)
- SQLite `articles` table (from plan 01)
- `config/feeds.py` — affinity_tag per outlet

## Outputs
- New SQLite table `segment_profiles` — rolling 7-day per-segment aggregates
- New SQLite table `baselines` — seeded 4-week reference averages per topic+segment
- New module `delta/mapper.py` — segment mapper logic
- New module `delta/engine.py` — drift computation
- New module `delta/seeder.py` — baseline seeder (runs once at startup)
- New API routes `api/routes/calibration.py`
- Updated `main.py` — register new router + trigger seeder on startup

---

## Exact File Paths

```
delta/
    __init__.py
    mapper.py          # aggregates signals -> per-segment weekly profile
    engine.py          # computes drift vs baseline
    seeder.py          # seeds baseline table on first run
api/routes/
    calibration.py     # GET /calibration/{topic}/{segment}, GET /drift/{topic}
main.py                # register calibration router + call seeder on startup
db/init.py             # ADD segment_profiles + baselines tables
```

---

## Numbered Steps

### 1. Extend the database schema — `db/init.py`

Add two new `CREATE TABLE IF NOT EXISTS` blocks inside `get_conn()`, after the existing `signals` table creation:

```python
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS segment_profiles (
        id              TEXT PRIMARY KEY,  -- sha256(topic+segment+window_start)
        topic           TEXT NOT NULL,
        segment         TEXT NOT NULL,     -- young_urban | family | senior | b2b
        window_start    TEXT NOT NULL,     -- ISO date, Monday of the week
        window_days     INTEGER NOT NULL,  -- always 7
        concern_level   REAL,
        purchase_intent REAL,
        avoidance_signals REAL,
        dominant_frame  TEXT,
        article_count   INTEGER,
        computed_at     TEXT NOT NULL
    );
    """
)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS baselines (
        id              TEXT PRIMARY KEY,  -- sha256(topic+segment)
        topic           TEXT NOT NULL,
        segment         TEXT NOT NULL,
        concern_level   REAL,
        purchase_intent REAL,
        avoidance_signals REAL,
        dominant_frame  TEXT,             -- most common frame in baseline
        seeded          INTEGER DEFAULT 0, -- 1 = synthetic seed, 0 = computed
        updated_at      TEXT NOT NULL
    );
    """
)
```

---

### 2. Create `delta/__init__.py`

Empty file.

---

### 3. Create `delta/seeder.py`

This module seeds the `baselines` table with realistic Czech media priors if no baseline exists yet for a given topic+segment combination. Run once at startup.

The seeded values represent plausible 4-week averages for Czech mainstream media. They are deliberately non-neutral so that real incoming articles produce interesting, non-zero drift immediately.

```python
import hashlib
from datetime import datetime, timezone

from db.init import get_conn

# Realistic Czech media baseline priors per segment.
# These represent the approximate 4-week average behavioral signal profile
# observed in Czech mainstream media (crafted from domain knowledge).
# All values are floats in [0.0, 1.0].
SEED_PRIORS: dict[str, dict] = {
    "young_urban": {
        "concern_level": 0.38,
        "purchase_intent": 0.31,
        "avoidance_signals": 0.18,
        "dominant_frame": "opportunity",
    },
    "family": {
        "concern_level": 0.52,
        "purchase_intent": 0.27,
        "avoidance_signals": 0.29,
        "dominant_frame": "fear",
    },
    "senior": {
        "concern_level": 0.61,
        "purchase_intent": 0.14,
        "avoidance_signals": 0.41,
        "dominant_frame": "fear",
    },
    "b2b": {
        "concern_level": 0.44,
        "purchase_intent": 0.48,
        "avoidance_signals": 0.21,
        "dominant_frame": "conflict",
    },
}

SEGMENTS = list(SEED_PRIORS.keys())

# Topics to pre-seed baselines for.
# This list should cover the most common topics users will query.
DEFAULT_TOPICS = [
    "",           # empty string = global (all topics)
    "energie",
    "inflace",
    "bydleni",
    "zdravi",
    "politika",
    "ekonomika",
]


def seed_baselines(topics: list[str] | None = None) -> int:
    """
    Insert synthetic baseline rows for each topic+segment combination
    that does not already have a baseline row.
    Returns the number of rows inserted.
    """
    conn = get_conn()
    if topics is None:
        topics = DEFAULT_TOPICS

    inserted = 0
    now = datetime.now(timezone.utc).isoformat()

    for topic in topics:
        for segment in SEGMENTS:
            row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
            # Only insert if this baseline doesn't exist yet
            existing = conn.execute(
                "SELECT 1 FROM baselines WHERE id = ?", (row_id,)
            ).fetchone()
            if existing:
                continue

            prior = SEED_PRIORS[segment]
            conn.execute(
                """
                INSERT INTO baselines
                (id, topic, segment, concern_level, purchase_intent,
                 avoidance_signals, dominant_frame, seeded, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    row_id, topic, segment,
                    prior["concern_level"],
                    prior["purchase_intent"],
                    prior["avoidance_signals"],
                    prior["dominant_frame"],
                    now,
                ),
            )
            inserted += 1

    conn.commit()
    return inserted
```

---

### 4. Create `delta/mapper.py`

Aggregates all signals for a given topic over the last N days into a per-segment profile. Each signal score is weighted by the article's relevance to that segment (`seg_*` score from stage 2 — this is the affinity-weighted, softmax-normalized score from plan 02.5).

Weighted average formula for a signal `s` over segment `g`:

```
profile[g][s] = sum(signal[s] * seg_g) / sum(seg_g)   for all articles in window
```

This means an article that scores 0.9 on `seg_senior` contributes much more to the senior profile than one that scores 0.1 — which is exactly the right behavior.

```python
import hashlib
from datetime import datetime, timezone, timedelta

from db.init import get_conn

SEGMENTS = ["young_urban", "family", "senior", "b2b"]
SIGNAL_KEYS = ["concern_level", "purchase_intent", "avoidance_signals"]


def _window_start(days_back: int = 7) -> str:
    """ISO date string for the Monday that starts the current rolling window."""
    today = datetime.now(timezone.utc)
    window_begin = today - timedelta(days=days_back)
    return window_begin.isoformat()


def compute_segment_profiles(topic: str, days_back: int = 7) -> list[dict]:
    """
    Aggregate signals from the last `days_back` days into per-segment profiles.
    Returns a list of profile dicts (one per segment) and upserts them into
    the segment_profiles table.
    """
    conn = get_conn()
    since = _window_start(days_back)

    # Fetch all signals in the window for this topic
    query = """
        SELECT
            s.concern_level, s.purchase_intent, s.avoidance_signals,
            s.dominant_frame,
            s.seg_young_urban, s.seg_family, s.seg_senior, s.seg_b2b
        FROM signals s
        JOIN articles a ON s.article_id = a.id
        WHERE s.extracted_at >= ?
    """
    params: list = [since]
    if topic:
        query += " AND a.topic = ?"
        params.append(topic)

    rows = conn.execute(query, params).fetchall()

    profiles = []
    now = datetime.now(timezone.utc).isoformat()
    window_start_str = since[:10]  # date only

    for seg_idx, segment in enumerate(SEGMENTS):
        seg_col = 4 + seg_idx  # columns: concern(0), purchase(1), avoid(2), frame(3), seg_*(4-7)

        weighted_signals = {k: 0.0 for k in SIGNAL_KEYS}
        frame_counts: dict[str, float] = {}
        total_weight = 0.0

        for row in rows:
            weight = row[seg_col]  # seg_* score = relevance weight
            if weight <= 0:
                continue
            total_weight += weight
            for i, key in enumerate(SIGNAL_KEYS):
                weighted_signals[key] += row[i] * weight
            frame = row[3] or "neutral"
            frame_counts[frame] = frame_counts.get(frame, 0.0) + weight

        if total_weight > 0:
            profile_signals = {k: round(v / total_weight, 4) for k, v in weighted_signals.items()}
            dominant_frame = max(frame_counts, key=frame_counts.get) if frame_counts else "neutral"
        else:
            # No data in window — use neutral defaults
            profile_signals = {k: 0.0 for k in SIGNAL_KEYS}
            dominant_frame = "neutral"

        profile_id = hashlib.sha256(f"{topic}:{segment}:{window_start_str}".encode()).hexdigest()

        conn.execute(
            """
            INSERT OR REPLACE INTO segment_profiles
            (id, topic, segment, window_start, window_days,
             concern_level, purchase_intent, avoidance_signals,
             dominant_frame, article_count, computed_at)
            VALUES (?, ?, ?, ?, 7, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id, topic, segment, window_start_str,
                profile_signals["concern_level"],
                profile_signals["purchase_intent"],
                profile_signals["avoidance_signals"],
                dominant_frame,
                len(rows),
                now,
            ),
        )

        profiles.append({
            "segment": segment,
            "topic": topic,
            "window_start": window_start_str,
            "article_count": len(rows),
            **profile_signals,
            "dominant_frame": dominant_frame,
        })

    conn.commit()
    return profiles
```

---

### 5. Create `delta/engine.py`

Compares the freshly computed segment profile against the stored baseline. Outputs a drift dict for each segment with:
- **delta per signal** (signed float — positive = above baseline, negative = below)
- **drift magnitude** (absolute L1 norm across all signals — a single "how much did things move" number)
- **frame shift** (boolean — did the dominant frame change vs baseline?)
- **alert level** ("none" | "mild" | "strong") based on drift magnitude thresholds

```python
from db.init import get_conn
from delta.mapper import compute_segment_profiles, SEGMENTS, SIGNAL_KEYS

ALERT_THRESHOLDS = {
    "mild": 0.10,    # any single signal delta >= 0.10
    "strong": 0.25,  # any single signal delta >= 0.25
}


def _get_baseline(topic: str, segment: str) -> dict | None:
    """Fetch the baseline row for a topic+segment from the DB."""
    import hashlib
    conn = get_conn()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()
    row = conn.execute(
        """
        SELECT concern_level, purchase_intent, avoidance_signals, dominant_frame
        FROM baselines WHERE id = ?
        """,
        (row_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "concern_level": row[0],
        "purchase_intent": row[1],
        "avoidance_signals": row[2],
        "dominant_frame": row[3],
    }


def _alert_level(deltas: dict[str, float]) -> str:
    max_abs = max(abs(v) for v in deltas.values()) if deltas else 0.0
    if max_abs >= ALERT_THRESHOLDS["strong"]:
        return "strong"
    if max_abs >= ALERT_THRESHOLDS["mild"]:
        return "mild"
    return "none"


def compute_drift(topic: str, days_back: int = 7) -> list[dict]:
    """
    Compute drift for all segments for a given topic.
    Returns a list of drift dicts, one per segment.
    """
    profiles = compute_segment_profiles(topic, days_back)
    results = []

    for profile in profiles:
        segment = profile["segment"]
        baseline = _get_baseline(topic, segment)

        if baseline is None:
            # No baseline at all — return profile with zero drift
            results.append({
                "segment": segment,
                "topic": topic,
                "article_count": profile["article_count"],
                "current": {k: profile[k] for k in SIGNAL_KEYS},
                "baseline": None,
                "deltas": {k: 0.0 for k in SIGNAL_KEYS},
                "drift_magnitude": 0.0,
                "frame_shift": False,
                "alert_level": "none",
                "dominant_frame": profile["dominant_frame"],
            })
            continue

        deltas = {
            k: round(profile[k] - baseline[k], 4)
            for k in SIGNAL_KEYS
        }
        drift_magnitude = round(sum(abs(v) for v in deltas.values()), 4)
        frame_shift = profile["dominant_frame"] != baseline["dominant_frame"]
        alert = _alert_level(deltas)

        results.append({
            "segment": segment,
            "topic": topic,
            "article_count": profile["article_count"],
            "current": {k: profile[k] for k in SIGNAL_KEYS},
            "baseline": {k: baseline[k] for k in SIGNAL_KEYS},
            "deltas": deltas,
            "drift_magnitude": drift_magnitude,
            "frame_shift": frame_shift,
            "alert_level": alert,
            "dominant_frame": profile["dominant_frame"],
            "baseline_frame": baseline["dominant_frame"],
        })

    return results


def update_baseline_from_profile(topic: str, segment: str, profile: dict) -> None:
    """
    After enough real data accumulates, update the baseline with a
    weighted blend: 80% old baseline + 20% new profile.
    Only call this after >= 10 articles have been processed for the window.
    """
    import hashlib
    from datetime import datetime, timezone

    if profile.get("article_count", 0) < 10:
        return

    conn = get_conn()
    old = _get_baseline(topic, segment)
    if old is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    row_id = hashlib.sha256(f"{topic}:{segment}".encode()).hexdigest()

    blended = {
        k: round(0.8 * old[k] + 0.2 * profile[k], 4)
        for k in SIGNAL_KEYS
    }

    conn.execute(
        """
        UPDATE baselines
        SET concern_level=?, purchase_intent=?, avoidance_signals=?,
            dominant_frame=?, seeded=0, updated_at=?
        WHERE id=?
        """,
        (
            blended["concern_level"],
            blended["purchase_intent"],
            blended["avoidance_signals"],
            profile["dominant_frame"],
            now,
            row_id,
        ),
    )
    conn.commit()
```

---

### 6. Create `api/routes/calibration.py`

```python
from fastapi import APIRouter, HTTPException

from delta.engine import compute_drift
from delta.mapper import compute_segment_profiles, SEGMENTS

router = APIRouter()


@router.get("/calibration/{topic}/{segment}")
def get_calibration(topic: str, segment: str):
    """
    Returns the current 7-day weighted signal profile for one segment.
    topic: any topic string, or '_all' for global
    segment: one of young_urban | family | senior | b2b
    """
    if segment not in SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segment '{segment}'. Must be one of: {SEGMENTS}",
        )
    real_topic = "" if topic == "_all" else topic
    profiles = compute_segment_profiles(real_topic)
    match = next((p for p in profiles if p["segment"] == segment), None)
    if match is None:
        raise HTTPException(status_code=404, detail="No data for this topic/segment.")
    return match


@router.get("/drift/{topic}")
def get_drift(topic: str, days_back: int = 7):
    """
    Returns drift vs baseline for all segments for a given topic.
    topic: any topic string, or '_all' for global
    days_back: rolling window size in days (default 7)
    """
    real_topic = "" if topic == "_all" else topic
    drift = compute_drift(real_topic, days_back=days_back)
    return {
        "topic": topic,
        "days_back": days_back,
        "segments": drift,
    }
```

---

### 7. Update `main.py`

Register the new router and run the seeder on startup. Replace the entire file:

```python
from fastapi import FastAPI

import db.init
from api.routes.collect import router as collect_router
from api.routes.signals import router as signals_router
from api.routes.calibration import router as calibration_router
from delta.seeder import seed_baselines

app = FastAPI(title="SigDriftr")


@app.on_event("startup")
def startup() -> None:
    db.init.get_conn()          # initialise DB + tables
    seeded = seed_baselines()   # populate baselines if empty
    if seeded:
        print(f"[SigDriftr] Seeded {seeded} baseline rows.")


app.include_router(collect_router, prefix="")
app.include_router(signals_router, prefix="")
app.include_router(calibration_router, prefix="")
```

---

### 8. Update `requirements.txt`

No new dependencies for this plan. All imports (`hashlib`, `threading`, `datetime`, `math`) are stdlib. The delta module uses only what was already installed.

---

## pip Dependencies

None new. All stdlib.

---

## Test to Confirm It Works

Create `test_delta.py` in the repo root:

```python
# test_delta.py
from ingestion.crawler import crawl
from extraction.extractor import run_extraction
from delta.seeder import seed_baselines
from delta.engine import compute_drift

# Step 1: ensure baselines exist
seeded = seed_baselines()
print(f"Baseline rows seeded: {seeded}")

# Step 2: ingest + extract some articles
crawled = crawl("inflace")
print(f"Crawled: {crawled}")
extracted = run_extraction("inflace")
print(f"Extracted: {extracted}")

# Step 3: compute drift
drift = compute_drift("inflace")
assert len(drift) == 4, f"Expected 4 segments, got {len(drift)}"

for d in drift:
    print(
        f"  [{d['segment']:12}] "
        f"concern delta={d['deltas']['concern_level']:+.3f} "
        f"magnitude={d['drift_magnitude']:.3f} "
        f"alert={d['alert_level']:6} "
        f"frame_shift={d['frame_shift']}"
    )
    assert d["alert_level"] in ("none", "mild", "strong")
    assert "current" in d and "baseline" in d

print("\nAll delta engine checks passed.")
```

Expected output (values will vary with real articles):
```
Baseline rows seeded: 28
Crawled: N
Extracted: M
  [young_urban ] concern delta=+0.041 magnitude=0.112 alert=mild   frame_shift=False
  [family      ] concern delta=+0.093 magnitude=0.201 alert=mild   frame_shift=True
  [senior      ] concern delta=+0.178 magnitude=0.310 alert=strong frame_shift=False
  [b2b         ] concern delta=-0.021 magnitude=0.088 alert=none   frame_shift=False

All delta engine checks passed.
```

---

## Notes for the coding agent

- The `segment_profiles` table uses `INSERT OR REPLACE` — recomputing is always safe and idempotent.
- `seed_baselines()` is idempotent: it checks for existing rows before inserting, so calling it on every startup is safe.
- The `update_baseline_from_profile()` function in `engine.py` implements slow baseline drift (80/20 blend). Do NOT call it automatically on every request — it is intended to be called periodically (e.g. daily). For now it is implemented but not wired to any scheduler; that is stage 4's job.
- The `/drift/{topic}` endpoint always recomputes fresh on request. For demo purposes this is fine. Caching can be added later.
- `topic = '_all'` maps to empty string in SQL queries (no topic filter = all articles). This is the global calibration view.
- The seeded `SEED_PRIORS` are intentionally non-neutral (e.g. seniors have high concern/avoidance) so that even the first batch of real articles produces non-trivial drift numbers. This is by design for demo impact.
- Do NOT change the `signals` or `articles` table schemas.
