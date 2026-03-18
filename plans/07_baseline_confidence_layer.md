# 07_baseline_confidence_layer

**Goal:** Extend SigDriftr's delta engine with a self-learning baseline and a per-segment confidence score so that drift outputs are trustworthy and qualify themselves — bridging the gap from a demo with seeded baselines to a production-grade signal pipeline.

**Where it fits in the pipeline:** Sits between the delta engine (`delta/engine.py`, `delta/mapper.py`) and the API/brief layer. Feeds updated metadata into `/signals`, `/calibration`, and `/brief` endpoints. Must be implemented after `03_segment_delta` and `04_brief_generator` plans are complete.

---

## Research Rationale

This phase is well-motivated by recent work. The core insight driving the design:

- **Confidence-weighted drift** is now standard in production-grade NLP monitoring. DriftLens (arXiv 2406.17813, 2024) showed that drift detectors that track *how much data they've seen* correlate with real drift far better (≥0.85) than threshold-only methods, and run 5× faster.
- **LLM-based embeddings for drift** (arXiv 2312.02337) demonstrate that using LLM-quality representations for baseline comparison is meaningfully more sensitive to subtle semantic shifts — relevant for validating that your behavioral baselines actually capture meaningful signal.
- The confidence formula proposed here mirrors the **coverage × baseline quality** decomposition used in Evidently AI's 5-method embedding drift benchmark — the empirically best-performing interpretable approach for small-to-medium data regimes like SigDriftr's.

The overall pattern is: **don't produce drift numbers you can't stand behind.** Low-confidence outputs should be phrased as hypotheses; high-confidence outputs as findings.

---

## Inputs

- `db/init.py` — existing `baselines` table schema
- `delta/engine.py` — `compute_drift()` function
- `delta/mapper.py` — segment profile computation and persistence
- `brief/models.py` and `brief/prompt.py` — brief generation layer
- `api/routes.py` (or wherever `/signals`, `/calibration`, `/brief` routes live)
- `config/settings.py` (or equivalent) — for config constants

## Outputs

- Extended `baselines` table with `sample_count` and `is_learned` columns
- `confidence` score attached to every `compute_drift()` result
- API responses for `/signals` and `/calibration/{topic}/{segment}` include confidence metadata
- Brief generator qualifies language based on confidence level
- All changes covered by unit + integration tests

---

## Steps

### 1. Extend the `baselines` database table

**File:** `db/init.py`

1.1. In the `CREATE TABLE baselines` statement, add:
```sql
sample_count INTEGER NOT NULL DEFAULT 0,
is_learned   INTEGER NOT NULL DEFAULT 0
```

1.2. Add a safe migration path that runs at startup for existing databases. In `db/init.py` (or a new `db/migrate.py`), implement `run_migrations(conn)`:
- Check if `sample_count` column exists via `PRAGMA table_info(baselines)`.
- If missing, run: `ALTER TABLE baselines ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0`
- Same for `is_learned`.
- This is idempotent — safe to run on every startup.

1.3. Call `run_migrations(conn)` at the top of the DB initialization sequence before any other DB operations.

---

### 2. Implement `update_baseline_from_profile()`

**File:** `delta/engine.py` (or `delta/baseline.py` if you prefer to split)

2.1. Add a config constant: `MIN_ARTICLES_FOR_BASELINE = 30` — minimum article count before a segment profile may contribute to baseline learning. Place in `config/settings.py`.

2.2. Implement the function:
```python
def update_baseline_from_profile(conn, topic: str, segment: str, profile: dict, article_count: int):
    """
    Blend the current profile into the stored baseline using EMA.
    Only runs if article_count >= MIN_ARTICLES_FOR_BASELINE.
    Updates sample_count and sets is_learned = 1.
    """
```
- Blending formula: `new_baseline[field] = ALPHA * current_baseline[field] + (1 - ALPHA) * profile[field]`
- Use `BASELINE_EMA_ALPHA = 0.8` as config constant (slow-moving baseline). Store in `config/settings.py`.
- After blending, run: `UPDATE baselines SET <fields>, sample_count = sample_count + 1, is_learned = 1, updated_at = <now> WHERE topic = ? AND segment = ?`

2.3. Edge case: if no existing baseline row exists for (topic, segment), do an INSERT with the profile values directly and `sample_count = 1`, `is_learned = 1`.

---

### 3. Wire `update_baseline_from_profile()` into the segment profile path

**File:** `delta/mapper.py` (wherever segment profiles are persisted after computation)

3.1. After segment profiles are written to the DB, iterate over (topic, segment) pairs in the new profile batch.

3.2. For each pair:
- Retrieve `article_count` for this batch (already available from the profile computation context).
- If `article_count >= MIN_ARTICLES_FOR_BASELINE`, call `update_baseline_from_profile(conn, topic, segment, profile, article_count)`.

3.3. Do not call this on every single article — only at the batch/profile aggregation step (likely triggered per `/collect` + `/extract` cycle).

---

### 4. Compute and attach confidence to drift results

**File:** `delta/engine.py` — `compute_drift()` function

4.1. When loading a baseline for drift computation, also SELECT `sample_count`, `is_learned`, and `updated_at`.

4.2. Compute `baseline_age_days = (now - updated_at).days`.

4.3. Compute `confidence`:
```python
coverage = min(1.0, article_count / 50)

if not is_learned:
    baseline_quality = 0.3
else:
    baseline_quality = min(1.0, sample_count / 50) + 0.3
    baseline_quality = min(1.0, baseline_quality)

confidence = max(0.0, min(1.0, 0.5 * coverage + 0.5 * baseline_quality))
```
- These weights (0.5/0.5) and thresholds (50 articles) should be config constants in `config/settings.py`:
  - `CONFIDENCE_COVERAGE_WEIGHT = 0.5`
  - `CONFIDENCE_BASELINE_WEIGHT = 0.5`
  - `CONFIDENCE_ARTICLE_DENOMINATOR = 50`

4.4. Attach to each per-segment drift result object:
```python
{
  ...,
  "confidence": confidence,
  "baseline_is_learned": is_learned,
  "baseline_sample_count": sample_count,
  "baseline_age_days": baseline_age_days
}
```

---

### 5. Expose confidence in API responses

**File:** `api/routes.py` (or wherever route handlers live)

5.1. `/signals?topic=...` — include `confidence`, `baseline_is_learned`, `baseline_sample_count`, `baseline_age_days` per segment in the response payload.

5.2. `/calibration/{topic}/{segment}` — same fields added to the calibration object.

5.3. Update Pydantic response models in `api/models.py` (or wherever they live) to include these new fields with appropriate types and `Optional` defaults so backward compatibility is maintained.

---

### 6. Integrate confidence into the brief generator

**File:** `brief/models.py` and `brief/prompt.py`

6.1. **`brief/models.py`:**
- Extend the internal DTO / input model for the brief generator to accept `segment_confidence: dict[str, float]`, `baseline_is_learned: dict[str, bool]`, and `baseline_sample_count: dict[str, int]`.

6.2. **`brief/prompt.py`:**
- Add a `MIN_BRIEF_CONFIDENCE` config constant (e.g. `0.4`) in `config/settings.py`.
- In the prompt construction, add a per-segment confidence qualifier:
  - `confidence >= 0.7` → "high confidence"
  - `0.4 <= confidence < 0.7` → "medium confidence"
  - `confidence < 0.4` → "exploratory finding"
- If **all** segments have `confidence < MIN_BRIEF_CONFIDENCE`, prepend a warning to the brief output:
  > "⚠️ Findings below are based on limited data and should be treated as early hypotheses."
- The LLM instruction in the prompt template should explicitly say:
  > "For each segment, qualify your language using the confidence level provided. Low-confidence segments must be framed as hypotheses, not conclusions."

---

### 7. Tests

**Files:** `test_delta.py` and/or a new `test_baseline_confidence.py`

7.1. **Unit test: DB migration is idempotent**
- Create a fresh DB.
- Run `run_migrations()` twice.
- Assert that `baselines` table has `sample_count` and `is_learned` columns with correct defaults.
- Assert no errors on second run.

7.2. **Unit test: `update_baseline_from_profile()` blending**
- Create a DB with a seeded baseline (all zeros, `is_learned=0`).
- Call `update_baseline_from_profile()` with a profile having `concern=0.8`.
- Assert `is_learned` flips to 1, `sample_count` increments to 1.
- Assert new baseline value is `0.8 * 0 + 0.2 * 0.8 = 0.16` (EMA with alpha=0.8).

7.3. **Unit test: confidence formula edge cases**
- `article_count=0, is_learned=0` → confidence ≈ 0.15 (pure seed, no data)
- `article_count=50, is_learned=1, sample_count=50` → confidence = 1.0
- `article_count=25, is_learned=0` → coverage=0.5, baseline_quality=0.3 → confidence=0.4

7.4. **Unit test: article count below threshold skips baseline update**
- Call `update_baseline_from_profile()` with `article_count=10` (below MIN=30).
- Assert baseline in DB is unchanged.

7.5. **Integration test: confidence increases over time**
- Start with empty DB.
- Simulate 3 profile computation cycles with increasing article counts (10, 30, 60).
- Call `update_baseline_from_profile()` at cycles 2 and 3.
- Run `compute_drift()` after each cycle.
- Assert: `confidence` increases across cycles, `is_learned` flips to 1 after cycle 2, `baseline_sample_count` increments.

7.6. **Integration test: brief qualification**
- Construct drift input with `confidence=0.2` for all segments.
- Generate brief.
- Assert the output contains the `⚠️` warning text.
- Construct drift input with `confidence=0.9` for all segments.
- Assert no warning, and "high confidence" appears in the brief narrative.

---

## pip Dependencies

No new pip dependencies required. All logic uses:
- `sqlite3` (stdlib)
- `datetime` (stdlib)
- Existing FastAPI + Pydantic stack

---

## One Test to Confirm It Works

Run the integration test in step 7.5:

```bash
python -m pytest test_baseline_confidence.py::test_confidence_increases_over_time -v
```

Expected: all 3 assertions pass — confidence increases, `is_learned` flips, `sample_count` increments. This confirms the entire learning loop is wired end-to-end.
