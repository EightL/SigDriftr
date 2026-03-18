# 04_trust_core — Drift & Brief Trust Layer

## Goal
One-sentence goal: Make SigDriftr’s drift metrics and research briefs explicitly reflect data quality and domain relevance, so analysts never see a "strong" finding without sufficient evidence.

## Where this fits in the pipeline
- Sits on top of existing aggregation and drift logic in `delta/mapper.py` and `delta/engine.py`.
- Feeds into brief generation in `brief/generator.py` and `brief/models.py`.
- Surfaces through API routes in `api/routes/brief.py` and `api/routes/pipeline.py`, and is consumed by the demo UI in `static/index.html`.

## Inputs / Outputs
- **Inputs**
  - Existing `segment_profiles` rows computed over a sliding time window (`delta/mapper.py`).
  - Existing `baselines` rows, including `sample_count`, `is_learned`, and `updated_at` (`delta/engine.py`).
  - Domain configuration and signal relevance from `config/domains.py`.
  - Confidence thresholds and article-count settings from `config/settings.py`.
- **Outputs**
  - Extended drift results including a simple per-segment `status` field and clearer metadata about when data is missing or low-confidence.
  - Extended `ResearchBrief` model with a top-level `status` field (`"insufficient_data" | "warming" | "ready"`).
  - A new "insufficient data" path in brief generation that avoids calling the LLM when evidence is clearly inadequate.
  - Pipeline/ UI wiring so an analyst can see at a glance whether a brief is exploratory or production-ready.

## Design notes informed by recent work
- Concept drift surveys for text streams emphasize sliding-window profiles and explicit drift metadata (e.g., differentiating "no data" from genuine change), rather than a single opaque score.[web:21][web:27]
- Production LLM monitoring guides (e.g., AWS embedding drift examples) recommend computing metrics against a baseline plus explicit confidence / coverage indicators, not just point estimates.[web:20][web:23]
- Public-opinion and synthetic-survey work with LLMs stresses that model-generated summaries should carry clear caveats when based on sparse or biased evidence, to avoid overconfident narratives.[web:22][web:31][web:25]
- SigDriftr already incorporates domain-aware signal weights, per-segment confidence scores, and baseline learning; this plan focuses on making those signals first-class in the public API and brief model, and on adding an explicit "insufficient data" branch rather than always emitting a confident-looking brief.[cite:36][cite:37][cite:38][cite:39][cite:44][cite:46]

## Numbered steps

1. **Add explicit per-segment `status` to drift results**
   - File: `delta/engine.py`.
   - For each segment result in `compute_drift`, derive a simple status string from existing fields:
     - `"no_data"` when `article_count == 0`.
     - `"warming"` when `article_count > 0` but either `baseline_is_learned` is false or `confidence < MIN_BRIEF_CONFIDENCE`.
     - `"ready"` when `article_count > 0`, `baseline_is_learned` is true, and `confidence >= MIN_BRIEF_CONFIDENCE`.
   - Add a `status` key to each result dict returned by `compute_drift` and include it in the no-baseline and no-data branches as well, so the field is always present.
   - Keep existing numeric fields (`drift_magnitude`, `confidence`, `baseline_*`, `relevant_fields`) unchanged for backward compatibility.
   - Update or add a focused unit test in `test_delta.py` to assert that status values behave as expected for:
     - a segment with zero articles → `status == "no_data"`.
     - a segment with articles but only seeded baselines / low confidence → `status == "warming"`.
     - a segment with learned baseline and decent coverage → `status == "ready"`.

2. **Extend `ResearchBrief` with a top-level `status` field**
   - File: `brief/models.py`.
   - Add a new field to `ResearchBrief`:
     - `status: Literal["insufficient_data", "warming", "ready"]` with a short description.
   - Ensure the field has no default (so all callers must set it) or, if a default is needed for migration, default to `"warming"` and immediately override in generation paths.
   - Regenerate or update any type hints / tests that construct `ResearchBrief` instances (see `test_brief.py`) to include a `status` value.

3. **Implement an "insufficient data" path in brief generation**
   - File: `brief/generator.py`.
   - In `generate_brief`:
     - After computing `drift = compute_drift(real_topic)`, derive:
       - `total_articles = sum(entry["article_count"] for entry in drift)`.
       - `ready_segments = [entry for entry in drift if entry.get("confidence", 0.0) >= MIN_BRIEF_CONFIDENCE and entry.get("has_data", False)]`.
     - If `not drift` or `total_articles == 0`, immediately construct and return a `ResearchBrief` with:
       - `status="insufficient_data"`.
       - `alert_level="none"`, `drift_type="stable"`.
       - A short, deterministic `headline` and `narrative` explaining that there are no recent articles for this topic and suggesting the analyst rerun the pipeline later.
       - Three simple, generic `SurveyHypothesis` entries that focus on data collection (e.g., "Will awareness move once coverage increases?").
       - `confidence_context` built as usual but likely empty maps.
       - **Skip** any call to `_call_ollama_json` in this branch to avoid unnecessary model usage.
     - Else if there are some articles but **all** segments have `confidence < MIN_BRIEF_CONFIDENCE`, set `brief.status = "warming"` after normal brief generation.
     - Else (at least one reasonably confident segment), set `brief.status = "ready"`.
   - Make sure the `_fallback_brief` helper also sets an appropriate `status` when used:
     - If called in an otherwise ready scenario → `status="ready"`.
     - If called when all segment confidences are low → `status="warming"`.
   - Update `_apply_confidence_language` to **not** override `status`, but continue to prefix the narrative with qualitative labels ("exploratory finding" etc.) based on the same confidence thresholds.
   - Adjust `generate_brief_cached` so it simply caches and returns the new `ResearchBrief` objects; the caching key and TTL stay unchanged.

4. **Expose brief status through the API and UI**
   - File: `api/routes/brief.py`.
     - The existing route likely returns a `ResearchBrief` model directly; no change needed beyond the new field, but add a brief docstring comment that `status` and `confidence_context` indicate data quality.
   - File: `api/routes/pipeline.py`.
     - In the `/pipeline/run` response dict, add a new `"brief_status"` key set to `brief.status`.
   - File: `static/index.html`.
     - Update the client-side rendering to:
       - Show a small label or banner near the brief: e.g., "Status: insufficient_data / warming / ready".
       - When `brief_status === "insufficient_data"`, visually downplay the brief (e.g., grey box or a warning icon) and optionally add a short explanatory line like "Too little recent coverage; this is a data-availability notice, not a drift finding.".
       - When `brief_status === "warming"`, indicate that drift is still stabilizing (e.g., "warming up — interpret with caution").
       - When `brief_status === "ready"`, show the existing layout unchanged except for the label.
   - Extend `test_ui.py` or add a lightweight new frontend-focused test (if feasible within the existing testing approach) that at least verifies the `/brief/{topic}` and `/pipeline/run` JSON shapes include the new `status` field.

5. **Align tests with the new trust semantics**
   - File: `test_brief.py`.
     - Add tests that:
       - For a topic with no articles in the DB, `generate_brief("topic")` returns a `ResearchBrief` with `status == "insufficient_data"` and `alert_level == "none"` without attempting to call the LLM (use monkeypatch or a stub to assert `_call_ollama_json` is not invoked).
       - For a topic with a few low-confidence segments (e.g., sample baselines, small article_count), `status == "warming"` and the narrative contains the low-confidence warning string from `LOW_CONFIDENCE_WARNING`.
       - For a topic with enough historical data and learned baselines (you can reuse existing fixtures or helper functions in `test_baseline_confidence.py`), `status == "ready"`.
   - File: `test_pipeline.py`.
     - Extend the end-to-end pipeline test to assert that `/pipeline/run` includes `"brief_status"` and that its value is one of the expected literals.

## File paths touched
- `delta/engine.py`
- `brief/models.py`
- `brief/generator.py`
- `api/routes/brief.py`
- `api/routes/pipeline.py`
- `static/index.html`
- `test_delta.py`
- `test_brief.py`
- `test_pipeline.py`
- Optionally: `test_ui.py`

## pip dependencies
- No new pip dependencies.
- Reuse existing stack (`fastapi`, `pydantic`, `uvicorn`, `feedparser`, `tenacity`, `sentence-transformers`, `torch`, etc.).[cite:3]

## One test to confirm it works end-to-end
- Scenario: cold-start topic with no articles yet.
  1. Start the API with a fresh SQLite DB.
  2. Call `GET /brief/some_new_topic` or run `POST /pipeline/run?topic=some_new_topic`.
  3. Confirm that the response JSON includes:
     - `status == "insufficient_data"` in the brief payload.
     - `alert_level == "none"`.
     - `confidence_context` present but with empty or zeroed maps.
  4. After running the scheduler or manually calling `POST /collect` and `POST /extract` enough times to accumulate data and warm up baselines, repeat the call and confirm that:
     - `status` transitions to `"warming"` and eventually `"ready"` once confidence and baseline learning criteria are met.
     - The UI label in `/ui` reflects the updated status for the topic.
