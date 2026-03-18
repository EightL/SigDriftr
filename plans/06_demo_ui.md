# 07_demo_ui — SigDriftr Demo Visualisation Page

**Goal:** Build a single self-contained `static/index.html` demo page that makes the SigDriftr pipeline legible to a technical audience in under 30 seconds — no frameworks, no AI-slop visuals, minimal and data-forward.

**Where it fits:** This is a presentation/demo layer only. It calls the existing FastAPI backend. No backend changes required except mounting the static directory.

---

## Inputs / Outputs

- **Input:** Running FastAPI backend at `http://localhost:8000`
- **Output:** `static/index.html` — single file, all CSS and JS inline, zero build step
- **Backend mount:** `main.py` updated to serve `static/` at `/ui`

---

## Visual Design Rules

These are strict — do not deviate:

- Background: `#0d0d0d`
- Text: `#e0e0e0`
- Accent / highlight: `#00ff88` (one color only)
- Danger / high drift: `#ff4444`
- Warning / mild drift: `#ffaa00`
- Font: `monospace` everywhere — `font-family: 'Courier New', monospace`
- No rounded corners (`border-radius: 0` everywhere)
- No shadows, no gradients, no icons, no emoji
- Borders: `1px solid #333`
- Tables: tight, `border-collapse: collapse`, alternating row bg `#111` / `#0d0d0d`
- Chart: Chart.js radar — no fill, just strokes. Grid lines `#222`. Tick labels monospace.
- Layout: single column, max-width 900px, centered, 24px padding
- Section headers: uppercase, letter-spacing 2px, `color: #555`, `font-size: 11px`
- Values: `color: #00ff88` for numbers, white for labels

---

## Sections (top to bottom)

### 1. Header bar
```
SIGDRIFTR          [ db: ok  articles: 1204  signals: 1204  ollama: ok ]
```
- Left: `SIGDRIFTR` in `#00ff88`, 18px
- Right: health status from `GET /health` — each key:value pair inline, status colored green/red
- Polls `/health` once on load only
- Thin `1px solid #333` bottom border

### 2. Topic input
```
TOPIC  [________________energia_____________]  [ANALYZE]
```
- Plain text input, dark bg, monospace, `#00ff88` caret
- `[ANALYZE]` button — flat, `border: 1px solid #00ff88`, `color: #00ff88`, bg transparent
- On click: runs the full pipeline sequence (see Step 7 below)
- Shows a log line below the input while running: `> collecting...`, `> extracting...`, `> computing drift...`

### 3. Drift heatmap table

Title: `SEGMENT DRIFT — last 7 days`

A plain HTML table:

| SEGMENT | CONCERN Δ | PURCHASE_INTENT Δ | AVOIDANCE Δ | DRIFT MAG | FRAME SHIFT | ALERT |
|---|---|---|---|---|---|---|
| young_urban | +0.12 | -0.03 | +0.22 | 0.37 | YES | MILD |
| family | ... | | | | | |
| senior | ... | | | | | |
| b2b | ... | | | | | |

- Data source: `GET /drift/{topic}` response `.segments[]`
- Delta values: from `.deltas.concern_level`, `.deltas.purchase_intent`, `.deltas.avoidance_signals`
- Color rules for cells:
  - Delta positive: `#ff4444`, negative: `#4488ff`, zero: `#555`
  - ALERT `strong` → `#ff4444`, `mild` → `#ffaa00`, `none` → `#555`
  - FRAME SHIFT `true` → `#ffaa00 YES`, `false` → `#333 —`
- If `has_data: false` for a row, show entire row in `#333` with `— NO DATA —`

### 4. Radar chart

Title: `SIGNAL PROFILE — {segment} / current vs baseline`

- A `<select>` dropdown to pick segment: `young_urban | family | senior | b2b`
- Chart.js radar, canvas 380×380, centered
- 3 axes: `CONCERN`, `PURCHASE_INTENT`, `AVOIDANCE`
- Two datasets:
  - `CURRENT` — stroke `#00ff88`, no fill
  - `BASELINE` — stroke `#444`, no fill, dashed
- On segment change, redraw with data from the already-fetched `/drift/{topic}` response
- No legend box — label the two lines directly in a `<div>` below: `— CURRENT   - - BASELINE`

### 5. Recent signals log

Title: `RECENT SIGNALS — {topic}`

A scrollable `<div>` (max-height 220px, overflow-y scroll), monospace, 12px:
```
2026-03-17T18:44:01Z  [fear      ]  concern=0.81  intent=0.12  avoid=0.44  young_urban=0.61
2026-03-17T18:43:55Z  [neutral   ]  concern=0.22  intent=0.55  avoid=0.08  b2b=0.71
```
- Data source: `GET /signals?topic={topic}` — show top 30, newest first
- Each line is one article signal
- `dominant_frame` padded to 9 chars with spaces
- The segment shown per line is whichever `seg_*` value is highest
- No click interaction needed

### 6. Research brief

Title: `RESEARCH BRIEF — {topic}`

- Data source: `GET /brief/{topic}`
- Render the brief response as plain preformatted text inside a `<pre>` block
- `color: #aaa`, `border-left: 2px solid #333`, `padding-left: 16px`
- If the brief has a `hypotheses` array, render each as a numbered line: `[1] ...`
- If the API returns an error or empty, show: `> no brief available for this topic`

---

## Step-by-step Implementation

1. Create directory `static/` at repo root.

2. Create `static/index.html` — single file, all CSS in `<style>`, all JS in `<script>` at bottom of body.

3. **On page load:**
   - Call `GET /health` → populate header bar
   - Pre-fill topic input with `"energie"` as default
   - Do NOT auto-run the full analyze pipeline on load (avoid hammering Ollama)

4. **On ANALYZE click:**
   - Disable button, show log line `> collecting...`
   - `POST /collect?topic={topic}` — await
   - Update log: `> extracting...`
   - `POST /extract?topic={topic}` — await
   - Update log: `> computing drift...`
   - `GET /drift/{topic}` — await, store result as `driftData`
   - `GET /signals?topic={topic}` — await, store as `signalsData`
   - `GET /brief/{topic}` — await, store as `briefData`
   - Update log: `> done. {N} signals processed.`
   - Re-enable button
   - Render all sections with new data

5. **Drift heatmap:** build table rows from `driftData.segments`. Format delta numbers with explicit `+` sign using `(v >= 0 ? '+' : '') + v.toFixed(3)`.

6. **Radar chart:** initialize Chart.js radar on first render. On re-render, call `chart.data.datasets[...].data = [...]` and `chart.update()`. Do not destroy/recreate.

7. **Signals log:** map `signalsData.slice(0, 30)` to formatted log lines. Find dominant segment with `Object.entries(row).filter(([k]) => k.startsWith('seg_')).sort(([,a],[,b]) => b-a)[0]`.

8. **Brief:** render `briefData.hypotheses` array if present, else stringify the whole response inside `<pre>`.

9. **Segment dropdown change:** only update the radar chart — do not re-fetch.

10. **Mount static files in `main.py`:**
    ```python
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
    ```
    Add this after all `include_router` calls. Add `aiofiles` to `requirements.txt`.

---

## File Paths

- `static/index.html` — the entire UI
- `main.py` — add static mount (modify existing file)
- `requirements.txt` — add `aiofiles`

---

## pip dependencies

```
aiofiles
```
(Chart.js loaded from CDN: `https://cdn.jsdelivr.net/npm/chart.js`)

---

## Test to confirm it works

1. Start server: `uvicorn main:app --reload`
2. Open `http://localhost:8000/ui`
3. Type `energie` in the topic box, click ANALYZE
4. After pipeline completes, all 4 sections (heatmap, radar, signals log, brief) must be populated with non-empty data
5. Change segment dropdown — radar must redraw without a new network request
6. Header bar must show `db: ok` and `ollama: ok`
