# 01_rss_ingestion

## Goal
Implement RSS ingestion from Czech news outlets, store articles deduplicated in SQLite, and expose a `POST /collect?topic=` endpoint.

## Pipeline Position
Stage 1 of 6. All downstream stages read from the `articles` table. Nothing else runs until this is done.

## Inputs
- `topic`: a string query param (e.g. `"ekonomika"`, `"zdravi"`, `"politika"`)

## Outputs
- Populated SQLite table `articles` in `data/sigdriftr.db`, filtered by topic, deduplicated by URL

---

## Steps

### 1. Create `config/feeds.py`
Define `FEEDS` as a list of dicts, each with keys `outlet`, `rss_url`, `affinity_tag`. Include:

| outlet       | rss_url                                                    | affinity_tag |
|--------------|------------------------------------------------------------|--------------|
| irozhlas     | https://www.irozhlas.cz/rss/irozhlas                       | mainstream   |
| irozhlas_eko | https://www.irozhlas.cz/rss/irozhlas/section/ekonomika     | b2b          |
| idnes        | https://servis.idnes.cz/rss.aspx                           | mainstream   |
| novinky      | https://www.novinky.cz/rss                                 | mainstream   |
| e15          | https://www.e15.cz/rss                                     | b2b          |
| blesk        | https://www.blesk.cz/rss                                   | senior       |
| ct24         | https://www.ceskatelevize.cz/rss/                          | mainstream   |

### 2. Create `db/init.py`
On import, connect to `data/sigdriftr.db` (create `data/` dir if missing) and execute:

```sql
CREATE TABLE IF NOT EXISTS articles (
    id           TEXT PRIMARY KEY,
    outlet       TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT,
    url          TEXT UNIQUE NOT NULL,
    topic        TEXT,
    published_at TEXT,
    fetched_at   TEXT NOT NULL
);
```

- `id` = `hashlib.sha256(url.encode()).hexdigest()`
- `fetched_at` = UTC ISO timestamp at insert time
- Expose a module-level `get_conn()` that returns the sqlite3 connection (reuse one connection, thread-safe with `check_same_thread=False`)

### 3. Create `ingestion/crawler.py`
Implement `crawl(topic: str) -> int`:

1. Import `FEEDS` from `config.feeds` and `get_conn` from `db.init`
2. Loop over every feed in `FEEDS`
3. Call `feedparser.parse(feed["rss_url"])` — no timeout needed, feedparser handles it
4. For each entry in `feed.entries`:
   - Skip if `topic.lower()` is not found in `(entry.title + " " + entry.get("summary", "")).lower()`
   - Compute `id = hashlib.sha256(entry.link.encode()).hexdigest()`
   - Run `INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)` — `INSERT OR IGNORE` is the deduplication mechanism
   - Use `entry.get("published", "")` for `published_at`
5. Commit after each outlet
6. Return total count of rows inserted (track via `cursor.rowcount` per insert)

### 4. Create `api/routes/collect.py`
FastAPI `APIRouter` with one route:

```
POST /collect?topic=...
```

Call `crawl(topic)`, return:
```json
{"inserted": <int>, "topic": "<str>"}
```

### 5. Create `main.py`
- Instantiate `FastAPI(title="SigDriftr")`
- On startup (`@app.on_event("startup")`): call `db.init` to ensure table exists
- Include collect router with prefix `""` (no prefix)
- Run with `uvicorn main:app --reload` (do not hardcode uvicorn call inside main.py, use CLI)

---

## File Structure

```
sigdriftr/
├── main.py
├── config/
│   └── feeds.py
├── db/
│   └── init.py
├── ingestion/
│   └── crawler.py
├── api/
│   └── routes/
│       └── collect.py
└── data/
    └── sigdriftr.db        ← created at runtime, add to .gitignore
```

---

## pip Dependencies

```
feedparser
fastapi
uvicorn[standard]
```

---

## Test to Confirm It Works

```bash
uvicorn main:app --reload &
sleep 2
curl -X POST "http://localhost:8000/collect?topic=ekonomika"
# Expected: {"inserted": N, "topic": "ekonomika"} where N >= 10

sqlite3 data/sigdriftr.db "SELECT outlet, title FROM articles LIMIT 5;"
# Expected: 5 rows with non-null outlet and title
```

If N < 10, check that the RSS URLs are reachable and that the topic keyword appears in article titles/summaries. Try a broader topic like `"česk"` to verify the pipeline itself works.
