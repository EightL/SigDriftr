┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: INGESTION                                          │
│  RSS feeds (CZ + DE + International)                         │
│  ↓ LinUCB Bandit (keep yours — it's unique)                  │
│  Output: raw articles {title, body, url, source, date}       │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: EMBEDDING  ← NEW                                   │
│  Model: all-MiniLM-L6-v2 (tiny, fast, free, local)          │
│  Embed title + first 2 sentences of each article             │
│  Output: article + 384-dim vector                            │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: CLUSTERING  ← NEW                                  │
│  Algorithm: HDBSCAN (no need to pre-set # of clusters)       │
│  Groups articles by semantic similarity into "story clusters" │
│  Each cluster = one real-world event/story                   │
│  Output: clusters of articles, each with a centroid vector   │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: SIGNAL EXTRACTION (keep yours, upgrade slightly)   │
│  Per cluster → LLM extracts:                                 │
│  - topic label                                               │
│  - dominant_frame, concern_level, sentiment                  │
│  - audience segment scores (keep your seg_ fields)          │
│  Output: structured signal per cluster (not per article)     │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 5: DRIFT DETECTION (keep yours, make it centroid-aware│
│  Compare cluster centroids today vs stored baseline centroids│
│  Drift score = cosine distance between centroids over time   │
│  + your existing segment aggregation + baseline comparison   │
│  Output: drift alerts {topic, segment, magnitude, direction} │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 6: HIERARCHICAL BRIEF  ← UPGRADE                      │
│  3 LLM calls (multi-agent style):                            │
│  Call 1 — "Analyst": list facts + numbers per drifting topic │
│  Call 2 — "Explainer": explain what changed and for whom     │
│  Call 3 — "Writer": produce final brief with hypotheses      │
│  Output: research brief + calibration weights                │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 7: OUTPUT (keep yours)                                │
│  /brief  /signals  /calibration  /drift-dashboard  UI        │
└──────────────────────────────────────────────────────────────┘
