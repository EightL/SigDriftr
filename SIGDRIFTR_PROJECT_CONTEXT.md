# SigDriftr — Project Context for AI Agent

## What We Are Building
SigDriftr is a real-time behavioral signal extraction pipeline that watches Czech media and outputs per-audience-segment calibration signals. It tells Lakmoos AI how each demographic group's behavior is drifting based on what they've been reading this week — so their synthetic research panels stay grounded in current media reality.

## Pipeline (in order)

1. **Contextual RL Bandit** — Decides which Czech news outlets to crawl, how often, and for which topics. Uses LinUCB. Reward = behavioral signal density per article. Arms = outlets. Context = (topic, outlet_metadata, time_of_day).

2. **LLM Extraction** — Each article summary is passed to a local LLM (Gemma/Phi-3 via Ollama, or API fallback). Output is structured behavioral JSON:
   { concern_level, purchase_intent, avoidance_signals[], dominant_frame, segment_relevance{young_urban, family, senior, b2b} }

3. **Segment Mapper** — Pre-defined outlet × audience affinity weights (e.g. Echo24 → senior/conservative). Multiplies LLM signal scores by affinity weights → per-segment exposure score per article.

4. **Delta Engine** — Compares this week's aggregated per-segment signal profile against a 4-week rolling baseline stored in SQLite. Outputs drift: what changed, how much, for which segment.

5. **Research Brief Generator (LLM)** — Synthesizes delta into a Lakmoos-style research hypothesis. Input: drift scores + narrative shift labels + dominant frames. Output: plain-language brief + 3 panel survey hypotheses.

6. **FastAPI** — Exposes all stages via clean endpoints:
   POST /collect?topic=...
   GET  /signals?topic=...
   GET  /calibration/{topic}/{segment}
   GET  /brief/{topic}
   GET  /health

## Tech Stack
- Python 3.11+
- FastAPI + Uvicorn
- feedparser + requests (RSS ingestion)
- spaCy cs_core_news_sm (Czech NER + dependency parse)
- HuggingFace transformers: multilingual-MiniLM-L6-v2 (embeddings), distilbart-cnn-12-6 (fallback summaries)
- Ollama (local LLM: Gemma-3-1B or Phi-3-mini)
- SQLite (storage)
- NumPy (LinUCB bandit)

## Key Design Decisions
- Bandit is correct here (not deep RL): source selection has no sequential state dependency, reward is immediate per article.
- LLM extracts behavioral intent (not just sentiment): concern, avoidance, purchase_intent mapped to marketing funnel stages.
- Outlet affinity weights are pre-defined (not learned): one-time setup, avoids cold-start in 24h.
- Drift = behavioral score delta vs rolling baseline: simple, explainable, demo-ready.
- No fine-tuning: inference-only for all models.

## Research Papers to Reference in Architecture
- Contextual bandit: Li et al. (2010) Yahoo! news recommendation (LinUCB)
- Narrative shift: Hybrid Dynamic Topic Models + LLM (2025 CEUR workshop)
- Embedding drift: Evidently AI — 5 methods for embedding drift detection
- Entity KG: Neon — News Entity-Interaction Extraction (2024)
- Intent decomposition: Google Research — Small models, big results (2026)
- Synthetic calibration: Fill In The Gaps — synthetic data calibration (2025)

## One-Line Pitch
"SigDriftr is a real-time media signal pipeline that tells Lakmoos how Czech audience behavior is drifting — so their synthetic panels never go stale."
