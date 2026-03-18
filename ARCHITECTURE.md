# SigDriftr Architecture Guide

## Overview

SigDriftr is a media intelligence pipeline that extracts behavioral signals from Czech RSS articles using local LLMs, aggregates them by audience segment, detects drift from baseline behaviors, and generates research briefs. This document explains the design, architecture, and key decision points.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SigDriftr Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  RSS Feeds  →  Ingestion  →  Filtering  →  Storage             │
│  (8 Czech)      (LinUCB)    (Semantic)    (SQLite)              │
│                             (String)                              │
│                      ↓                                            │
│  Extraction (Ollama LLM)  →  Signals  →  Enrichment (spaCy)    │
│  (qwen2.5:7b)              (8-field)      (Entities)             │
│                      ↓                                            │
│  Aggregation  →  Baselines  →  Drift  →  Brief Generation      │
│  (4 segments)    (Seeded)     (Engine)    (Ollama + Fallback)   │
│                      ↓                                            │
│  Dashboard UI  ←  JSON API  ←  Results                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

(See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical documentation)