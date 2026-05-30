# Stage 1: Cross-Source Ingestion + Digest API

## Summary
- Make stage one the foundation for a Perplexity-style news summary flow: better ingestion, richer article records, and one filtered digest endpoint.
- Do **not** start clustering yet. The current repo is missing the data shape stage two needs: DE/global sources, full article bodies, and correct multi-topic article associations.
- Keep the LinUCB bandit, but use it only after narrowing the feed pool by requested filters.

## Key Changes
- Unify feed config into one source registry with `outlet`, `country` (`CZ | DE | GLOBAL`), `language`, `rss_url`, `affinity_tag`, and `enabled`.
- Populate that registry with existing CZ feeds plus at least one DE and one international outlet, preferably 2 per bucket for demo resilience.
- Extend article storage to include full text and source metadata. Minimum new fields: `body`, `country`, `language`, and `canonical_url`.
- Add `topics` and `topic_aliases`, and store topic links in `article_topics`
  with `raw_topic`, `canonical_topic_id`, `relevance_score`, and `matched_at`.
  Stop relying on a single `articles.topic` value for topic filtering.
- Change ingestion so `POST /collect` and `POST /pipeline/run` accept `topic` plus optional `country` and `source`, narrow the eligible feeds, crawl RSS, fetch article pages, extract cleaned body text, and store a topic match score.
- Use a real full-text extractor now, with RSS summary as fallback when page extraction fails.
- Add `GET /summaries` as the public hackathon endpoint. Query params: `topic` required, `country` optional, `source` optional, `limit` optional.
- Return a citation-first digest response: filters used, article count, sources used, top articles, and one LLM-generated summary with bullet points grounded in the selected articles.
- Reuse the existing Ollama path for the digest LLM call in stage one. Do not add a second provider yet.

## Public API / Data Shape
- `POST /collect?topic=...&country=...&source=...`
- `POST /pipeline/run?topic=...&country=...&source=...`
- `GET /summaries?topic=...&country=...&source=...&limit=8`
- `GET /summaries` response should include:
  - `topic`, `requested_topic`, `canonical_topic_id`,
    `canonical_display_name`, `country`, `source`, `article_count`,
    `generated_at`
  - `sources_used`
  - `summary_headline`
  - `summary_text`
  - `key_points`
  - `articles` with `title`, `url`, `outlet`, `country`, `published_at`, `relevance_score`

## Test Plan
- Migration test: old DB upgrades cleanly with new columns/table.
- Ingestion filter test: `country` and `source` only crawl matching feeds.
- Multi-topic dedupe test: same URL can appear under multiple topics via `article_topics` without duplicate article rows.
- Alias history test: `energie` and `energy` share canonical topic history while
  preserving the raw topic strings.
- Full-text fallback test: page extraction failure still stores RSS summary and keeps the article usable.
- Summary endpoint test: response honors filters and includes citations.
- End-to-end mocked-feed test across CZ, DE, and GLOBAL sources.

## Assumptions
- Stage one is backend-only; UI changes can wait.
- Existing drift/brief endpoints should keep working, but topic-filtered queries
  should use canonical topics wherever correctness matters.
- Embeddings/clustering stay out of stage one except for the current relevance scoring step.
