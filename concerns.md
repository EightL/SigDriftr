# SigDriftr Design Concerns

## Fixed Schema Applied Universally
- `purchase_intent` meaningless for politics/health topics — LLM extracts noise
- All three signal fields (concern, purchase_intent, avoidance) treated equally in drift calculation
- Drift magnitude sums all deltas uniformly, so irrelevant field movements pollute the score
- Brief generation picks largest delta — random `purchase_intent` spike on politics topic could wrongly highlight "purchase surge"
- Fields waste LLM reasoning tokens on semantically invalid extractions

**Potential fixes:**
- Topic-aware schema: pass topic to LLM so it selects relevant fields
- Domain-specific schemas (commerce vs. politics vs. health)
- Field relevance weighting: store relevance score per field per topic, zero out during aggregation

## Feed Configuration & Generalization
- Feeds hardcoded to Czech outlets in config/feeds.py
- Each feed has manual `affinity_tag` priors (e.g., family, b2b, senior) that bias segment relevance
- Cannot scale: adding BBC, Reuters, etc. requires knowing their actual audience seg distribution — editorial judgment
- Affinity priors stored globally but not learned or validated against real engagement

**How to generalize?**
- Move feeds to database with columns: rss_url, outlet, affinity_tag, region, language
- AI agent profiles new feeds: fetch 5 sample articles, ask LLM to estimate audience seg distribution
- Default to "mainstream" (uniform 0.25 priors) until tuned by real behavior
- **Open question:** How to know if LLM-guessed affinity is correct? Need offline calibration phase (run 100 articles, measure actual segment engagement)

## RSS as Data Source
- RSS chosen for demo simplicity (no API keys, free, immediate access)
- Cannot backfill historical data — feeds only expose last 10-50 articles
- Time window queries (e.g., `?days_back=90`) only work if articles were collected during that period
- No way to retrieve data older than what's already in DB unless continuous collection is running
- In production, would need: NewsAPI, Mediastack, or custom scraper with archive support

## Baselines Are Hardcoded, Not Learned
- Baselines in delta/seeder.py are arbitrary editorial guesses (e.g., young_urban concern=0.38, purchase_intent=0.31)
- Not validated against actual behavioral data
- Drift calculations compare against fictional "normal" states, not observed reality
- If real young_urban concern is 0.50 but seeded to 0.38, false positive drift appears when baseline is wrong
- No warmup period to learn baselines from historical data  
- `update_baseline_from_profile()` exists but nothing calls it — no mechanism to blend learned values over time

**How to fix?**
- Implement warmup period: run system 3-6 months collecting real signals before drift analysis
- Auto-learn baselines: compute per-(topic,segment) aggregates after N articles, replace seed values
- Enable `update_baseline_from_profile()` callpath with blend ratios (0.8 old / 0.2 new) to gradually update
- Store baseline confidence/sample_count to track trustworthiness of learned values

## Brief Generation Inherits Upstream Errors
- Brief is well-engineered (good normalization, fallback logic, Pydantic validation)
- But it synthesizes potentially garbage drift data into authoritative-sounding narratives
- Upstream problems flow downhill: fixed schema pollutes drift → wrong segments ranked high → brief makes false conclusions sound legitimate
- Example: family segment shows "strong drift" only because purchase_intent noise inflated it (irrelevant field) + bad baseline; brief then confidently suggests "shopping behavior changing"
- Fallback brief also uses the same broken drift magnitude to rank segments
- **Compounding problem:** brief is presented as definitive even though based on guesses (hardcoded baselines, wrong fields for topic)
- 30-minute cache means brief stale if new articles arrive
- No confidence/uncertainty indicators — user can't tell if brief is based on solid data or speculation

**How to fix?**
- Add "confidence" scores that propagate from baseline quality + article count + field relevance
- Surface warnings: "Drift based on guessed baseline, low confidence until X days of history"
- Disable brief generation until warmup period or min article count threshold reached
- Include drift_magnitude in brief output so user can judge signal strength themselves