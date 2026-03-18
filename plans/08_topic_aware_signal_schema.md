# 08_topic_aware_signal_schema

**Goal:** Make SigDriftr's extraction and drift layers domain-aware — so that `purchase_intent` never contaminates drift scores for political or health topics, irrelevant signals are masked before they reach baselines, and drift magnitude is weighted by what actually matters for the topic at hand.

**Where it fits in the pipeline:** Cross-cuts `extraction/llm_client.py`, `extraction/extractor.py`, `delta/engine.py`, `delta/mapper.py`, `brief/prompt.py`, and API response models. Must be implemented after Phase 1 (baseline confidence layer) is stable. Does NOT require changes to the DB schema.

---

## Research Rationale

The core problem: the current `PROMPT_TEMPLATE` in `llm_client.py` has no awareness of topic domain. The LLM is asked for `purchase_intent` regardless of whether the article is about elections or energy prices — and the Phase 1 confidence system will now faithfully track drift in a field that was never meaningful. This creates ghost signals.

Recent research confirms this pattern is a real failure mode:

- **Goal-oriented prompt engineering** (Li et al., ACL survey 2024) shows that decomposing prompts by goal/sub-goal — here: "what domain is this?" then "which signals apply?" — yields consistent accuracy gains on structured extraction tasks [web:55]. Telling the LLM explicitly which fields matter for a given topic is the cheapest and most robust fix.
- **Persona/domain prompting for classification** (Freise et al., 2025, arXiv:2502.03078) found that domain-conditioned prompts — where the model is told what kind of analyst it is and what to focus on — outperform generic prompts for behaviorally specific constructs like concern and avoidance [web:79]. This is directly applicable to SigDriftr's extraction prompt.
- **Weighted drift magnitude** is the right design: the DIS methodology for NLP drift (JMIR 2024) recommends field-specific weighting before summing drift scores, rather than treating all features as equally informative — Jensen-Shannon divergence per field, then weighted sum [web:71].

---

## Inputs

- `extraction/llm_client.py` — `PROMPT_TEMPLATE`, `extract_signals()`, `_normalize_signals()`
- `extraction/extractor.py` — `run_extraction()` call site
- `delta/engine.py` — `compute_drift()`, drift magnitude calculation
- `delta/mapper.py` — `compute_segment_profiles()`, `SIGNAL_KEYS`
- `brief/prompt.py` — `build_context_block()`, `BRIEF_TEMPLATE`
- `api/models.py` — response Pydantic models
- `api/routes/signals.py`, `api/routes/calibration.py` — route handlers
- `config/settings.py` — for new constants

## Outputs

- New module `config/domains.py` with domain definitions and signal weight tables
- `topic_to_domain(topic)` classifier function (keyword rules + LLM fallback)
- `extract_signals()` extended with `topic` parameter; domain mask applied post-normalization
- `compute_drift()` uses weighted drift magnitude per domain
- API responses include `domain` and `relevant_fields`
- Brief prompt mentions domain and omits irrelevant field commentary
- All logic covered by unit + regression tests

---

## Steps

### 1. Create `config/domains.py`

**New file:** `config/domains.py`

1.1. Define the four domains and their relevant signal fields:
```python
DOMAINS = {
    "commerce": {
        "relevant_fields": ["concern_level", "purchase_intent", "avoidance_signals"],
        "signal_weights": {"concern_level": 0.4, "purchase_intent": 0.4, "avoidance_signals": 0.2},
        "prompt_hint": "Focus on consumer behavior: purchasing decisions, product avoidance, financial concerns.",
    },
    "civic": {
        "relevant_fields": ["concern_level", "avoidance_signals"],
        "signal_weights": {"concern_level": 0.6, "purchase_intent": 0.0, "avoidance_signals": 0.4},
        "prompt_hint": "Focus on civic concern and social avoidance behaviors. Purchase intent is not relevant here; keep it near 0.0.",
    },
    "health": {
        "relevant_fields": ["concern_level", "avoidance_signals"],
        "signal_weights": {"concern_level": 0.6, "purchase_intent": 0.0, "avoidance_signals": 0.4},
        "prompt_hint": "Focus on health-related concern and avoidance. Purchase intent is not relevant unless this is about healthcare products; keep it near 0.0.",
    },
    "generic": {
        "relevant_fields": ["concern_level"],
        "signal_weights": {"concern_level": 1.0, "purchase_intent": 0.0, "avoidance_signals": 0.0},
        "prompt_hint": "Focus only on general concern level. Purchase intent and avoidance are not relevant; keep them near 0.0.",
    },
}

DEFAULT_DOMAIN = "generic"
```

1.2. Add a Czech keyword → domain mapping dict:
```python
TOPIC_DOMAIN_RULES: list[tuple[list[str], str]] = [
    # commerce
    (["ceny", "energie", "nafta", "benzin", "inflace", "nakup", "obchod", "trh", "ekonomika", "HDP", "spotreba"], "commerce"),
    # civic
    (["politika", "volby", "vlade", "parlament", "demokracie", "protest", "bezpecnost", "migrace", "EU"], "civic"),
    # health
    (["zdravi", "nemoc", "covid", "vakcina", "nemocnice", "leky", "epidemie", "psychologie"], "health"),
]
```

1.3. Implement `topic_to_domain(topic: str) -> str`:
- Normalize input: `topic.strip().lower()`
- Iterate `TOPIC_DOMAIN_RULES`; if any keyword appears as a substring in the normalized topic, return that domain.
- If no rule matches, return `DEFAULT_DOMAIN` (`"generic"`).
- This function must be pure and fast — no I/O, no LLM calls. Domain classification via LLM is explicitly NOT done in this phase (unnecessary complexity, costs latency for a keyword problem).

1.4. Add a helper `get_domain_config(domain: str) -> dict` that returns the domain config dict from `DOMAINS`, falling back to `DOMAINS[DEFAULT_DOMAIN]` for unknown values.

---

### 2. Extend `extract_signals()` with domain awareness

**File:** `extraction/llm_client.py`

2.1. Update `PROMPT_TEMPLATE` to accept an optional domain hint block:
```python
PROMPT_TEMPLATE = """
You are a behavioral analyst specializing in Czech media. Analyze the article below.

Topic: {topic}
Article title: {title}
Article summary: {summary}

Domain guidance: {domain_hint}

Output ONLY this JSON object (nothing before or after it):
...
""".strip()
```
- `{topic}` and `{domain_hint}` are new format fields. `{domain_hint}` comes from `DOMAINS[domain]["prompt_hint"]`.

2.2. Update `_try_ollama(title, summary, model, topic, domain_hint)` signature to accept and pass `topic` and `domain_hint`.

2.3. Update `extract_signals()` signature:
```python
def extract_signals(title: str, summary: str, affinity_tag: str = "mainstream", topic: str = "") -> dict:
```
- Resolve domain: `domain = topic_to_domain(topic)` (import from `config.domains`).
- Get `domain_hint` from `get_domain_config(domain)["prompt_hint"]`.
- Pass `topic`, `domain_hint` to `_try_ollama` and `_ollama_fallback`.

2.4. After `_normalize_signals()` returns, apply the **domain mask**:
```python
def _apply_domain_mask(signals: dict, domain: str) -> dict:
    """
    Zero out fields not relevant in this domain.
    Adds 'domain' and 'irrelevant_fields' metadata to the signals dict.
    """
    config = get_domain_config(domain)
    relevant = set(config["relevant_fields"])
    irrelevant = []
    for key in ["concern_level", "purchase_intent", "avoidance_signals"]:
        if key not in relevant:
            signals[key] = 0.0
            irrelevant.append(key)
    signals["domain"] = domain
    signals["irrelevant_fields"] = irrelevant
    return signals
```
- Call `_apply_domain_mask(signals, domain)` in `extract_signals()` after `_apply_affinity_prior()` and before returning.
- The `domain` and `irrelevant_fields` keys travel along with the signals dict. The DB INSERT in `extractor.py` must store `domain` in `raw_json` but NOT in a dedicated column (no schema change needed — it goes into the existing `raw_json` TEXT field).

2.5. **`extraction/extractor.py`:** Update the `extract_signals()` call to pass `topic`:
```python
signals = extract_signals(title or "", summary or "", affinity_tag=affinity_tag, topic=topic)
```
`topic` is already in scope from the outer `run_extraction(topic)` parameter.

---

### 3. Weighted drift magnitude in the delta engine

**File:** `delta/engine.py`

3.1. Import `topic_to_domain` and `get_domain_config` from `config.domains`.

3.2. In `compute_drift(topic, days_back)`, resolve domain once at the top:
```python
domain = topic_to_domain(topic)
domain_config = get_domain_config(domain)
signal_weights = domain_config["signal_weights"]
relevant_fields = domain_config["relevant_fields"]
```

3.3. Replace the raw L1 drift magnitude calculation:
```python
# OLD:
drift_magnitude = round(sum(abs(value) for value in deltas.values()), 4)

# NEW:
drift_magnitude = round(
    sum(signal_weights.get(key, 0.0) * abs(delta) for key, delta in deltas.items()),
    4,
)
```

3.4. Attach `domain` and `relevant_fields` to every result dict:
```python
{
    ...,
    "domain": domain,
    "relevant_fields": relevant_fields,
}
```
This must be done in ALL return paths inside `compute_drift` — including the `has_data=False` and `baseline=None` branches.

---

### 4. Propagate domain to API responses

**File:** `api/models.py`

4.1. Add to the drift/signals response Pydantic model:
```python
domain: str = "generic"
relevant_fields: list[str] = []
```
Use `Optional` with defaults so existing API consumers don't break.

**Files:** `api/routes/signals.py`, `api/routes/calibration.py`

4.2. No handler changes needed if the route is already passing the full drift dict through. Verify the dict fields are forwarded to the response model — if Pydantic is using `.model_validate(drift_result)`, it will pick up the new fields automatically. If fields are manually mapped, add `domain` and `relevant_fields` to the mapping.

---

### 5. Brief prompt domain awareness

**File:** `brief/prompt.py`

5.1. In `build_context_block()`, extract `domain` and `relevant_fields` from the first drift result (all segments share the same topic/domain):
```python
domain = drift_results[0].get("domain", "generic") if drift_results else "generic"
relevant_fields = drift_results[0].get("relevant_fields", []) if drift_results else []
```
Add a domain context line at the top of the context block:
```
## Domain Context
Topic domain: {domain} | Relevant signals: {', '.join(relevant_fields)}
```

5.2. Update `BRIEF_TEMPLATE` output instructions to add:
```
Domain: {domain}. Relevant signals for this domain: {relevant_fields}.
Do NOT mention purchase behavior or purchase intent if "purchase_intent" is not in the relevant fields list.
When ranking segments and writing hypotheses, only reference signals in the relevant fields list.
```
Add `{domain}` and `{relevant_fields}` as format parameters to `BRIEF_TEMPLATE`. These must be passed in from wherever `BRIEF_TEMPLATE.format(...)` is called (check `brief/generator.py` or equivalent).

---

### 6. Tests

**New file:** `test_domains.py`

6.1. **Unit test: `topic_to_domain` keyword mapping**
```python
assert topic_to_domain("inflace") == "commerce"
assert topic_to_domain("politika") == "civic"
assert topic_to_domain("zdravi") == "health"
assert topic_to_domain("pocasi") == "generic"   # no match → default
assert topic_to_domain("") == "generic"
assert topic_to_domain("INFLACE") == "commerce"  # case insensitive
```

6.2. **Unit test: domain mask zeroes irrelevant fields**
- Call `_apply_domain_mask({"concern_level": 0.5, "purchase_intent": 0.8, "avoidance_signals": 0.3}, "civic")`
- Assert `purchase_intent == 0.0`
- Assert `"purchase_intent" in result["irrelevant_fields"]`
- Assert `concern_level == 0.5` (unchanged)
- Assert `avoidance_signals == 0.3` (unchanged)

6.3. **Unit test: weighted drift magnitude**
- Construct a fake drift with `deltas = {"concern_level": 0.1, "purchase_intent": 0.5, "avoidance_signals": 0.1}` for domain `"civic"` (weights: concern=0.6, purchase=0.0, avoidance=0.4).
- Expected `drift_magnitude = 0.6 * 0.1 + 0.0 * 0.5 + 0.4 * 0.1 = 0.1`.
- Assert the computed magnitude equals `0.1` (purchase ignored).

6.4. **Regression test: high purchase_intent does NOT inflate civic drift**
- Simulate a profile with `purchase_intent = 0.9` but normal concern/avoidance for topic `"politika"`.
- Run `compute_drift()` with this profile against a seeded baseline.
- Assert `drift_magnitude < 0.3` (purchase not contributing).
- Assert `"domain": "civic"` in the result.
- Assert `"purchase_intent" not in result["relevant_fields"]`.

6.5. **Unit test: domain and relevant_fields appear in all drift result branches**
- Force the `has_data=False` path: compute drift with zero articles.
- Assert `"domain"` key exists in the result.
- Force the `baseline=None` path: compute drift with no seeded baselines.
- Assert `"domain"` key exists in the result.

6.6. **Unit test: brief context block includes domain header**
- Call `build_context_block()` with a drift result that has `domain="civic"`, `relevant_fields=["concern_level", "avoidance_signals"]`.
- Assert the output string contains `"Topic domain: civic"`.
- Assert the output string does NOT contain `"purchase_intent"` in the domain context line.

---

## pip Dependencies

No new pip dependencies. All logic uses stdlib string operations and the existing FastAPI + Pydantic stack.

---

## One Test to Confirm It Works

Run the regression test in step 6.4:

```bash
python -m pytest test_domains.py::test_high_purchase_intent_does_not_inflate_civic_drift -v
```

Expected: `drift_magnitude < 0.3`, `domain == "civic"`, `purchase_intent` absent from `relevant_fields`. This confirms the domain mask and weighted magnitude are both wired end-to-end through extraction → delta engine → result dict.
