SEGMENT_LABELS = {
    "young_urban": "young urban adults (18-35, city dwellers, digital-first)",
    "family": "family households (parents with children, suburban/rural)",
    "senior": "seniors (60+, traditional media consumers)",
    "b2b": "business decision-makers (SME owners, executives)",
}

FRAME_MEANINGS = {
    "fear": "articles emphasise threats, danger, and negative outcomes",
    "opportunity": "articles emphasise gains, positive futures, and solutions",
    "conflict": "articles emphasise disagreement, opposition, and tension",
    "neutral": "articles are balanced or informational without strong framing",
}


def build_context_block(drift_results: list[dict], top_articles: list[dict]) -> str:
    """Format drift data and grounding snippets into a compact prompt block."""
    lines = ["## Drift Evidence"]
    for drift in drift_results:
        segment = drift["segment"]
        label = SEGMENT_LABELS.get(segment, segment)
        deltas = drift.get("deltas", {})
        frame = drift.get("dominant_frame", "neutral")
        frame_desc = FRAME_MEANINGS.get(frame, frame)
        frame_shift = drift.get("frame_shift", False)
        alert = drift.get("alert_level", "none")
        article_count = drift.get("article_count", 0)

        lines.append(f"\n### Segment: {label}")
        lines.append(f"- Articles analysed: {article_count}")
        lines.append(f"- Alert level: {alert}")
        lines.append(f"- concern_level delta: {deltas.get('concern_level', 0.0):+.3f}")
        lines.append(
            f"- purchase_intent delta: {deltas.get('purchase_intent', 0.0):+.3f}"
        )
        lines.append(
            f"- avoidance_signals delta: {deltas.get('avoidance_signals', 0.0):+.3f}"
        )
        lines.append(f"- Dominant frame: {frame} ({frame_desc})")
        lines.append(f"- Frame shifted vs baseline: {frame_shift}")

    if top_articles:
        lines.append("\n## Representative Article Snippets (grounding evidence)")
        for index, article in enumerate(top_articles[:5], 1):
            title = (article.get("title") or "[no title]")[:120]
            summary = (article.get("summary") or "")[:200]
            lines.append(
                f"{index}. [{article.get('segment', '?')}] {title} — {summary}"
            )

    return "\n".join(lines)


BRIEF_TEMPLATE = """IMPORTANT: Your entire response MUST be in English. Do not use Czech or any other language.

You are a Lakmoos AI research analyst writing a calibration brief.

Topic: {topic}
Date: {date}

{context_block}

## Output Instructions
Return ONLY valid JSON matching this exact schema. No markdown, no prose outside the JSON.
Reason from the evidence in the drift summary and representative article snippets.
Choose the most affected segment based on drift magnitude, alert level, and article grounding.
Write exactly 3 hypotheses covering the most relevant segments.
Use ONLY canonical segment keys in `most_affected_segment` and every hypothesis `segment` field:
`young_urban`, `family`, `senior`, `b2b`.
Never use human-readable labels such as "young urban adults", "seniors", or "business decision-makers".

{{
  "topic": "{topic}",
  "headline": "<8-12 word summary of dominant behavioral shift>",
  "narrative": "<2-3 sentence plain-language brief for a Lakmoos analyst>",
  "most_affected_segment": "<MUST be exactly one of these strings: young_urban, family, senior, b2b>",
  "drift_type": "<MUST be exactly one of these strings: concern_spike, purchase_surge, avoidance_rise, frame_shift, mixed, stable>",
  "alert_level": "<one of: none | mild | strong>",
  "hypotheses": [
    {{
      "segment": "<MUST be exactly one of: young_urban, family, senior, b2b>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. concern_level +0.18>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<MUST be exactly one of: young_urban, family, senior, b2b>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. purchase_intent -0.12>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<MUST be exactly one of: young_urban, family, senior, b2b>",
      "hypothesis": "<falsifiable behavioral hypothesis>",
      "signal_basis": "<e.g. avoidance_signals +0.09>",
      "suggested_question": "<one Likert-scale survey question>"
    }}
  ],
  "generated_at": "{date}",
  "model_used": "{model}"
}}"""
