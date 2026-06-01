import json

from config.settings import HIGH_BRIEF_CONFIDENCE, MIN_BRIEF_CONFIDENCE


LOW_CONFIDENCE_WARNING = (
    "⚠️ Findings below are based on limited data and should be treated as early hypotheses."
)


def confidence_label(confidence: float) -> str:
    if confidence >= HIGH_BRIEF_CONFIDENCE:
        return "high signal readiness"
    if confidence >= MIN_BRIEF_CONFIDENCE:
        return "medium signal readiness"
    return "exploratory finding"


def build_json_input_block(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


ANALYST_TEMPLATE = """IMPORTANT: Your entire response MUST be in English. Do not use Czech or any other language.

You are the Analyst in a three-step research brief pipeline.
Work only from the structured evidence in the INPUT JSON.
Return facts, numeric changes, cited clusters/articles, and evidence gaps only.
Do not explain implications, strategy, or recommendations.
Do not invent evidence that is not present in the input.

INPUT JSON:
{input_json}

Return ONLY valid JSON matching this exact schema:
{{
  "facts": [
    "<grounded factual statement>"
  ],
  "numeric_changes": [
    "<numeric change with direction and scope>"
  ],
  "cited_clusters": [
    "<track_id or observation_id>"
  ],
  "cited_articles": [
    "<article_id>"
  ],
  "evidence_gaps": [
    "<missing evidence or uncertainty>"
  ]
}}"""


EXPLAINER_TEMPLATE = """IMPORTANT: Your entire response MUST be in English. Do not use Czech or any other language.

You are the Explainer in a three-step research brief pipeline.
Use the analyst output and segment rollups to explain what changed, for whom, and where uncertainty remains.
Stay grounded in the input. Do not add new facts or citations.

INPUT JSON:
{input_json}

Return ONLY valid JSON matching this exact schema:
{{
  "what_changed": "<plain-language description of the main change>",
  "for_whom": "<who is most affected and who is secondary>",
  "uncertainty_and_caveats": [
    "<uncertainty or caveat>"
  ]
}}"""


WRITER_TEMPLATE = """IMPORTANT: Your entire response MUST be in English. Do not use Czech or any other language.

You are the Writer in a three-step research brief pipeline.
Use only the analyst output, explainer output, and deterministic calibration weights in the INPUT JSON.
Return the existing ResearchBrief core fields only. Do not include metadata fields.
Use ONLY canonical segment keys in `most_affected_segment` and every hypothesis `segment` field:
`young_urban`, `family`, `senior`, `b2b`.
Use ONLY these drift types:
`concern_spike`, `purchase_surge`, `avoidance_rise`, `frame_shift`, `mixed`, `stable`.
Alert level must be one of `none`, `mild`, or `strong`.
Write exactly 3 hypotheses.
Do not mention purchase behavior if `purchase_intent` is not in the relevant fields list.

INPUT JSON:
{input_json}

Return ONLY valid JSON matching this exact schema:
{{
  "headline": "<8-12 word summary of dominant coverage shift>",
  "narrative": "<2-3 sentence plain-language brief for a Lakmoos analyst>",
  "most_affected_segment": "<young_urban | family | senior | b2b>",
  "drift_type": "<concern_spike | purchase_surge | avoidance_rise | frame_shift | mixed | stable>",
  "alert_level": "<none | mild | strong>",
  "hypotheses": [
    {{
      "segment": "<young_urban | family | senior | b2b>",
      "hypothesis": "<falsifiable follow-up hypothesis>",
      "signal_basis": "<grounded signal basis>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<young_urban | family | senior | b2b>",
      "hypothesis": "<falsifiable follow-up hypothesis>",
      "signal_basis": "<grounded signal basis>",
      "suggested_question": "<one Likert-scale survey question>"
    }},
    {{
      "segment": "<young_urban | family | senior | b2b>",
      "hypothesis": "<falsifiable follow-up hypothesis>",
      "signal_basis": "<grounded signal basis>",
      "suggested_question": "<one Likert-scale survey question>"
    }}
  ]
}}"""
