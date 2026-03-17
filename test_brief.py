#!/usr/bin/env python3
import json
import tempfile
from unittest.mock import patch

import db.init
import pytest

pytest.importorskip("pydantic")

from brief.generator import OLLAMA_MODEL, _extract_json, generate_brief
from brief.models import ResearchBrief
from delta.seeder import seed_baselines


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = db.init.Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    temp_dir.cleanup()


def insert_article_with_signal(
    article_id: str,
    topic: str,
    title: str,
    summary: str,
    concern: float,
    purchase: float,
    avoidance: float,
    frame: str,
    seg_young_urban: float,
    seg_family: float,
    seg_senior: float,
    seg_b2b: float,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, 'unit-test', ?, ?, ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, title, summary, f"https://example.test/{article_id}", topic),
    )
    raw_json = {
        "concern_level": concern,
        "purchase_intent": purchase,
        "avoidance_signals": avoidance,
        "dominant_frame": frame,
        "seg_young_urban": seg_young_urban,
        "seg_family": seg_family,
        "seg_senior": seg_senior,
        "seg_b2b": seg_b2b,
    }
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-03-17T00:00:00+00:00')
        """,
        (
            article_id,
            concern,
            purchase,
            avoidance,
            frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            json.dumps(raw_json),
        ),
    )
    conn.commit()


def test_extract_json_strips_code_fences() -> None:
    text = """Reasoning first

```json
{
  "headline": "Shift detected",
  "nested": {"value": 1}
}
```
"""
    parsed = _extract_json(text)
    assert parsed == {"headline": "Shift detected", "nested": {"value": 1}}


def test_generate_brief_returns_valid_structured_brief() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="senior-1",
            topic="inflace",
            title="Inflation fears hit pensioners",
            summary="Older households are cutting spending as prices keep rising.",
            concern=0.92,
            purchase=0.08,
            avoidance=0.61,
            frame="fear",
            seg_young_urban=0.05,
            seg_family=0.10,
            seg_senior=0.85,
            seg_b2b=0.00,
        )
        insert_article_with_signal(
            article_id="family-1",
            topic="inflace",
            title="Families delay big purchases",
            summary="Parents are postponing appliance upgrades and travel plans.",
            concern=0.74,
            purchase=0.18,
            avoidance=0.49,
            frame="fear",
            seg_young_urban=0.10,
            seg_family=0.75,
            seg_senior=0.15,
            seg_b2b=0.00,
        )

        responses = iter(
            [
                "Seniors show the strongest behavioral shift because fear-framed inflation coverage is concentrated there.",
                """```json
{
  "topic": "inflace",
  "headline": "Seniors show rising concern as inflation coverage intensifies",
  "narrative": "Inflation coverage is skewing more fear-based for seniors, with concern and avoidance both climbing. Family households also show caution, but the strongest reaction is still concentrated among older audiences.",
  "most_affected_segment": "senior",
  "drift_type": "concern_spike",
  "alert_level": "strong",
  "hypotheses": [
    {
      "segment": "senior",
      "hypothesis": "Seniors exposed to fear-framed inflation coverage will report lower willingness to make discretionary purchases over the next month.",
      "signal_basis": "concern_level +0.18",
      "suggested_question": "I expect to postpone non-essential purchases because of inflation news."
    },
    {
      "segment": "family",
      "hypothesis": "Family households will report increased budgeting behavior after repeated inflation stories tied to household essentials.",
      "signal_basis": "avoidance_signals +0.11",
      "suggested_question": "Inflation coverage has made me more cautious about household spending."
    },
    {
      "segment": "young_urban",
      "hypothesis": "Young urban adults will maintain attention to inflation news but show a smaller reduction in purchase intent than seniors.",
      "signal_basis": "purchase_intent -0.05",
      "suggested_question": "Inflation news has changed my willingness to make planned purchases."
    }
  ],
  "generated_at": "ignored",
  "model_used": "ignored"
}
```""",
            ]
        )

        with patch("brief.generator._call_ollama", side_effect=lambda prompt: next(responses)):
            brief = generate_brief("inflace")

        assert isinstance(brief, ResearchBrief)
        assert brief.topic == "inflace"
        assert brief.most_affected_segment == "senior"
        assert brief.drift_type == "concern_spike"
        assert brief.alert_level == "strong"
        assert brief.model_used == OLLAMA_MODEL
        assert len(brief.hypotheses) == 3
        assert brief.hypotheses[0].segment == "senior"
    finally:
        cleanup_temp_db(temp_dir)

def test_brief_route_returns_generator_result() -> None:
    pytest.importorskip("fastapi")
    from api.routes import brief as brief_route

    expected = ResearchBrief(
        topic="inflace",
        headline="Seniors show rising concern in inflation coverage",
        narrative="Inflation stories are amplifying concern among seniors, while other audiences remain less affected.",
        most_affected_segment="senior",
        drift_type="concern_spike",
        alert_level="mild",
        hypotheses=[
            {
                "segment": "senior",
                "hypothesis": "Seniors will reduce discretionary spending after repeated inflation coverage.",
                "signal_basis": "concern_level +0.12",
                "suggested_question": "Inflation news makes me less willing to spend on non-essentials.",
            },
            {
                "segment": "family",
                "hypothesis": "Families will report more cautious budgeting after inflation news.",
                "signal_basis": "avoidance_signals +0.07",
                "suggested_question": "Inflation coverage makes me more careful with household spending.",
            },
            {
                "segment": "young_urban",
                "hypothesis": "Young urban adults will show weaker concern shifts than seniors.",
                "signal_basis": "purchase_intent -0.03",
                "suggested_question": "Inflation coverage changes how likely I am to make planned purchases.",
            },
        ],
        generated_at="2026-03-17T00:00:00+00:00",
        model_used=OLLAMA_MODEL,
    )

    with patch("api.routes.brief.generate_brief", return_value=expected):
        result = brief_route.get_brief("inflace")

    assert result == expected


def test_main_registers_brief_route() -> None:
    pytest.importorskip("fastapi")
    import main

    paths = {route.path for route in main.app.routes}
    assert "/brief/{topic}" in paths
