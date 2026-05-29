#!/usr/bin/env python3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db.init
import pytest

from config.feeds import FEEDS

pytest.importorskip("pydantic")

from brief.generator import clear_brief_cache, generate_brief
from delta.engine import compute_drift
from delta.mapper import compute_segment_profiles
from extraction.extractor import run_extraction
from ingestion.bandit import get_bandit_snapshot
from ingestion.crawler import crawl


ORIGINAL_DB_PATH = db.init.DB_PATH


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    clear_brief_cache()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    clear_brief_cache()
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()


def test_end_to_end_pipeline_produces_brief_and_bandit_updates() -> None:
    temp_dir = setup_temp_db()
    try:
        feed = {
            "outlet": "irozhlas",
            "rss_url": "https://example.test/rss",
            "affinity_tag": "mainstream",
        }
        rss_payload = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>SigDriftr Test Feed</title>
    <item>
      <title>Inflace tlaci domacnosti do uspor</title>
      <description>Inflace meni nakupni plany domacnosti.</description>
      <link>https://example.test/articles/1</link>
      <pubDate>Tue, 18 Mar 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Inflace zdrazuje zakladni zbozi</title>
      <description>Spotrebitele meni sve chovani kvuli inflaci.</description>
      <link>https://example.test/articles/2</link>
      <pubDate>Tue, 18 Mar 2026 11:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Firmy reaguji na inflace v dodavatelskych retezcich</title>
      <description>Inflace ovlivnuje planovani firem.</description>
      <link>https://example.test/articles/3</link>
      <pubDate>Tue, 18 Mar 2026 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
""".encode("utf-8")
        parsed_feed = SimpleNamespace(
            entries=[
                {
                    "title": "Inflace tlaci domacnosti do uspor",
                    "summary": "Inflace meni nakupni plany domacnosti.",
                    "link": "https://example.test/articles/1",
                    "published": "2026-03-18T10:00:00+00:00",
                },
                {
                    "title": "Inflace zdrazuje zakladni zbozi",
                    "summary": "Spotrebitele meni sve chovani kvuli inflaci.",
                    "link": "https://example.test/articles/2",
                    "published": "2026-03-18T11:00:00+00:00",
                },
                {
                    "title": "Firmy reaguji na inflace v dodavatelskych retezcich",
                    "summary": "Inflace ovlivnuje planovani firem.",
                    "link": "https://example.test/articles/3",
                    "published": "2026-03-18T12:00:00+00:00",
                },
            ]
        )
        extracted_signals = {
            "concern_level": 0.7,
            "purchase_intent": 0.1,
            "avoidance_signals": 0.4,
            "dominant_frame": "fear",
            "seg_young_urban": 0.3,
            "seg_family": 0.3,
            "seg_senior": 0.2,
            "seg_b2b": 0.2,
            "domain": "commerce",
            "irrelevant_fields": [],
        }
        brief_payload = {
            "headline": "Inflation coverage drives caution across key segments",
            "narrative": "Families and young urban audiences show rising caution after repeated inflation coverage.",
            "most_affected_segment": "family",
            "drift_type": "concern_spike",
            "alert_level": "mild",
            "hypotheses": [
                {
                    "segment": "family",
                    "hypothesis": "Families will postpone household purchases after repeated inflation coverage.",
                    "signal_basis": "concern_level +0.10",
                    "suggested_question": "Inflation news makes me more cautious about household spending.",
                },
                {
                    "segment": "young_urban",
                    "hypothesis": "Young urban adults will reduce optional spending.",
                    "signal_basis": "avoidance_signals +0.08",
                    "suggested_question": "Inflation coverage changes how likely I am to make discretionary purchases.",
                },
                {
                    "segment": "b2b",
                    "hypothesis": "Business buyers will delay non-essential commitments.",
                    "signal_basis": "purchase_intent -0.04",
                    "suggested_question": "Recent inflation coverage makes me more likely to delay a planned purchase.",
                },
            ],
        }

        with patch(
            "ingestion.crawler.select_feeds",
            return_value=[feed],
        ), patch(
            "ingestion.crawler._fetch_feed_bytes",
            return_value=rss_payload,
        ), patch(
            "ingestion.crawler._fetch_article_body",
            side_effect=lambda url, fallback_summary: (fallback_summary, url),
        ), patch(
            "ingestion.crawler.feedparser.parse",
            return_value=parsed_feed,
        ), patch(
            "extraction.extractor.extract_signals",
            return_value=extracted_signals,
        ), patch(
            "extraction.extractor.extract_entities",
            return_value=[],
        ), patch(
            "brief.generator._call_ollama_json",
            side_effect=[
                {
                    "facts": ["Senior coverage is showing the strongest movement."],
                    "numeric_changes": ["senior concern_level +0.12"],
                    "cited_clusters": ["track-1"],
                    "cited_articles": ["article-1"],
                    "evidence_gaps": [],
                },
                {
                    "what_changed": "Senior concern is rising fastest in the current window.",
                    "for_whom": "Seniors are the clearest leading segment.",
                    "uncertainty_and_caveats": [],
                },
                brief_payload,
            ],
        ):
            inserted = crawl("inflace")
            processed = run_extraction("inflace")
            compute_segment_profiles("inflace", learn_baseline=True)
            drift = compute_drift("inflace")
            brief = generate_brief("inflace")

        snapshot = get_bandit_snapshot("irozhlas")
    finally:
        cleanup_temp_db(temp_dir)

    assert any(configured_feed["outlet"] == "irozhlas" for configured_feed in FEEDS)
    assert inserted >= 1
    assert processed >= 1
    assert len(drift) == 4
    for entry in drift:
        assert "confidence" in entry
        assert "baseline_is_learned" in entry
        assert "baseline_sample_count" in entry
        assert "domain" in entry
        assert "relevant_fields" in entry

    assert brief.topic == "inflace"
    assert brief.confidence_context is not None
    assert set(brief.confidence_context.segment_confidence.keys()) == {
        "young_urban",
        "family",
        "senior",
        "b2b",
    }
    assert brief.alert_level in ("none", "mild", "strong")
    assert len(brief.hypotheses) == 3
    assert snapshot["pulls"] >= 1
