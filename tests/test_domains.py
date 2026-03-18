#!/usr/bin/env python3
import pytest

from config.domains import topic_to_domain


pytest.importorskip("tenacity")
from extraction.llm_client import _apply_domain_mask


def test_topic_to_domain_maps_keywords_case_insensitively() -> None:
    assert topic_to_domain("inflace") == "commerce"
    assert topic_to_domain("politika") == "civic"
    assert topic_to_domain("zdravi") == "health"
    assert topic_to_domain("pocasi") == "generic"
    assert topic_to_domain("") == "generic"
    assert topic_to_domain("INFLACE") == "commerce"


def test_apply_domain_mask_zeroes_irrelevant_fields() -> None:
    result = _apply_domain_mask(
        {
            "concern_level": 0.5,
            "purchase_intent": 0.8,
            "avoidance_signals": 0.3,
        },
        "civic",
    )

    assert result["concern_level"] == 0.5
    assert result["purchase_intent"] == 0.0
    assert result["avoidance_signals"] == 0.3
    assert "purchase_intent" in result["irrelevant_fields"]
    assert result["domain"] == "civic"
