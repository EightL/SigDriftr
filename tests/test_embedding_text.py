#!/usr/bin/env python3

import pytest

from extraction.embedding_text import build_embedding_text


def test_build_embedding_text_uses_first_two_body_sentences() -> None:
    result = build_embedding_text(
        title="Inflation climbs",
        body="First useful body sentence is definitely long enough. Second useful sentence also has enough detail. Third sentence should be ignored.",
        summary="Short summary.",
    )

    assert result == (
        "Inflation climbs - First useful body sentence is definitely long enough. "
        "Second useful sentence also has enough detail."
    )


def test_build_embedding_text_skips_boilerplate_sentences() -> None:
    result = build_embedding_text(
        title="Europe markets",
        body="Read more in our app. Cookie preferences can be managed online. This sentence is the first useful one with enough length. Another valid sentence follows with details.",
        summary="",
    )

    assert result == (
        "Europe markets - This sentence is the first useful one with enough length. "
        "Another valid sentence follows with details."
    )


def test_build_embedding_text_falls_back_to_summary() -> None:
    result = build_embedding_text(
        title="German exports",
        body="",
        summary="A long enough summary sentence appears first. Another long enough summary sentence follows immediately.",
    )

    assert result == (
        "German exports - A long enough summary sentence appears first. "
        "Another long enough summary sentence follows immediately."
    )


def test_build_embedding_text_returns_title_when_no_usable_lead() -> None:
    result = build_embedding_text(
        title="Meaningful Title",
        body="Read more. Click here.",
        summary="Cookie settings.",
    )

    assert result == "Meaningful Title"


def test_build_embedding_text_rejects_too_short_final_text() -> None:
    with pytest.raises(ValueError, match="embedding text too short"):
        build_embedding_text(title="Short", body="", summary="")
