from __future__ import annotations

import re
from typing import Optional


_MIN_SENTENCE_LENGTH = 20
_MIN_FINAL_TEXT_LENGTH = 10
_BOILERPLATE_PATTERNS = (
    "read more",
    "cookie",
    "subscribe",
    "click here",
    "sign up",
    "newsletter",
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\t", " ").replace("\n", " ")).strip()


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    try:
        from syntok import segmenter
    except Exception:
        raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
        return [_normalize_whitespace(sentence) for sentence in raw_sentences if sentence.strip()]

    sentences: list[str] = []
    for paragraph in segmenter.process(normalized):
        for sentence in paragraph:
            sentence_text = "".join(token.spacing + token.value for token in sentence).strip()
            if sentence_text:
                sentences.append(_normalize_whitespace(sentence_text))
    return sentences


def _is_trivial_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < _MIN_SENTENCE_LENGTH:
        return True
    return any(pattern in lowered for pattern in _BOILERPLATE_PATTERNS)


def build_embedding_text_candidate(
    *,
    title: str,
    body: Optional[str],
    summary: Optional[str],
) -> str:
    clean_title = _normalize_whitespace(title or "")
    lead = _normalize_whitespace(body or "") or _normalize_whitespace(summary or "")

    sentences = [
        sentence
        for sentence in _split_sentences(lead)
        if not _is_trivial_sentence(sentence)
    ]
    lead_excerpt = " ".join(sentences[:2]).strip()

    if clean_title and lead_excerpt:
        return _normalize_whitespace(f"{clean_title} - {lead_excerpt}")
    if clean_title:
        return clean_title
    return lead_excerpt


def build_embedding_text(
    *,
    title: str,
    body: Optional[str],
    summary: Optional[str],
) -> str:
    final_text = build_embedding_text_candidate(title=title, body=body, summary=summary)
    if len(final_text) < _MIN_FINAL_TEXT_LENGTH:
        raise ValueError("embedding text too short")
    return final_text
