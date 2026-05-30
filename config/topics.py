from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from config.domains import DEFAULT_DOMAIN, topic_to_domain


@dataclass(frozen=True)
class TopicSeed:
    canonical_topic_id: str
    display_name: str
    domain: str
    aliases: tuple[tuple[str, str | None], ...]


TOPIC_SEEDS: tuple[TopicSeed, ...] = (
    TopicSeed(
        canonical_topic_id="energy",
        display_name="Energy",
        domain="commerce",
        aliases=(
            ("energie", "cs"),
            ("energetika", "cs"),
            ("ceny energii", "cs"),
            ("ceny energií", "cs"),
            ("energy", "en"),
            ("energy prices", "en"),
            ("energiepreise", "de"),
        ),
    ),
    TopicSeed(
        canonical_topic_id="inflation",
        display_name="Inflation",
        domain="commerce",
        aliases=(
            ("inflace", "cs"),
            ("inflation", "en"),
            ("inflation rate", "en"),
            ("inflationsrate", "de"),
        ),
    ),
    TopicSeed(
        canonical_topic_id="healthcare",
        display_name="Healthcare",
        domain="health",
        aliases=(
            ("zdravotnictvi", "cs"),
            ("zdravotnictví", "cs"),
            ("zdravi", "cs"),
            ("zdraví", "cs"),
            ("healthcare", "en"),
            ("health", "en"),
            ("gesundheit", "de"),
        ),
    ),
    TopicSeed(
        canonical_topic_id="politics",
        display_name="Politics",
        domain="civic",
        aliases=(
            ("politika", "cs"),
            ("politics", "en"),
            ("policy", "en"),
            ("politik", "de"),
        ),
    ),
)


def normalize_topic(raw_topic: str) -> str:
    """Normalize topic aliases for lookup while staying dependency-free."""
    normalized = unicodedata.normalize("NFKD", raw_topic.strip().lower())
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = re.sub(r"[_\-]+", " ", ascii_text)
    ascii_text = re.sub(r"[^\w\s]", " ", ascii_text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", ascii_text).strip()


def slugify_topic(raw_topic: str) -> str:
    normalized = normalize_topic(raw_topic)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug or "topic"


def display_name_for_topic(raw_topic: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_topic.strip().replace("-", " "))
    if not normalized:
        return "Topic"
    return normalized[:1].upper() + normalized[1:]


def seed_for_alias(raw_topic: str) -> TopicSeed | None:
    normalized = normalize_topic(raw_topic)
    if not normalized:
        return None
    for seed in TOPIC_SEEDS:
        if seed.canonical_topic_id == normalized:
            return seed
        if any(normalize_topic(alias) == normalized for alias, _ in seed.aliases):
            return seed
    return None


def domain_for_topic(raw_topic: str) -> str:
    seed = seed_for_alias(raw_topic)
    if seed is not None:
        return seed.domain
    return topic_to_domain(raw_topic) or DEFAULT_DOMAIN
