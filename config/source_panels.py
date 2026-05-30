from __future__ import annotations

from collections.abc import Sequence

from config.topics import domain_for_topic


DEFAULT_PANEL_BY_COUNTRY: dict[str, list[str]] = {
    "CZ": ["irozhlas", "ct24", "idnes", "e15"],
    "DE": ["tagesschau", "spiegel"],
    "GLOBAL": ["bbc_world", "bbc", "nytimes", "the_guardian"],
}

DOMAIN_PANEL_EXTRAS: dict[str, list[str]] = {
    "commerce": ["irozhlas_eko", "e15"],
    "health": ["irozhlas", "ct24"],
    "civic": ["irozhlas", "ct24", "idnes"],
}


def fixed_panel_outlets(topic: str = "", country: str = "") -> list[str]:
    normalized_country = (country or "").strip().upper()
    domain = domain_for_topic(topic)

    if normalized_country:
        outlets = list(DEFAULT_PANEL_BY_COUNTRY.get(normalized_country, []))
    else:
        outlets = [
            outlet
            for country_outlets in DEFAULT_PANEL_BY_COUNTRY.values()
            for outlet in country_outlets
        ]

    outlets.extend(DOMAIN_PANEL_EXTRAS.get(domain, []))
    return list(dict.fromkeys(outlets))


def select_fixed_panel_feeds(
    feeds: Sequence[dict],
    *,
    topic: str = "",
    country: str = "",
) -> list[dict]:
    allowed = set(fixed_panel_outlets(topic=topic, country=country))
    if not allowed:
        return list(feeds)
    selected = [feed for feed in feeds if str(feed.get("outlet", "")) in allowed]
    return selected or list(feeds)
