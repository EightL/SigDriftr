from __future__ import annotations

from urllib.parse import urlsplit

from config.german_feeds import GERMAN_FEEDS
from config.global_feeds import GLOBAL_FEEDS


_BASE_FEEDS = [
    {
        "outlet": "irozhlas",
        "rss_url": "https://www.irozhlas.cz/rss/irozhlas",
        "affinity_tag": "mainstream",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "irozhlas_eko",
        "rss_url": "https://www.irozhlas.cz/rss/irozhlas/section/ekonomika",
        "affinity_tag": "b2b",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "wave",
        "rss_url": "https://wave.rozhlas.cz/rss.xml",
        "affinity_tag": "young_urban",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "idnes",
        "rss_url": "https://servis.idnes.cz/rss.aspx",
        "affinity_tag": "mainstream",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "novinky",
        "rss_url": "https://www.novinky.cz/rss",
        "affinity_tag": "mainstream",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "e15",
        "rss_url": "https://www.e15.cz/rss",
        "affinity_tag": "b2b",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "blesk",
        "rss_url": "https://www.blesk.cz/rss",
        "affinity_tag": "senior",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "maminka",
        "rss_url": "https://www.maminka.cz/rss",
        "affinity_tag": "family",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "ct24",
        "rss_url": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy",
        "affinity_tag": "mainstream",
        "country": "CZ",
        "language": "cs",
        "enabled": True,
    },
    {
        "outlet": "tagesschau",
        "rss_url": "https://www.tagesschau.de/xml/rss2",
        "affinity_tag": "mainstream",
        "country": "DE",
        "language": "de",
        "enabled": True,
    },
    {
        "outlet": "spiegel",
        "rss_url": "https://www.spiegel.de/schlagzeilen/index.rss",
        "affinity_tag": "mainstream",
        "country": "DE",
        "language": "de",
        "enabled": True,
    },
    {
        "outlet": "bbc_world",
        "rss_url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "affinity_tag": "mainstream",
        "country": "GLOBAL",
        "language": "en",
        "enabled": True,
    },
    {
        "outlet": "reuters_world",
        "rss_url": "https://feeds.reuters.com/Reuters/worldNews",
        "affinity_tag": "mainstream",
        "country": "GLOBAL",
        "language": "en",
        "enabled": True,
    },
]


VALID_COUNTRIES = {"CZ", "DE", "GLOBAL"}


def _normalize_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def _coerce_feed(
    raw_feed: dict,
    *,
    country: str,
    language: str,
) -> dict[str, object]:
    return {
        "outlet": str(raw_feed["outlet"]).strip(),
        "rss_url": str(raw_feed["rss_url"]).strip(),
        "affinity_tag": str(raw_feed.get("affinity_tag", "mainstream")).strip()
        or "mainstream",
        "country": country,
        "language": language,
        "enabled": bool(raw_feed.get("enabled", True)),
    }


def _dedupe_feeds(feeds: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen_outlets: set[str] = set()
    seen_urls: set[str] = set()

    for feed in feeds:
        outlet = str(feed.get("outlet", "")).strip().lower()
        url = _normalize_url(str(feed.get("rss_url", "")))
        if not outlet or not url:
            continue
        if outlet in seen_outlets or url in seen_urls:
            continue
        seen_outlets.add(outlet)
        seen_urls.add(url)
        deduped.append(feed)
    return deduped


def _extra_feeds() -> list[dict[str, object]]:
    german = [
        _coerce_feed(feed, country="DE", language="de")
        for feed in GERMAN_FEEDS
    ]
    global_news = [
        _coerce_feed(feed, country="GLOBAL", language="en")
        for feed in GLOBAL_FEEDS
    ]
    return german + global_news


FEEDS = _dedupe_feeds([*_BASE_FEEDS, *_extra_feeds()])


def normalize_country(country: str) -> str:
    return country.strip().upper()


def get_enabled_feeds(country: str = "", source: str = "") -> list[dict]:
    normalized_country = normalize_country(country) if country else ""
    normalized_source = source.strip().lower()

    results: list[dict] = []
    for feed in FEEDS:
        if not feed.get("enabled", True):
            continue
        if (
            normalized_country
            and normalize_country(str(feed.get("country", ""))) != normalized_country
        ):
            continue
        if normalized_source and str(feed.get("outlet", "")).lower() != normalized_source:
            continue
        results.append(feed)
    return results
