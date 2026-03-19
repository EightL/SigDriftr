from __future__ import annotations


FEEDS = [
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


def normalize_country(country: str) -> str:
    return country.strip().upper()


def get_enabled_feeds(country: str = "", source: str = "") -> list[dict]:
    normalized_country = normalize_country(country) if country else ""
    normalized_source = source.strip().lower()

    results: list[dict] = []
    for feed in FEEDS:
        if not feed.get("enabled", True):
            continue
        if normalized_country and normalize_country(str(feed.get("country", ""))) != normalized_country:
            continue
        if normalized_source and str(feed.get("outlet", "")).lower() != normalized_source:
            continue
        results.append(feed)
    return results
