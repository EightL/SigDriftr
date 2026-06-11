from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency guard.
    yaml = None

from config.german_feeds import GERMAN_FEEDS
from config.global_feeds import GLOBAL_FEEDS


FEED_CATALOG_PATH = Path(__file__).with_name("feed_catalog.yaml")

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


def normalize_country(country: str) -> str:
    return country.strip().upper()


def _normalize_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def _coerce_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_categories(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = raw_value.split(",")
    elif isinstance(raw_value, list):
        values = raw_value
    else:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _coerce_feed(
    raw_feed: dict,
    *,
    country: str = "",
    language: str = "",
) -> dict[str, object]:
    return {
        "outlet": str(raw_feed.get("outlet", "")).strip(),
        "rss_url": str(raw_feed.get("rss_url", "")).strip(),
        "affinity_tag": str(raw_feed.get("affinity_tag", "mainstream")).strip()
        or "mainstream",
        "country": normalize_country(str(raw_feed.get("country", country))),
        "language": str(raw_feed.get("language", language)).strip().lower(),
        "enabled": _coerce_bool(raw_feed.get("enabled", True)),
        "categories": _coerce_categories(raw_feed.get("categories")),
    }


def _load_feed_catalog(path: Path = FEED_CATALOG_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load config/feed_catalog.yaml. "
            "Install dependencies from requirements.txt."
        )

    with path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}

    if not isinstance(catalog, dict):
        raise ValueError("Feed catalog must be a YAML mapping with a 'feeds' list.")

    raw_feeds = catalog.get("feeds", [])
    if not isinstance(raw_feeds, list):
        raise ValueError("Feed catalog field 'feeds' must be a list.")

    return [
        _coerce_feed(raw_feed)
        for raw_feed in raw_feeds
        if isinstance(raw_feed, dict)
    ]


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


def _legacy_feeds() -> list[dict[str, object]]:
    return [*_BASE_FEEDS, *_extra_feeds()]


def _load_configured_feeds() -> list[dict[str, object]]:
    catalog_feeds = _load_feed_catalog()
    return catalog_feeds or _legacy_feeds()


FEEDS = _dedupe_feeds(_load_configured_feeds())
VALID_COUNTRIES = {"CZ", "DE", "GLOBAL", "SK", "PL"} | {
    normalize_country(str(feed.get("country", "")))
    for feed in FEEDS
    if str(feed.get("country", "")).strip()
}


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
