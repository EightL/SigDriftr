#!/usr/bin/env python3

from config.feeds import FEED_CATALOG_PATH, FEEDS, get_enabled_feeds


def test_feed_catalog_file_is_the_source_of_configured_feeds() -> None:
    assert FEED_CATALOG_PATH.exists()
    assert any(str(feed["outlet"]) == "irozhlas" for feed in FEEDS)
    assert all("categories" in feed for feed in FEEDS)


def test_feed_registry_exposes_german_and_global_catalog_feeds() -> None:
    outlets = {str(feed["outlet"]) for feed in FEEDS}

    assert "faz" in outlets
    assert "sueddeutsche" in outlets
    assert "welt" in outlets
    assert "nytimes" in outlets
    assert "the_guardian" in outlets
    assert "cnn" in outlets


def test_country_filter_exposes_added_global_and_german_feeds_without_duplicates() -> None:
    global_feeds = get_enabled_feeds(country="GLOBAL")
    global_outlets = {str(feed["outlet"]) for feed in global_feeds}
    global_urls = [str(feed["rss_url"]) for feed in global_feeds]

    german_feeds = get_enabled_feeds(country="DE")
    german_outlets = {str(feed["outlet"]) for feed in german_feeds}

    assert {"bbc_world", "reuters_world", "nytimes", "the_guardian", "cnn"} <= global_outlets
    assert "faz" in german_outlets
    assert "zeit_index" in german_outlets
    assert len(global_urls) == len(set(global_urls))


def test_source_filter_works_with_catalog_loaded_feeds() -> None:
    feeds = get_enabled_feeds(source="irozhlas_eko")

    assert len(feeds) == 1
    assert feeds[0]["country"] == "CZ"
    assert feeds[0]["language"] == "cs"
    assert feeds[0]["categories"] == ["economy", "business"]
