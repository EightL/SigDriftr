#!/usr/bin/env python3

from config.feeds import FEEDS, get_enabled_feeds


def test_feed_registry_merges_extra_german_and_global_feeds() -> None:
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
