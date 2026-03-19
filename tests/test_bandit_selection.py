#!/usr/bin/env python3

from unittest.mock import patch

from ingestion.bandit import select_feeds


def _feed(outlet: str) -> dict[str, object]:
    return {
        "outlet": outlet,
        "rss_url": f"https://example.test/{outlet}.rss",
        "affinity_tag": "mainstream",
        "country": "GLOBAL",
        "language": "en",
        "enabled": True,
    }


def test_select_feeds_returns_all_when_pool_is_at_or_below_limit() -> None:
    feeds = [_feed("a"), _feed("b"), _feed("c"), _feed("d"), _feed("e"), _feed("f")]

    with patch("ingestion.bandit.score_feed") as mock_score:
        selected = select_feeds("war", k=6, feeds=feeds)

    assert [feed["outlet"] for feed in selected] == ["a", "b", "c", "d", "e", "f"]
    mock_score.assert_not_called()


def test_select_feeds_caps_larger_pools_using_bandit_scores() -> None:
    feeds = [_feed(name) for name in ["a", "b", "c", "d", "e", "f", "g", "h"]]
    scored = [
        {"feed": feed, "score": 1.0 - index * 0.1, "pulls": 0, "total_reward": 0.0}
        for index, feed in enumerate(feeds)
    ]

    with patch("ingestion.bandit.score_feed", side_effect=scored) as mock_score:
        selected = select_feeds("war", k=6, feeds=feeds)

    assert len(selected) == 6
    assert [feed["outlet"] for feed in selected] == ["a", "b", "c", "d", "e", "f"]
    assert mock_score.call_count == len(feeds)
