import hashlib
from datetime import datetime, timezone

import feedparser

from config.feeds import FEEDS
from db.init import get_conn


def crawl(topic: str) -> int:
    conn = get_conn()
    inserted = 0
    topic_lower = topic.lower()

    for feed in FEEDS:
        parsed = feedparser.parse(feed["rss_url"])

        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            combined_text = f"{title} {summary}".lower()
            if topic_lower not in combined_text:
                continue

            url = entry.get("link", "")
            if not url:
                continue

            article_id = hashlib.sha256(url.encode()).hexdigest()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    article_id,
                    feed["outlet"],
                    title,
                    summary,
                    url,
                    topic,
                    entry.get("published", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted += cursor.rowcount

        conn.commit()

    return inserted
