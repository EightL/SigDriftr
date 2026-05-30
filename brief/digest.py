import json
from datetime import datetime, timezone

from api.models import DigestResponse
from brief.generator import OLLAMA_MODEL, _call_ollama_json
from db.topic_queries import get_digest_articles
from db.topic_resolver import resolve_topic


_DIGEST_PROMPT_TEMPLATE = """IMPORTANT: Your entire response MUST be in English.

You are preparing a concise, citation-first media digest for a news analyst.

Topic: {topic}
Country filter: {country}
Source filter: {source}

Use only the articles below. Do not invent facts. Prefer synthesis over repetition.
Return ONLY valid JSON matching this schema:
{{
  "summary_headline": "<6-12 words>",
  "summary_text": "<2-4 sentences grounded in the articles>",
  "key_points": [
    "<bullet 1>",
    "<bullet 2>",
    "<bullet 3>"
  ]
}}

Articles:
{articles_block}
""".strip()


def _article_block(articles: list[dict]) -> str:
    lines: list[str] = []
    for index, article in enumerate(articles, 1):
        summary = (article.get("summary") or "")[:260]
        body = (article.get("body") or "")[:420]
        lines.append(
            f"{index}. [{article['outlet']} | {article['country']}] "
            f"{article['title']} | relevance={article['relevance_score']:.2f} | "
            f"url={article['url']}"
        )
        if summary:
            lines.append(f"   RSS summary: {summary}")
        if body:
            lines.append(f"   Body excerpt: {body}")
    return "\n".join(lines)


def _fallback_digest(
    topic: str,
    country: str,
    source: str,
    articles: list[dict],
) -> DigestResponse:
    generated_at = datetime.now(timezone.utc).isoformat()
    resolution = resolve_topic(topic)
    if not articles:
        return DigestResponse(
            topic=topic,
            requested_topic=topic,
            canonical_topic_id=resolution.canonical_topic_id,
            canonical_display_name=resolution.display_name,
            country=country or "all",
            source=source or "all",
            article_count=0,
            generated_at=generated_at,
            sources_used=[],
            summary_headline=f"No recent matches for {topic}",
            summary_text=(
                "No stored articles matched the requested topic and source filters. "
                "Run collection first or widen the filters."
            ),
            key_points=[
                "No matching articles are currently stored.",
                "Try a broader country or source filter.",
                "Run the collection pipeline again for fresher coverage.",
            ],
            articles=[],
        )

    unique_sources = list(dict.fromkeys(article["outlet"] for article in articles))
    top_titles = [article["title"] for article in articles[:3]]
    return DigestResponse(
        topic=topic,
        requested_topic=topic,
        canonical_topic_id=resolution.canonical_topic_id,
        canonical_display_name=resolution.display_name,
        country=country or "all",
        source=source or "all",
        article_count=len(articles),
        generated_at=generated_at,
        sources_used=unique_sources,
        summary_headline=f"{topic.title()} coverage across {len(unique_sources)} sources",
        summary_text=(
            f"The current digest is based on {len(articles)} stored articles from "
            f"{', '.join(unique_sources[:4])}. The most prominent items are: "
            + "; ".join(top_titles)
            + "."
        ),
        key_points=[
            f"{articles[0]['outlet']} carries the highest-ranked matching article.",
            f"{len(unique_sources)} distinct sources contributed to this digest.",
            "The response fell back to a deterministic summary because the LLM output was unavailable.",
        ],
        articles=[
            {
                "article_id": article["article_id"],
                "title": article["title"],
                "url": article["url"],
                "outlet": article["outlet"],
                "country": article["country"],
                "published_at": article["published_at"],
                "relevance_score": article["relevance_score"],
            }
            for article in articles
        ],
    )


def generate_digest(
    topic: str,
    country: str = "",
    source: str = "",
    limit: int = 8,
) -> DigestResponse:
    resolution = resolve_topic(topic)
    articles = get_digest_articles(topic, country=country, source=source, limit=limit)
    fallback = _fallback_digest(topic, country, source, articles)
    if not articles:
        return fallback

    prompt = _DIGEST_PROMPT_TEMPLATE.format(
        topic=topic,
        country=country or "all",
        source=source or "all",
        articles_block=_article_block(articles),
    )
    try:
        payload = _call_ollama_json(prompt)
        summary_headline = str(payload.get("summary_headline", "")).strip()
        summary_text = str(payload.get("summary_text", "")).strip()
        key_points = [
            str(item).strip()
            for item in payload.get("key_points", [])
            if str(item).strip()
        ][:5]
        if not summary_headline or not summary_text or len(key_points) < 3:
            return fallback
        return DigestResponse(
            topic=topic,
            requested_topic=topic,
            canonical_topic_id=resolution.canonical_topic_id,
            canonical_display_name=resolution.display_name,
            country=country or "all",
            source=source or "all",
            article_count=len(articles),
            generated_at=datetime.now(timezone.utc).isoformat(),
            sources_used=list(dict.fromkeys(article["outlet"] for article in articles)),
            summary_headline=summary_headline,
            summary_text=summary_text,
            key_points=key_points,
            articles=[
                {
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "url": article["url"],
                    "outlet": article["outlet"],
                    "country": article["country"],
                    "published_at": article["published_at"],
                    "relevance_score": article["relevance_score"],
                }
                for article in articles
            ],
        )
    except Exception:
        return fallback


def generate_digest_json(
    topic: str,
    country: str = "",
    source: str = "",
    limit: int = 8,
) -> dict:
    result = generate_digest(topic, country=country, source=source, limit=limit)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return json.loads(result.json())
