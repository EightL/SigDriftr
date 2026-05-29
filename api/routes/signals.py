import json

from fastapi import APIRouter, BackgroundTasks

from api.models import SignalRecord
from brief.generator import clear_brief_cache
from config.domains import get_domain_config
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from delta.engine import compute_drift
from delta.mapper import compute_segment_profiles
from extraction.extractor import run_extraction


router = APIRouter()


@router.post("/extract")
def extract(topic: str = "") -> dict[str, int | str]:
    processed = run_extraction(topic)
    compute_segment_profiles(topic, learn_baseline=True)
    if processed > 0:
        clear_brief_cache()
    return {"processed": processed, "topic": topic}


def query_signals(
    topic: str = "",
    *,
    country: str = "",
    source: str = "",
    language: str | None = None,
) -> list[dict]:
    conn = get_conn()
    drift = compute_drift(
        topic,
        country=country,
        source=source,
        language=language,
    )
    segment_confidence = {
        entry["segment"]: {
            "confidence": entry["confidence"],
            "baseline_is_learned": entry["baseline_is_learned"],
            "baseline_sample_count": entry["baseline_sample_count"],
            "baseline_age_days": entry["baseline_age_days"],
        }
        for entry in drift
    }
    topic_sql, topic_params = topic_filter_sql("a", topic)
    query = f"""
        SELECT COALESCE(
                   (SELECT at.topic
                    FROM article_topics at
                    WHERE at.article_id = a.id
                    ORDER BY at.matched_at DESC
                    LIMIT 1),
                   a.topic,
                   ''
               ) AS effective_topic,
               s.article_id, a.title, a.outlet, a.country, a.language, a.url,
               s.concern_level, s.purchase_intent,
               s.avoidance_signals, s.dominant_frame, s.seg_young_urban,
               s.seg_family, s.seg_senior, s.seg_b2b,
               s.seg_young_urban_relevance, s.seg_family_relevance,
               s.seg_senior_relevance, s.seg_b2b_relevance,
               s.raw_json, s.extracted_at
        FROM signals s
        JOIN articles a ON a.id = s.article_id
        WHERE 1 = 1
          {topic_sql}
    """
    params: list[object] = [*topic_params]
    if country:
        query += " AND a.country = ?"
        params.append(country.strip().upper())
    if source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(source.strip().lower())
    if language is not None:
        query += " AND LOWER(a.language) = ?"
        params.append(language.strip().lower())
    query += """
        ORDER BY s.extracted_at DESC, s.article_id DESC
    """
    rows = conn.execute(query, params).fetchall()

    records: list[dict] = []
    for row in rows:
        raw_json = json.loads(row[19]) if row[19] else {}
        domain = raw_json.get("domain", "generic")
        records.append(
            {
                "article_id": row[1],
                "topic": row[0],
                "title": row[2] or "[no title]",
                "outlet": row[3] or "",
                "country": row[4] or "",
                "language": row[5],
                "url": row[6] or "",
                "domain": domain,
                "relevant_fields": list(get_domain_config(domain)["relevant_fields"]),
                "concern_level": row[7],
                "purchase_intent": row[8],
                "avoidance_signals": row[9],
                "dominant_frame": row[10],
                "seg_young_urban": row[11],
                "seg_family": row[12],
                "seg_senior": row[13],
                "seg_b2b": row[14],
                "seg_young_urban_relevance": row[15],
                "seg_family_relevance": row[16],
                "seg_senior_relevance": row[17],
                "seg_b2b_relevance": row[18],
                "topic_relevance_score": raw_json.get("topic_relevance_score"),
                "topic_relevance": raw_json.get("topic_relevance"),
                "raw_json": raw_json,
                "extracted_at": row[20],
                "segment_confidence": segment_confidence,
            }
        )
    return records


@router.get("/signals", response_model=list[SignalRecord])
def get_signals(
    topic: str = "",
    country: str = "",
    source: str = "",
    language: str | None = None,
    background_tasks: BackgroundTasks = None,
) -> list[dict]:
    records = query_signals(
        topic,
        country=country,
        source=source,
        language=language,
    )

    if topic and background_tasks is not None:
        background_tasks.add_task(
            compute_segment_profiles,
            topic,
            7,
            True,
        )

    return records
