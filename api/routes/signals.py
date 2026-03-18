import json

from fastapi import APIRouter

from api.models import SignalRecord
from config.domains import get_domain_config
from db.init import get_conn
from delta.engine import compute_drift
from delta.mapper import compute_segment_profiles
from extraction.extractor import run_extraction


router = APIRouter()


@router.post("/extract")
def extract(topic: str = "") -> dict[str, int | str]:
    processed = run_extraction(topic)
    compute_segment_profiles(topic, learn_baseline=True)
    return {"processed": processed, "topic": topic}


@router.get("/signals", response_model=list[SignalRecord])
def get_signals(topic: str = "") -> list[dict]:
    conn = get_conn()
    drift = compute_drift(topic)
    segment_confidence = {
        entry["segment"]: {
            "confidence": entry["confidence"],
            "baseline_is_learned": entry["baseline_is_learned"],
            "baseline_sample_count": entry["baseline_sample_count"],
            "baseline_age_days": entry["baseline_age_days"],
        }
        for entry in drift
    }
    rows = conn.execute(
        """
        SELECT a.topic, s.article_id, s.concern_level, s.purchase_intent,
               s.avoidance_signals, s.dominant_frame, s.seg_young_urban,
               s.seg_family, s.seg_senior, s.seg_b2b, s.raw_json, s.extracted_at
        FROM signals s
        JOIN articles a ON a.id = s.article_id
        WHERE (a.topic = ? OR ? = '')
        ORDER BY s.extracted_at DESC, s.article_id DESC
        """,
        (topic, topic),
    ).fetchall()

    records: list[dict] = []
    for row in rows:
        raw_json = json.loads(row[10]) if row[10] else {}
        domain = raw_json.get("domain", "generic")
        records.append(
            {
                "article_id": row[1],
                "topic": row[0],
                "domain": domain,
                "relevant_fields": list(get_domain_config(domain)["relevant_fields"]),
                "concern_level": row[2],
                "purchase_intent": row[3],
                "avoidance_signals": row[4],
                "dominant_frame": row[5],
                "seg_young_urban": row[6],
                "seg_family": row[7],
                "seg_senior": row[8],
                "seg_b2b": row[9],
                "raw_json": raw_json,
                "extracted_at": row[11],
                "segment_confidence": segment_confidence,
            }
        )

    return records
