import json

from fastapi import APIRouter

from db.init import get_conn
from extraction.extractor import run_extraction


router = APIRouter()


@router.get("/signals")
def get_signals(topic: str = "") -> list[dict]:
    run_extraction(topic)
    conn = get_conn()
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

    return [
        {
            "article_id": row[1],
            "topic": row[0],
            "concern_level": row[2],
            "purchase_intent": row[3],
            "avoidance_signals": row[4],
            "dominant_frame": row[5],
            "seg_young_urban": row[6],
            "seg_family": row[7],
            "seg_senior": row[8],
            "seg_b2b": row[9],
            "raw_json": json.loads(row[10]) if row[10] else {},
            "extracted_at": row[11],
        }
        for row in rows
    ]
