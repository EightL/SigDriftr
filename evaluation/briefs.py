from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brief.models import ResearchBrief


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDENS_PATH = ROOT / "eval" / "briefs" / "golden_briefs.jsonl"
DEFAULT_REPORT_PATH = ROOT / "eval" / "reports" / "brief_eval_report.json"
EXACT_FIELDS = ["status", "most_affected_segment", "drift_type", "alert_level"]


class BriefEvaluationError(RuntimeError):
    pass


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BriefEvaluationError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BriefEvaluationError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise BriefEvaluationError(
                    f"JSONL record at {path}:{line_number} must be an object."
                )
            records.append(payload)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("items") or payload.get("goldens")
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            merged = []
            for record_id, record in payload.items():
                item = {"id": record_id}
                item.update(record)
                merged.append(item)
            return merged
    raise BriefEvaluationError(f"Could not find records in {path}.")


def load_golden_records(path: Path) -> list[dict[str, Any]]:
    records = _load_json_or_jsonl(path)
    for index, record in enumerate(records, start=1):
        if not str(record.get("id", "")).strip():
            raise BriefEvaluationError(f"Golden record #{index} is missing id.")
    return records


def load_prediction_records(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for record in _load_json_or_jsonl(path):
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            continue
        predictions[record_id] = record
    return predictions


def _parse_brief(raw: object) -> tuple[ResearchBrief | None, str | None]:
    if raw is None:
        return None, "missing"
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None, "invalid_json"
    if not isinstance(raw, dict):
        return None, "invalid_type"
    try:
        if hasattr(ResearchBrief, "model_validate"):
            return ResearchBrief.model_validate(raw), None
        return ResearchBrief.parse_obj(raw), None
    except Exception:
        return None, "schema_validation"


def _prediction_for_record(
    record: dict[str, Any],
    predictions: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if predictions is not None:
        return predictions.get(str(record["id"]))
    embedded = record.get("prediction")
    return embedded if isinstance(embedded, dict) else None


def _distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: round(value / total, 4) for key, value in sorted(counter.items())}


def _field_summary(rows: list[tuple[str, str]]) -> dict[str, Any]:
    support = len(rows)
    correct = sum(1 for expected, predicted in rows if expected == predicted)
    return {
        "support": support,
        "correct": correct,
        "accuracy": round(correct / support, 4) if support else None,
        "expected_distribution": _distribution(Counter(expected for expected, _ in rows)),
        "predicted_distribution": _distribution(Counter(predicted for _, predicted in rows)),
    }


def _support_expectations(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("gold", {})
    if not isinstance(gold, dict):
        return {}
    support = gold.get("support_expectations", {})
    return support if isinstance(support, dict) else {}


def _gold_fields(record: dict[str, Any]) -> dict[str, str]:
    gold = record.get("gold", {})
    if not isinstance(gold, dict):
        return {}
    fields = gold.get("brief", gold)
    if not isinstance(fields, dict):
        return {}
    return {
        field: str(fields[field])
        for field in EXACT_FIELDS
        if fields.get(field) not in (None, "")
    }


def _hypothesis_segments(brief: ResearchBrief) -> set[str]:
    return {hypothesis.segment for hypothesis in brief.hypotheses}


def _support_ids(prediction: dict[str, Any], key: str) -> set[str]:
    support = prediction.get("support", prediction.get("brief_support", {}))
    if not isinstance(support, dict):
        return set()
    raw = support.get(key, [])
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw}


def _validation_issue_count(prediction: dict[str, Any]) -> int:
    support = prediction.get("support", prediction.get("brief_support", {}))
    if not isinstance(support, dict):
        return 0
    raw = support.get("validation_issues", [])
    return len(raw) if isinstance(raw, list) else 0


def evaluate_records(
    golden_records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]] | None = None,
    *,
    goldens_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    field_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing_prediction_ids: list[str] = []
    invalid_counts = Counter()
    support_failures: list[dict[str, object]] = []
    hypothesis_failures: list[dict[str, object]] = []
    generation_modes = Counter()
    validation_issue_total = 0

    for record in golden_records:
        record_id = str(record["id"])
        prediction = _prediction_for_record(record, predictions)
        if prediction is None:
            missing_prediction_ids.append(record_id)
            continue

        brief, parse_error = _parse_brief(prediction.get("brief", prediction))
        if parse_error is not None or brief is None:
            invalid_counts[parse_error or "schema_validation"] += 1
            continue

        generation_modes[brief.generation_mode or "unknown"] += 1
        validation_issue_total += _validation_issue_count(prediction)

        for field, expected in _gold_fields(record).items():
            field_rows[field].append((expected, str(getattr(brief, field))))

        support = _support_expectations(record)
        cited_tracks = _support_ids(prediction, "cited_track_ids")
        cited_articles = _support_ids(prediction, "cited_article_ids")
        for track_id in support.get("required_track_ids", []) or []:
            if str(track_id) not in cited_tracks:
                support_failures.append(
                    {
                        "id": record_id,
                        "kind": "missing_required_track",
                        "value": str(track_id),
                    }
                )
        for track_id in support.get("forbidden_track_ids", []) or []:
            if str(track_id) in cited_tracks:
                support_failures.append(
                    {
                        "id": record_id,
                        "kind": "forbidden_track_cited",
                        "value": str(track_id),
                    }
                )
        for article_id in support.get("required_article_ids", []) or []:
            if str(article_id) not in cited_articles:
                support_failures.append(
                    {
                        "id": record_id,
                        "kind": "missing_required_article",
                        "value": str(article_id),
                    }
                )

        expected_segments = {
            str(item)
            for item in support.get("expected_hypothesis_segments", []) or []
        }
        missing_segments = sorted(expected_segments - _hypothesis_segments(brief))
        if missing_segments:
            hypothesis_failures.append(
                {
                    "id": record_id,
                    "missing_segments": missing_segments,
                }
            )

    field_summaries = {
        field: _field_summary(field_rows[field])
        for field in EXACT_FIELDS
    }
    all_rows = [row for rows in field_rows.values() for row in rows]
    valid_prediction_count = sum(generation_modes.values())
    total = len(golden_records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goldens_path": str(goldens_path) if goldens_path else None,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "record_count": total,
        "valid_prediction_count": valid_prediction_count,
        "missing_prediction_count": len(missing_prediction_ids),
        "missing_prediction_sample": missing_prediction_ids[:20],
        "invalid_prediction_count": sum(invalid_counts.values()),
        "invalid_prediction_breakdown": dict(invalid_counts),
        "fallback_rate": (
            round(generation_modes["fallback"] / valid_prediction_count, 4)
            if valid_prediction_count
            else None
        ),
        "generation_mode_distribution": _distribution(generation_modes),
        "overall": _field_summary(all_rows),
        "fields": field_summaries,
        "support_failure_count": len(support_failures),
        "support_failures": support_failures[:50],
        "hypothesis_failure_count": len(hypothesis_failures),
        "hypothesis_failures": hypothesis_failures[:50],
        "validation_issue_total": validation_issue_total,
    }


def _run_generator_prediction(record: dict[str, Any]) -> dict[str, Any]:
    from brief.generator import (
        BriefSourceResolution,
        _build_bundle,
        _generate_hierarchical_brief_artifacts,
        generate_brief_cached,
        get_brief_support,
    )

    input_payload = record.get("input", {})
    if not isinstance(input_payload, dict):
        input_payload = {}
    snapshot = record.get("cluster_snapshot")
    if isinstance(snapshot, dict):
        topic = str(input_payload.get("topic") or snapshot.get("topic") or "")
        resolution = BriefSourceResolution(
            display_topic=topic,
            real_topic=topic,
            canonical_topic_id=str(snapshot.get("canonical_topic_id") or topic),
            canonical_display_name=str(snapshot.get("canonical_display_name") or topic),
            country=str(snapshot.get("country") or input_payload.get("country") or ""),
            source=str(snapshot.get("source") or input_payload.get("source") or ""),
            language=snapshot.get("language") or input_payload.get("language"),
            source_mode="cluster_drift",
            generation_mode="hierarchical_cluster",
            run_id=str(snapshot.get("run_id") or record["id"]),
            cluster_snapshot=snapshot,
        )
        artifacts = _generate_hierarchical_brief_artifacts(_build_bundle(resolution))
        return {
            "id": record["id"],
            "brief": artifacts.brief.model_dump() if hasattr(artifacts.brief, "model_dump") else artifacts.brief.dict(),
            "support": artifacts.support,
        }

    topic = str(input_payload.get("topic", ""))
    if not topic:
        raise BriefEvaluationError(f"Record {record['id']} has no input.topic.")
    kwargs = {
        "country": str(input_payload.get("country", "")),
        "source": str(input_payload.get("source", "")),
        "language": input_payload.get("language"),
        "run_id": input_payload.get("run_id"),
    }
    brief = generate_brief_cached(topic, **kwargs)
    support = get_brief_support(topic, **kwargs)
    return {
        "id": record["id"],
        "brief": brief.model_dump() if hasattr(brief, "model_dump") else brief.dict(),
        "support": support,
    }


def run_generator_predictions(records: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    predictions = [_run_generator_prediction(record) for record in records]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_count": len(predictions),
        "records": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated research briefs against frozen golden expectations."
    )
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS_PATH)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--run-generator",
        action="store_true",
        help="Generate predictions from the current brief pipeline before scoring.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=ROOT / "eval" / "model_outputs" / "current_brief_predictions.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    golden_records = load_golden_records(args.goldens)
    predictions_path = args.predictions
    if args.run_generator:
        run_generator_predictions(golden_records, args.prediction_output)
        predictions_path = args.prediction_output
    predictions = load_prediction_records(predictions_path) if predictions_path else None
    report = evaluate_records(
        golden_records,
        predictions,
        goldens_path=args.goldens,
        predictions_path=predictions_path,
    )
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall_accuracy": report["overall"]["accuracy"],
                "valid_prediction_count": report["valid_prediction_count"],
                "fallback_rate": report["fallback_rate"],
                "support_failure_count": report["support_failure_count"],
                "validation_issue_total": report["validation_issue_total"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
