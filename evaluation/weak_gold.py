from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = ROOT / "eval" / "annotations" / "czech_batch_200_semi_gold.csv"
DEFAULT_ARTICLES_PATH = ROOT / "eval" / "annotations" / "czenec_batch_200.csv"
DEFAULT_REPORT_PATH = ROOT / "eval" / "reports" / "semi_gold_eval_report.json"
DEFAULT_EXTRACTOR_OUTPUT_PATH = (
    ROOT / "eval" / "model_outputs" / "current_extractor_predictions.json"
)

EVAL_FIELDS = [
    "topic_relevance",
    "dominant_frame",
    "concern_bucket",
    "purchase_relevance",
    "avoidance_relevance",
    "seg_young_urban_relevance",
    "seg_family_relevance",
    "seg_senior_relevance",
    "seg_b2b_relevance",
]
GROUP_FIELDS = {
    "topic": ["topic_relevance"],
    "frame_concern": ["dominant_frame", "concern_bucket"],
    "action": ["purchase_relevance", "avoidance_relevance"],
    "segments": [
        "seg_young_urban_relevance",
        "seg_family_relevance",
        "seg_senior_relevance",
        "seg_b2b_relevance",
    ],
}
FIELD_LABELS = {
    "topic_relevance": ["0", "1", "2"],
    "dominant_frame": ["fear", "opportunity", "conflict", "neutral", "other"],
    "concern_bucket": ["low", "medium", "high"],
    "purchase_relevance": ["0", "1", "2"],
    "avoidance_relevance": ["0", "1", "2"],
    "seg_young_urban_relevance": ["0", "1", "2"],
    "seg_family_relevance": ["0", "1", "2"],
    "seg_senior_relevance": ["0", "1", "2"],
    "seg_b2b_relevance": ["0", "1", "2"],
}
DIRECT_LABEL_FIELDS = set(EVAL_FIELDS)
SIGNAL_TO_LABEL_FIELD = {
    "concern_level": "concern_bucket",
    "purchase_intent": "purchase_relevance",
    "avoidance_signals": "avoidance_relevance",
    "seg_young_urban": "seg_young_urban_relevance",
    "seg_family": "seg_family_relevance",
    "seg_senior": "seg_senior_relevance",
    "seg_b2b": "seg_b2b_relevance",
}
FRAME_ALIASES = {
    "alert": "fear",
    "anxiety": "fear",
    "concern": "fear",
    "risk": "fear",
    "threat": "fear",
    "benefit": "opportunity",
    "growth": "opportunity",
    "solution": "opportunity",
    "controversy": "conflict",
    "debate": "conflict",
    "dispute": "conflict",
    "explanatory": "neutral",
    "informational": "neutral",
    "mixed": "neutral",
    "unclear": "neutral",
    "unknown": "neutral",
}
NUMERIC_LABEL_FIELDS = {
    "topic_relevance",
    "purchase_relevance",
    "avoidance_relevance",
    "seg_young_urban_relevance",
    "seg_family_relevance",
    "seg_senior_relevance",
    "seg_b2b_relevance",
}


class EvaluationError(RuntimeError):
    pass


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_label(field: str, value: object) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    if field == "dominant_frame":
        text = text.replace("-", "_").replace(" ", "_")
        text = FRAME_ALIASES.get(text, text)
        return text if text in FIELD_LABELS[field] else "other"
    if field == "concern_bucket":
        aliases = {"medium_concern": "medium", "med": "medium"}
        text = aliases.get(text, text)
        return text if text in FIELD_LABELS[field] else ""
    if field in NUMERIC_LABEL_FIELDS:
        try:
            score = float(text)
        except ValueError:
            return ""
        if score in {0.0, 1.0, 2.0}:
            return str(int(score))
        if 0.0 <= score <= 1.0:
            if field == "topic_relevance":
                return _topic_relevance_from_score(score)
            if field in {"purchase_relevance", "avoidance_relevance"}:
                return _bucket_0_1(score, low_cutoff=0.25, high_cutoff=0.67)
            return _bucket_0_1(score, low_cutoff=0.20, high_cutoff=0.45)
        return ""
    return text


def _topic_relevance_from_score(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    if score < 0.55:
        return "0"
    if score < 0.80:
        return "1"
    return "2"


def _bucket_0_1(value: object, *, low_cutoff: float, high_cutoff: float) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    if score < low_cutoff:
        return "0"
    if score < high_cutoff:
        return "1"
    return "2"


def _concern_bucket(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""
    if score < 0.34:
        return "low"
    if score < 0.67:
        return "medium"
    return "high"


def _label_from_signal(signal_field: str, value: object) -> str:
    if signal_field == "concern_level":
        return _concern_bucket(value)
    if signal_field in {"purchase_intent", "avoidance_signals"}:
        return _bucket_0_1(value, low_cutoff=0.25, high_cutoff=0.67)
    if signal_field.startswith("seg_"):
        return _bucket_0_1(value, low_cutoff=0.20, high_cutoff=0.45)
    return ""


def load_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_markdown_table_records(path: Path) -> list[dict[str, str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    if len(lines) < 3:
        raise EvaluationError(f"Could not find a Markdown table in {path}.")

    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    records = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            raise EvaluationError(
                f"Markdown table row has {len(cells)} cells, expected {len(header)}."
            )
        records.append(dict(zip(header, cells)))
    return records


def load_annotation_records(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_records(path)
    if suffix in {".md", ".markdown"}:
        return load_markdown_table_records(path)
    raise EvaluationError(f"Unsupported annotation file type: {path.suffix}")


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("labels", "predictions", "signals", "records", "items"):
            records = payload.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            records = []
            for article_id, record in payload.items():
                merged = {"article_id": article_id}
                merged.update(record)
                records.append(merged)
            return records
    raise EvaluationError(f"Could not find prediction records in {path}.")


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_records(path)
    if suffix == ".csv":
        return load_csv_records(path)
    if suffix in {".md", ".markdown"}:
        return load_markdown_table_records(path)
    raise EvaluationError(f"Unsupported prediction file type: {path.suffix}")


def labels_from_record(record: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    raw_json = record.get("raw_json")
    if isinstance(raw_json, dict):
        merged = dict(raw_json)
        for key, value in record.items():
            if value not in (None, ""):
                merged[key] = value
        record = merged
    nested = record.get("labels")
    if isinstance(nested, dict):
        record = {**record, **nested}

    for field in DIRECT_LABEL_FIELDS:
        label = _normalize_label(field, record.get(field))
        if label:
            labels[field] = label

    if "dominant_frame" in record and "dominant_frame" not in labels:
        label = _normalize_label("dominant_frame", record.get("dominant_frame"))
        if label:
            labels["dominant_frame"] = label

    for signal_field, label_field in SIGNAL_TO_LABEL_FIELD.items():
        if label_field in labels:
            continue
        label = _label_from_signal(signal_field, record.get(signal_field))
        if label:
            labels[label_field] = label

    if "topic_relevance" not in labels:
        for score_field in ("topic_relevance_score", "relevance_score"):
            label = _topic_relevance_from_score(record.get(score_field))
            if label:
                labels["topic_relevance"] = label
                break

    return labels


def load_baseline(path: Path) -> list[dict[str, Any]]:
    records = load_annotation_records(path)
    baseline = []
    for record in records:
        labels = {
            field: _normalize_label(field, record.get(field))
            for field in EVAL_FIELDS
        }
        labels = {field: label for field, label in labels.items() if label}
        baseline.append(
            {
                "article_id": _clean_text(record.get("article_id")),
                "canonical_topic_id": _clean_text(record.get("canonical_topic_id")),
                "source": _clean_text(record.get("source")),
                "labels": labels,
            }
        )
    return baseline


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for record in load_prediction_records(path):
        article_id = _clean_text(record.get("article_id"))
        if not article_id:
            continue
        predictions[article_id] = labels_from_record(record)
    return predictions


def _distribution(counter: Counter[str], labels: list[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: round(counter[label] / total, 4) for label in labels}


def _total_variation(expected: Counter[str], predicted: Counter[str], labels: list[str]) -> float:
    expected_dist = _distribution(expected, labels)
    predicted_dist = _distribution(predicted, labels)
    distance = 0.5 * sum(
        abs(expected_dist[label] - predicted_dist[label]) for label in labels
    )
    return round(distance, 4)


def _macro_f1(rows: list[tuple[str, str]], labels: list[str]) -> float | None:
    if not rows:
        return None
    f1_values = []
    for label in labels:
        tp = sum(1 for expected, predicted in rows if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in rows if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in rows if expected == label and predicted != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    if not f1_values:
        return None
    return round(sum(f1_values) / len(f1_values), 4)


def _metric_summary(rows: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    support = len(rows)
    correct = sum(1 for expected, predicted in rows if expected == predicted)
    expected = Counter(expected for expected, _ in rows)
    predicted = Counter(predicted for _, predicted in rows)
    return {
        "support": support,
        "correct": correct,
        "accuracy": round(correct / support, 4) if support else None,
        "macro_f1": _macro_f1(rows, labels),
        "expected_distribution": _distribution(expected, labels),
        "predicted_distribution": _distribution(predicted, labels),
        "total_variation_distance": _total_variation(expected, predicted, labels),
    }


def evaluate(
    baseline_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, str]],
    *,
    baseline_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    field_rows: dict[str, list[tuple[str, str]]] = {field: [] for field in EVAL_FIELDS}
    by_topic_rows: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: {field: [] for field in EVAL_FIELDS}
    )
    missing_predictions_by_field = Counter()
    unlabeled_by_field = Counter()
    missing_articles = []

    baseline_ids = {row["article_id"] for row in baseline_rows if row["article_id"]}

    for row in baseline_rows:
        article_id = row["article_id"]
        topic = row["canonical_topic_id"] or "unknown"
        predicted_labels = predictions.get(article_id)
        if predicted_labels is None:
            missing_articles.append(article_id)
            predicted_labels = {}
        for field in EVAL_FIELDS:
            expected = row["labels"].get(field, "")
            if not expected:
                unlabeled_by_field[field] += 1
                continue
            predicted = predicted_labels.get(field, "")
            if not predicted:
                missing_predictions_by_field[field] += 1
                continue
            field_rows[field].append((expected, predicted))
            by_topic_rows[topic][field].append((expected, predicted))

    fields = {}
    for field in EVAL_FIELDS:
        fields[field] = {
            **_metric_summary(field_rows[field], FIELD_LABELS[field]),
            "unlabeled_baseline_count": unlabeled_by_field[field],
            "missing_prediction_count": missing_predictions_by_field[field],
        }

    group_rows: dict[str, list[tuple[str, str]]] = {}
    for group, group_fields in GROUP_FIELDS.items():
        rows: list[tuple[str, str]] = []
        for field in group_fields:
            rows.extend(field_rows[field])
        labels = sorted({label for field in group_fields for label in FIELD_LABELS[field]})
        group_rows[group] = rows
        fields_for_group = {field: len(field_rows[field]) for field in group_fields}
        fields_for_group["support"] = len(rows)

    groups = {
        group: _metric_summary(
            rows,
            sorted({label for field in GROUP_FIELDS[group] for label in FIELD_LABELS[field]}),
        )
        for group, rows in group_rows.items()
    }

    all_rows = [row for rows in field_rows.values() for row in rows]
    overall_labels = sorted({label for labels in FIELD_LABELS.values() for label in labels})

    by_topic = {}
    for topic, topic_fields in sorted(by_topic_rows.items()):
        topic_all_rows = [row for rows in topic_fields.values() for row in rows]
        by_topic[topic] = {
            "overall": _metric_summary(topic_all_rows, overall_labels),
            "fields": {
                field: _metric_summary(rows, FIELD_LABELS[field])
                for field, rows in topic_fields.items()
            },
        }

    unexpected_articles = sorted(set(predictions) - baseline_ids)
    warnings = []
    if fields["topic_relevance"]["missing_prediction_count"]:
        warnings.append(
            "topic_relevance is evaluated as an ingestion/filtering label. "
            "Provide topic_relevance, topic_relevance_score, or relevance_score "
            "in prediction records to score it."
        )
    if any(
        fields[field]["support"] == 0
        for field in [
            "seg_young_urban_relevance",
            "seg_family_relevance",
            "seg_senior_relevance",
            "seg_b2b_relevance",
        ]
    ):
        warnings.append(
            "Segment scores are relative shares in production but ordinal relevance "
            "labels in the weak-gold set; threshold mapping should be calibrated."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(baseline_path) if baseline_path else None,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "article_count": len(baseline_rows),
        "prediction_article_count": len(predictions),
        "missing_article_count": len(missing_articles),
        "missing_article_sample": missing_articles[:20],
        "unexpected_article_count": len(unexpected_articles),
        "unexpected_article_sample": unexpected_articles[:20],
        "overall": _metric_summary(all_rows, overall_labels),
        "groups": groups,
        "fields": fields,
        "by_topic": by_topic,
        "mapping": {
            "concern_level": "low < 0.34, medium < 0.67, high otherwise",
            "purchase_intent": "0 < 0.25, 1 < 0.67, 2 otherwise",
            "avoidance_signals": "0 < 0.25, 1 < 0.67, 2 otherwise",
            "segments": "0 < 0.20, 1 < 0.45, 2 otherwise",
            "dominant_frame": "normalized enum with unknown values mapped to other",
        },
        "warnings": warnings,
    }


def run_extractor_predictions(
    articles_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    from config.feeds import FEEDS
    from extraction.llm_client import extract_signals

    outlet_affinity = {feed["outlet"]: feed["affinity_tag"] for feed in FEEDS}
    rows = load_csv_records(articles_path)
    if limit is not None:
        rows = rows[:limit]

    labels = []
    provider_counts = Counter()
    model_counts = Counter()
    fallback_count = 0
    for row in rows:
        article_id = _clean_text(row.get("article_id"))
        outlet = _clean_text(row.get("source"))
        topic = _clean_text(row.get("canonical_topic_id"))
        raw = extract_signals(
            _clean_text(row.get("title")),
            _clean_text(row.get("summary")),
            affinity_tag=outlet_affinity.get(outlet, "mainstream"),
            topic=topic,
            body=_clean_text(row.get("body_excerpt")),
        )
        provider = _clean_text(raw.get("extractor_provider"))
        model = _clean_text(raw.get("extractor_model"))
        provider_counts[provider] += 1
        model_counts[model] += 1
        if model == "gemma3:1b":
            fallback_count += 1
        labels.append(
            {
                "article_id": article_id,
                **labels_from_record(raw),
                "raw_signals": raw,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "articles_path": str(articles_path),
        "label_count": len(labels),
        "provider_counts": dict(provider_counts),
        "model_counts": dict(model_counts),
        "fallback_count": fallback_count,
        "labels": labels,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate extractor/model labels against the CZE-NEC semi-gold baseline."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Semi-gold annotation CSV.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Prediction JSON/CSV. JSON may contain labels, predictions, signals, records, or items.",
    )
    parser.add_argument(
        "--run-extractor",
        action="store_true",
        help="Run the configured live extractor over --articles before evaluating.",
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=DEFAULT_ARTICLES_PATH,
        help="Article CSV used when --run-extractor is set.",
    )
    parser.add_argument(
        "--extractor-output",
        type=Path,
        default=DEFAULT_EXTRACTOR_OUTPUT_PATH,
        help="Where to write live extractor predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit live extractor runs; useful for smoke tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the evaluation report JSON.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Fail if overall accuracy is below this threshold.",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        help="Fail if overall macro F1 is below this threshold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.run_extractor:
        run_extractor_predictions(
            args.articles,
            args.extractor_output,
            limit=args.limit,
        )
        predictions_path = args.extractor_output
    elif args.predictions:
        predictions_path = args.predictions
    else:
        raise EvaluationError("Provide --predictions or use --run-extractor.")

    baseline_rows = load_baseline(args.baseline)
    predictions = load_predictions(predictions_path)
    report = evaluate(
        baseline_rows,
        predictions,
        baseline_path=args.baseline,
        predictions_path=predictions_path,
    )
    write_report(report, args.output)
    if (
        args.min_accuracy is not None
        and report["overall"]["accuracy"] is not None
        and report["overall"]["accuracy"] < args.min_accuracy
    ):
        raise EvaluationError(
            f"Overall accuracy {report['overall']['accuracy']} is below "
            f"--min-accuracy {args.min_accuracy}."
        )
    if (
        args.min_macro_f1 is not None
        and report["overall"]["macro_f1"] is not None
        and report["overall"]["macro_f1"] < args.min_macro_f1
    ):
        raise EvaluationError(
            f"Overall macro F1 {report['overall']['macro_f1']} is below "
            f"--min-macro-f1 {args.min_macro_f1}."
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall_accuracy": report["overall"]["accuracy"],
                "overall_macro_f1": report["overall"]["macro_f1"],
                "support": report["overall"]["support"],
                "warnings": report["warnings"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
