from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Literal

from brief.models import ResearchBrief
from config.settings import MIN_BRIEF_CONFIDENCE


IssueSeverity = Literal["info", "warning", "error"]

_ALERT_LEVEL_RANK = {
    "no_data": -1,
    "none": 0,
    "mild": 1,
    "strong": 2,
}
_SEGMENTS = {"young_urban", "family", "senior", "b2b"}
_DELTA_PATTERN = re.compile(
    r"\b(?P<signal>concern_level|purchase_intent|avoidance_signals|concern|purchase|avoidance)"
    r"\b[^+-]{0,32}(?P<value>[+-]\s*(?:\d+(?:\.\d+)?|\.\d+))",
    re.IGNORECASE,
)
_SIGNAL_ALIASES = {
    "concern": "concern_level",
    "concern_level": "concern_level",
    "purchase": "purchase_intent",
    "purchase_intent": "purchase_intent",
    "avoidance": "avoidance_signals",
    "avoidance_signals": "avoidance_signals",
}


@dataclass(frozen=True)
class BriefValidationIssue:
    code: str
    message: str
    severity: IssueSeverity = "warning"
    field: str | None = None
    references: list[str] = dc_field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.field:
            payload["field"] = self.field
        if self.references:
            payload["references"] = list(self.references)
        return payload


def validation_issues_as_dicts(
    issues: list[BriefValidationIssue],
) -> list[dict[str, object]]:
    return [issue.as_dict() for issue in issues]


def _bundle_items(bundle: Any, name: str) -> list[dict[str, object]]:
    value = getattr(bundle, name, [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _segment_lookup(bundle: Any) -> dict[str, dict[str, object]]:
    return {
        str(item.get("segment")): item
        for item in _bundle_items(bundle, "segment_rollups")
        if str(item.get("segment")) in _SEGMENTS
    }


def _rank_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        segments,
        key=lambda item: (
            -_ALERT_LEVEL_RANK.get(str(item.get("alert_level", "none")), 0),
            -float(item.get("drift_magnitude", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
            -int(item.get("article_count", 0) or 0),
            str(item.get("segment", "")),
        ),
    )


def _selected_track_ids(bundle: Any) -> set[str]:
    ids: set[str] = set()
    for item in _bundle_items(bundle, "selected_observations"):
        for key in ("track_id", "observation_id"):
            value = str(item.get(key, "")).strip()
            if value:
                ids.add(value)
    return ids


def _selected_article_ids(bundle: Any) -> set[str]:
    ids: set[str] = set()
    for item in _bundle_items(bundle, "selected_observations"):
        raw_ids = item.get("exemplar_article_ids", [])
        if isinstance(raw_ids, list):
            ids.update(str(value) for value in raw_ids if str(value).strip())
        raw_articles = item.get("exemplar_articles", [])
        if isinstance(raw_articles, list):
            for article in raw_articles:
                if isinstance(article, dict):
                    article_id = str(article.get("article_id", "")).strip()
                    if article_id:
                        ids.add(article_id)
    return ids


def _normalize_delta_value(raw: str) -> float | None:
    try:
        return float(raw.replace(" ", ""))
    except ValueError:
        return None


def _check_citations(
    *,
    bundle: Any,
    cited_track_ids: list[str],
    cited_article_ids: list[str],
) -> list[BriefValidationIssue]:
    issues: list[BriefValidationIssue] = []
    selected_tracks = _selected_track_ids(bundle)
    selected_articles = _selected_article_ids(bundle)

    for track_id in cited_track_ids:
        if selected_tracks and track_id not in selected_tracks:
            issues.append(
                BriefValidationIssue(
                    code="cited_track_not_selected",
                    field="support.cited_track_ids",
                    references=[track_id],
                    message=(
                        f"Cited track '{track_id}' is not part of the selected "
                        "evidence observations for this brief."
                    ),
                )
            )

    for article_id in cited_article_ids:
        if selected_articles and article_id not in selected_articles:
            issues.append(
                BriefValidationIssue(
                    code="cited_article_not_selected",
                    field="support.cited_article_ids",
                    references=[article_id],
                    message=(
                        f"Cited article '{article_id}' is not present in the "
                        "selected exemplar articles for this brief."
                    ),
                )
            )

    return issues


def _check_status_consistency(
    brief: ResearchBrief,
    segments: dict[str, dict[str, object]],
) -> list[BriefValidationIssue]:
    issues: list[BriefValidationIssue] = []
    total_articles = sum(int(item.get("article_count", 0) or 0) for item in segments.values())
    ready_segments = [
        item
        for item in segments.values()
        if (
            bool(item.get("baseline_is_learned", False))
            and float(item.get("confidence", 0.0) or 0.0) >= MIN_BRIEF_CONFIDENCE
            and bool(item.get("has_data", int(item.get("article_count", 0) or 0) > 0))
        )
    ]

    if brief.status == "insufficient_data" and total_articles > 0:
        issues.append(
            BriefValidationIssue(
                code="insufficient_data_with_articles",
                field="status",
                message=(
                    "Brief status is insufficient_data even though the evidence "
                    "snapshot contains recent segment articles."
                ),
            )
        )
    if brief.status == "ready" and not ready_segments:
        issues.append(
            BriefValidationIssue(
                code="ready_without_ready_segment",
                field="status",
                message=(
                    "Brief status is ready, but no segment has both a learned "
                    "baseline and enough signal readiness."
                ),
            )
        )
    return issues


def _check_top_segment(
    brief: ResearchBrief,
    segments: dict[str, dict[str, object]],
) -> list[BriefValidationIssue]:
    if not segments or brief.status == "insufficient_data":
        return []
    top = _rank_segments(list(segments.values()))[0]
    top_segment = str(top.get("segment", ""))
    issues: list[BriefValidationIssue] = []
    if top_segment and brief.most_affected_segment != top_segment:
        issues.append(
            BriefValidationIssue(
                code="most_affected_segment_not_top_ranked",
                field="most_affected_segment",
                references=[brief.most_affected_segment, top_segment],
                message=(
                    f"Brief names '{brief.most_affected_segment}' as most affected, "
                    f"but the evidence ranking points to '{top_segment}'."
                ),
            )
        )

    top_alert = str(top.get("alert_level", "none"))
    if top_alert in {"none", "mild", "strong"} and brief.alert_level != top_alert:
        issues.append(
            BriefValidationIssue(
                code="alert_level_differs_from_top_segment",
                field="alert_level",
                references=[brief.alert_level, top_alert],
                message=(
                    f"Brief alert level is '{brief.alert_level}', while the top "
                    f"evidence segment is '{top_alert}'."
                ),
            )
        )
    return issues


def _check_hypotheses(
    brief: ResearchBrief,
    segments: dict[str, dict[str, object]],
    *,
    delta_tolerance: float,
) -> list[BriefValidationIssue]:
    issues: list[BriefValidationIssue] = []
    for index, hypothesis in enumerate(brief.hypotheses):
        segment_key = hypothesis.segment
        segment = segments.get(segment_key)
        if segment is None and segments:
            issues.append(
                BriefValidationIssue(
                    code="hypothesis_segment_missing_from_evidence",
                    field=f"hypotheses[{index}].segment",
                    references=[segment_key],
                    message=(
                        f"Hypothesis segment '{segment_key}' is not present in "
                        "the evidence snapshot."
                    ),
                )
            )
            continue

        deltas = segment.get("deltas", {}) if isinstance(segment, dict) else {}
        if not isinstance(deltas, dict):
            continue
        for match in _DELTA_PATTERN.finditer(hypothesis.signal_basis):
            signal = _SIGNAL_ALIASES[match.group("signal").lower()]
            claimed = _normalize_delta_value(match.group("value"))
            if claimed is None:
                continue
            actual = float(deltas.get(signal, 0.0) or 0.0)
            if abs(claimed - actual) > delta_tolerance:
                issues.append(
                    BriefValidationIssue(
                        code="hypothesis_delta_mismatch",
                        field=f"hypotheses[{index}].signal_basis",
                        references=[segment_key, signal],
                        message=(
                            f"Hypothesis claims {signal} {claimed:+.2f} for "
                            f"{segment_key}, but evidence delta is {actual:+.2f}."
                        ),
                    )
                )
    return issues


def validate_brief(
    brief: ResearchBrief,
    *,
    bundle: Any,
    cited_track_ids: list[str] | None = None,
    cited_article_ids: list[str] | None = None,
    delta_tolerance: float = 0.05,
) -> list[BriefValidationIssue]:
    segments = _segment_lookup(bundle)
    issues: list[BriefValidationIssue] = []
    issues.extend(
        _check_citations(
            bundle=bundle,
            cited_track_ids=list(cited_track_ids or []),
            cited_article_ids=list(cited_article_ids or []),
        )
    )
    issues.extend(_check_status_consistency(brief, segments))
    issues.extend(_check_top_segment(brief, segments))
    issues.extend(_check_hypotheses(brief, segments, delta_tolerance=delta_tolerance))
    return issues
