from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Mapping, Protocol, TypeVar


SEMANTIC_DUPLICATE_THRESHOLD = 0.97
SEMANTIC_LEXICAL_DUPLICATE_THRESHOLD = 0.93
MIN_SEMANTIC_OVERLAP = 0.20
MIN_LEXICAL_OVERLAP = 0.60


class EmbeddingRow(Protocol):
    article_id: str
    vector: list[float]


TEmbeddingRow = TypeVar("TEmbeddingRow", bound=EmbeddingRow)


@dataclass(frozen=True)
class ArticleDedupeContext:
    article_id: str
    title: str = ""
    summary: str = ""
    body: str = ""
    canonical_url: str = ""
    published_at: str | None = None


@dataclass(frozen=True)
class DuplicateGroup:
    representative_article_id: str
    member_article_ids: list[str]
    reasons: list[str]


@dataclass(frozen=True)
class DedupeResult(Generic[TEmbeddingRow]):
    rows: list[TEmbeddingRow]
    groups: list[DuplicateGroup]
    stats: dict[str, int]


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_TOKEN_STOPWORDS = {
    "article",
    "body",
    "headline",
    "news",
    "story",
    "summary",
    "title",
}


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            parent = self.find(parent)
            self.parent[index] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def _collapse_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _normalize_title(value: str) -> str:
    tokens = _TOKEN_RE.findall((value or "").casefold())
    return " ".join(tokens)


def _token_set(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(value.casefold())
        if len(token) > 1
        and token not in _TOKEN_STOPWORDS
        and not token.isdigit()
    }


def _lexical_overlap(left: ArticleDedupeContext, right: ArticleDedupeContext) -> float:
    left_text = " ".join([left.title, left.summary, left.body[:500]])
    right_text = " ".join([right.title, right.summary, right.body[:500]])
    left_tokens = _token_set(left_text)
    right_tokens = _token_set(right_text)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _published_rank(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _representative_index(
    indexes: list[int],
    contexts: Mapping[str, ArticleDedupeContext],
    rows: list[TEmbeddingRow],
) -> int:
    def rank(index: int) -> tuple[int, int, float, str]:
        row = rows[index]
        context = contexts.get(row.article_id, ArticleDedupeContext(row.article_id))
        body_len = len(_collapse_text(context.body))
        summary_len = len(_collapse_text(context.summary))
        return (
            body_len,
            summary_len,
            _published_rank(context.published_at),
            row.article_id,
        )

    return max(indexes, key=rank)


def _record_reason(
    pair_reasons: dict[tuple[int, int], set[str]],
    left: int,
    right: int,
    reason: str,
) -> None:
    key = (left, right) if left < right else (right, left)
    pair_reasons.setdefault(key, set()).add(reason)


def _group_reasons(
    indexes: list[int],
    pair_reasons: dict[tuple[int, int], set[str]],
) -> list[str]:
    reasons: set[str] = set()
    index_set = set(indexes)
    for (left, right), pair_reason_values in pair_reasons.items():
        if left in index_set and right in index_set:
            reasons.update(pair_reason_values)
    return sorted(reasons)


def dedupe_embeddings(
    rows: list[TEmbeddingRow],
    contexts: Mapping[str, ArticleDedupeContext],
    *,
    cosine_exact_threshold: float = SEMANTIC_DUPLICATE_THRESHOLD,
    cosine_lexical_threshold: float = SEMANTIC_LEXICAL_DUPLICATE_THRESHOLD,
    min_semantic_overlap: float = MIN_SEMANTIC_OVERLAP,
    min_lexical_overlap: float = MIN_LEXICAL_OVERLAP,
) -> DedupeResult:
    if not rows:
        return DedupeResult(
            rows=[],
            groups=[],
            stats={
                "raw_article_count": 0,
                "cluster_input_count": 0,
                "duplicate_group_count": 0,
                "duplicates_removed": 0,
            },
        )

    union_find = _UnionFind(len(rows))
    pair_reasons: dict[tuple[int, int], set[str]] = {}
    normalized_titles: dict[int, str] = {}
    canonical_urls: dict[int, str] = {}

    for index, row in enumerate(rows):
        context = contexts.get(row.article_id, ArticleDedupeContext(row.article_id))
        title = _normalize_title(context.title)
        if len(title) >= 12 and len(title.split()) >= 2:
            normalized_titles[index] = title
        canonical_url = (context.canonical_url or "").strip().casefold()
        if canonical_url:
            canonical_urls[index] = canonical_url

    for left_index in range(len(rows)):
        left = rows[left_index]
        left_context = contexts.get(
            left.article_id,
            ArticleDedupeContext(left.article_id),
        )
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            right_context = contexts.get(
                right.article_id,
                ArticleDedupeContext(right.article_id),
            )

            if (
                canonical_urls.get(left_index)
                and canonical_urls.get(left_index) == canonical_urls.get(right_index)
            ):
                union_find.union(left_index, right_index)
                _record_reason(pair_reasons, left_index, right_index, "canonical_url")
                continue

            if (
                normalized_titles.get(left_index)
                and normalized_titles.get(left_index) == normalized_titles.get(right_index)
            ):
                union_find.union(left_index, right_index)
                _record_reason(pair_reasons, left_index, right_index, "title")
                continue

            cosine = _cosine_similarity(left.vector, right.vector)
            overlap = _lexical_overlap(left_context, right_context)
            if cosine >= cosine_exact_threshold and overlap >= min_semantic_overlap:
                union_find.union(left_index, right_index)
                _record_reason(pair_reasons, left_index, right_index, "semantic")
                continue

            if cosine >= cosine_lexical_threshold and overlap >= min_lexical_overlap:
                union_find.union(left_index, right_index)
                _record_reason(
                    pair_reasons,
                    left_index,
                    right_index,
                    "semantic_lexical",
                )

    grouped_indexes: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped_indexes.setdefault(union_find.find(index), []).append(index)

    duplicate_groups: list[DuplicateGroup] = []
    representative_by_root: dict[int, int] = {}
    for root, indexes in grouped_indexes.items():
        if len(indexes) == 1:
            representative_by_root[root] = indexes[0]
            continue
        representative = _representative_index(indexes, contexts, rows)
        representative_by_root[root] = representative
        duplicate_groups.append(
            DuplicateGroup(
                representative_article_id=rows[representative].article_id,
                member_article_ids=sorted(rows[index].article_id for index in indexes),
                reasons=_group_reasons(indexes, pair_reasons),
            )
        )

    representative_indexes = set(representative_by_root.values())
    deduped_rows = [
        row
        for index, row in enumerate(rows)
        if index in representative_indexes
    ]
    duplicate_count = len(rows) - len(deduped_rows)

    return DedupeResult(
        rows=deduped_rows,
        groups=sorted(
            duplicate_groups,
            key=lambda group: (
                group.representative_article_id,
                group.member_article_ids,
            ),
        ),
        stats={
            "raw_article_count": len(rows),
            "cluster_input_count": len(deduped_rows),
            "duplicate_group_count": len(duplicate_groups),
            "duplicates_removed": duplicate_count,
        },
    )
