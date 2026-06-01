from __future__ import annotations

from dataclasses import dataclass

from clustering.dedupe import ArticleDedupeContext, dedupe_embeddings


@dataclass(frozen=True)
class Row:
    article_id: str
    vector: list[float]


def test_dedupe_embeddings_collapses_exact_title_and_semantic_groups() -> None:
    rows = [
        Row("a", [1.0, 0.0, 0.0]),
        Row("b", [1.0, 0.0, 0.0]),
        Row("c", [0.0, 1.0, 0.0]),
        Row("d", [0.0, 0.96, 0.04]),
        Row("e", [0.0, 0.0, 1.0]),
    ]
    contexts = {
        "a": ArticleDedupeContext(
            article_id="a",
            title="Inflation pressure rises again",
            body="This is the longer canonical article body.",
        ),
        "b": ArticleDedupeContext(
            article_id="b",
            title="Inflation pressure rises again",
            body="short",
        ),
        "c": ArticleDedupeContext(
            article_id="c",
            title="Energy prices strain household budgets",
            summary="Energy prices strain household budgets in winter.",
            body="This is the longer energy duplicate source article body.",
        ),
        "d": ArticleDedupeContext(
            article_id="d",
            title="Energy prices strain household budgets again",
            summary="Energy prices strain household budgets in winter again.",
        ),
        "e": ArticleDedupeContext(article_id="e", title="Different story"),
    }

    result = dedupe_embeddings(rows, contexts)

    assert [row.article_id for row in result.rows] == ["a", "c", "e"]
    assert result.stats == {
        "raw_article_count": 5,
        "cluster_input_count": 3,
        "duplicate_group_count": 2,
        "duplicates_removed": 2,
    }
    assert [group.member_article_ids for group in result.groups] == [
        ["a", "b"],
        ["c", "d"],
    ]


def test_dedupe_embeddings_does_not_merge_low_overlap_semantic_neighbors() -> None:
    rows = [
        Row("a", [1.0, 0.0]),
        Row("b", [0.93, 0.367559]),
    ]
    contexts = {
        "a": ArticleDedupeContext(
            article_id="a",
            title="Central bank inflation rate decision",
            summary="Central bank board members discussed inflation.",
        ),
        "b": ArticleDedupeContext(
            article_id="b",
            title="Hospital waiting rooms remain crowded",
            summary="Patients reported long waits at regional hospitals.",
        ),
    }

    result = dedupe_embeddings(
        rows,
        contexts,
        cosine_exact_threshold=0.97,
        cosine_lexical_threshold=0.93,
        min_lexical_overlap=0.60,
    )

    assert [row.article_id for row in result.rows] == ["a", "b"]
    assert result.stats["duplicates_removed"] == 0
