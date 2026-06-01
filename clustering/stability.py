from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from clustering.clustering_service import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_UMAP_COMPONENTS,
    DEFAULT_UMAP_NEIGHBORS,
    MIN_ARTICLES_FOR_CLUSTERING,
    SelectedEmbedding,
    _load_dedupe_context,
    _normalize_country,
    _normalize_language,
    _normalize_source,
    _normalize_topic,
    _select_embeddings,
    _window_bounds,
    cluster_rows,
)
from clustering.dedupe import dedupe_embeddings
from db.topic_resolver import resolve_topic
from extraction.embedder import get_model_name, get_model_version


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "median": round(float(np.median(array)), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
    }


def _noise_rate(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.sum(labels == -1) / len(labels))


def _cluster_coherence(
    rows: list[SelectedEmbedding],
    labels: np.ndarray,
    centroids: dict[int, list[float]],
) -> dict[int, float]:
    if not centroids:
        return {}

    vectors = np.asarray([row.vector for row in rows], dtype=float)
    coherence: dict[int, float] = {}
    for label, centroid_values in centroids.items():
        member_vectors = vectors[labels == label]
        if len(member_vectors) == 0:
            coherence[label] = 0.0
            continue
        centroid = np.asarray(centroid_values, dtype=float)
        centroid_norm = np.linalg.norm(centroid)
        member_norms = np.linalg.norm(member_vectors, axis=1)
        denominator = member_norms * centroid_norm
        similarities = np.divide(
            member_vectors @ centroid,
            denominator,
            out=np.zeros(len(member_vectors), dtype=float),
            where=denominator > 0,
        )
        coherence[label] = round(float(np.clip(similarities, 0.0, 1.0).mean()), 4)
    return coherence


def _base_cluster_members(labels: np.ndarray) -> dict[int, set[int]]:
    members: dict[int, set[int]] = {}
    for index, label in enumerate(labels):
        int_label = int(label)
        if int_label == -1:
            continue
        members.setdefault(int_label, set()).add(index)
    return members


def _survival_rates(
    *,
    base_members: dict[int, set[int]],
    sample_indices: np.ndarray,
    sample_labels: np.ndarray,
) -> dict[int, float]:
    sample_label_by_index = {
        int(index): int(label)
        for index, label in zip(sample_indices.tolist(), sample_labels.tolist())
    }
    rates: dict[int, float] = {}
    for label, member_indexes in base_members.items():
        overlap = sorted(index for index in member_indexes if index in sample_label_by_index)
        if len(overlap) < 2:
            continue
        label_counts = Counter(sample_label_by_index[index] for index in overlap)
        rates[label] = max(label_counts.values()) / len(overlap)
    return rates


def _import_sklearn_metrics():
    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cluster stability evaluation requires scikit-learn metrics. "
            "Install scikit-learn or the full embedding dependencies."
        ) from exc
    return adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score


def _sample_size(n_articles: int, sample_fraction: float) -> int:
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the (0, 1] interval.")
    return min(
        n_articles,
        max(MIN_ARTICLES_FOR_CLUSTERING, int(round(n_articles * sample_fraction))),
    )


def evaluate_cluster_stability(
    *,
    topic: str,
    country: str | None = None,
    source: str | None = None,
    language: str | None = None,
    window_hours: int = 24,
    bootstrap_samples: int = 25,
    sample_fraction: float = 0.8,
    random_state: int = 42,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, object]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1.")

    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score = (
        _import_sklearn_metrics()
    )

    normalized_topic = _normalize_topic(topic)
    canonical_topic_id = resolve_topic(normalized_topic).canonical_topic_id
    normalized_country = _normalize_country(country)
    normalized_source = _normalize_source(source)
    normalized_language = _normalize_language(language)
    window_start, window_end = _window_bounds(window_hours)

    rows = _select_embeddings(
        topic=normalized_topic,
        country=normalized_country,
        source=normalized_source,
        language=normalized_language,
        window_start=window_start,
        window_end=window_end,
    )
    contexts = _load_dedupe_context([row.article_id for row in rows])
    dedupe_result = dedupe_embeddings(rows, contexts)
    deduped_rows = dedupe_result.rows

    base_payload: dict[str, object] = {
        "topic": normalized_topic,
        "canonical_topic_id": canonical_topic_id,
        "country": normalized_country,
        "source": normalized_source,
        "language": normalized_language,
        "window_start": window_start,
        "window_end": window_end,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": get_model_name(),
        "model_version": get_model_version(),
        "params": {
            "bootstrap_samples": bootstrap_samples,
            "sample_fraction": sample_fraction,
            "random_state": random_state,
            "umap_components": umap_components,
            "umap_neighbors": umap_neighbors,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        },
        "dedupe": {
            **dedupe_result.stats,
            "groups": [
                {
                    "representative_article_id": group.representative_article_id,
                    "member_article_ids": group.member_article_ids,
                    "reasons": group.reasons,
                }
                for group in dedupe_result.groups
            ],
        },
    }

    if len(deduped_rows) < MIN_ARTICLES_FOR_CLUSTERING:
        return {
            **base_payload,
            "status": "skipped_small_sample",
            "base": {
                "n_articles": len(deduped_rows),
                "n_clusters": 0,
                "n_noise": 0,
                "noise_rate": 0.0,
            },
            "global": {
                "completed_bootstraps": 0,
                "failed_bootstraps": 0,
                "ari": _metric_summary([]),
                "nmi": _metric_summary([]),
                "ami": _metric_summary([]),
                "noise_rate": _metric_summary([]),
            },
            "clusters": [],
        }

    base_labels, _, base_centroids, n_components, n_neighbors = cluster_rows(
        deduped_rows,
        umap_components=umap_components,
        umap_neighbors=umap_neighbors,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        random_state=random_state,
    )
    base_members = _base_cluster_members(base_labels)
    base_coherence = _cluster_coherence(deduped_rows, base_labels, base_centroids)

    rng = np.random.default_rng(random_state)
    sample_size = _sample_size(len(deduped_rows), sample_fraction)
    ari_values: list[float] = []
    nmi_values: list[float] = []
    ami_values: list[float] = []
    noise_rates: list[float] = []
    survival_by_label: dict[int, list[float]] = {
        label: [] for label in sorted(base_members)
    }
    failed_bootstraps = 0

    for _ in range(bootstrap_samples):
        sample_indices = np.sort(
            rng.choice(len(deduped_rows), size=sample_size, replace=False)
        )
        sample_rows = [deduped_rows[index] for index in sample_indices.tolist()]
        try:
            sample_labels, _, _, _, _ = cluster_rows(
                sample_rows,
                umap_components=umap_components,
                umap_neighbors=umap_neighbors,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                random_state=random_state,
            )
        except Exception:
            failed_bootstraps += 1
            continue

        base_subset_labels = base_labels[sample_indices]
        ari_values.append(
            float(adjusted_rand_score(base_subset_labels, sample_labels))
        )
        nmi_values.append(
            float(normalized_mutual_info_score(base_subset_labels, sample_labels))
        )
        ami_values.append(
            float(adjusted_mutual_info_score(base_subset_labels, sample_labels))
        )
        noise_rates.append(_noise_rate(sample_labels))

        for label, rate in _survival_rates(
            base_members=base_members,
            sample_indices=sample_indices,
            sample_labels=sample_labels,
        ).items():
            survival_by_label.setdefault(label, []).append(rate)

    clusters = []
    for label, members in sorted(base_members.items()):
        survival = survival_by_label.get(label, [])
        clusters.append(
            {
                "cluster_label": label,
                "size": len(members),
                "coherence_score": base_coherence.get(label, 0.0),
                "sampled_iterations": len(survival),
                "survival_rate": _metric_summary(survival),
            }
        )

    return {
        **base_payload,
        "status": "completed" if base_centroids else "all_noise",
        "base": {
            "n_articles": len(deduped_rows),
            "n_clusters": len(base_centroids),
            "n_noise": int(np.sum(base_labels == -1)),
            "noise_rate": round(_noise_rate(base_labels), 4),
            "umap_n_components": n_components,
            "umap_n_neighbors": n_neighbors,
        },
        "global": {
            "completed_bootstraps": len(ari_values),
            "failed_bootstraps": failed_bootstraps,
            "ari": _metric_summary(ari_values),
            "nmi": _metric_summary(nmi_values),
            "ami": _metric_summary(ami_values),
            "noise_rate": _metric_summary(noise_rates),
        },
        "clusters": clusters,
    }
