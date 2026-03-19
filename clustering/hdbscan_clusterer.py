from __future__ import annotations

import numpy as np


def cluster_reduced(
    reduced_vectors: np.ndarray,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if reduced_vectors.ndim != 2:
        raise ValueError("Expected a 2D reduced embedding matrix.")

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(reduced_vectors)
    return np.asarray(clusterer.labels_), np.asarray(clusterer.probabilities_)
