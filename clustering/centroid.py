from __future__ import annotations

import numpy as np


def compute_centroids(
    labels: np.ndarray,
    original_vectors: np.ndarray,
) -> dict[int, list[float]]:
    centroids: dict[int, list[float]] = {}
    unique_labels = set(int(label) for label in np.unique(labels)) - {-1}
    for label in unique_labels:
        member_vectors = original_vectors[labels == label]
        centroid = member_vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[label] = centroid.tolist()
    return centroids
