from __future__ import annotations

import numpy as np


def reduce_embeddings(
    vectors: np.ndarray,
    *,
    n_components: int = 10,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")

    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
    )
    reduced = reducer.fit_transform(vectors)
    return np.asarray(reduced)
