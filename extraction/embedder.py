from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from config.settings import EMBED_BATCH_SIZE, EMBED_MODEL_NAME


SUPPORTED_EMBED_MODELS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}
EXPECTED_EMBED_DIM = SUPPORTED_EMBED_MODELS.get(EMBED_MODEL_NAME)
if EXPECTED_EMBED_DIM is None:
    raise ValueError(
        f"Unsupported embedding model {EMBED_MODEL_NAME!r}. "
        f"Supported models: {sorted(SUPPORTED_EMBED_MODELS)}"
    )

_model = None


def get_model_name() -> str:
    return EMBED_MODEL_NAME


def get_expected_dim() -> int:
    return EXPECTED_EMBED_DIM


def get_model_version() -> str | None:
    try:
        return version("sentence-transformers")
    except PackageNotFoundError:
        return None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    vectors = embeddings.tolist()
    for vector in vectors:
        dim = len(vector)
        if dim != EXPECTED_EMBED_DIM:
            raise AssertionError(f"Unexpected embedding dim {dim} for {EMBED_MODEL_NAME}")
    return vectors
