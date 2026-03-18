from functools import lru_cache


ENTITY_LABELS = {"PERSON", "ORG", "GPE"}


@lru_cache(maxsize=1)
def _get_nlp():
    try:
        import spacy

        return spacy.load("cs_core_news_sm")
    except Exception:
        return None


def _normalize_entity_text(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_entity_key(text: str) -> str:
    return _normalize_entity_text(text).lower()


def extract_entities(text: str, limit: int = 6) -> list[dict[str, str]]:
    if not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for entity in doc.ents:
        label = entity.label_.upper()
        if label not in ENTITY_LABELS:
            continue
        normalized = _normalize_entity_text(entity.text)
        if not normalized:
            continue
        key = (normalized.lower(), label)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"text": normalized, "label": label})
        if len(entities) >= limit:
            break

    return entities
