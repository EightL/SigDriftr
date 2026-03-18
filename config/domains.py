from __future__ import annotations


DOMAIN_SIGNAL_KEYS = [
    "concern_level",
    "purchase_intent",
    "avoidance_signals",
]

DOMAINS: dict[str, dict[str, object]] = {
    "commerce": {
        "relevant_fields": [
            "concern_level",
            "purchase_intent",
            "avoidance_signals",
        ],
        "signal_weights": {
            "concern_level": 0.4,
            "purchase_intent": 0.4,
            "avoidance_signals": 0.2,
        },
        "prompt_hint": (
            "Focus on consumer behavior: purchasing decisions, product avoidance, "
            "and financial concerns."
        ),
    },
    "civic": {
        "relevant_fields": ["concern_level", "avoidance_signals"],
        "signal_weights": {
            "concern_level": 0.6,
            "purchase_intent": 0.0,
            "avoidance_signals": 0.4,
        },
        "prompt_hint": (
            "Focus on civic concern and social avoidance behaviors. Purchase "
            "intent is not relevant here; keep it near 0.0."
        ),
    },
    "health": {
        "relevant_fields": ["concern_level", "avoidance_signals"],
        "signal_weights": {
            "concern_level": 0.6,
            "purchase_intent": 0.0,
            "avoidance_signals": 0.4,
        },
        "prompt_hint": (
            "Focus on health-related concern and avoidance. Purchase intent is "
            "not relevant unless this is about healthcare products; keep it near 0.0."
        ),
    },
    "generic": {
        "relevant_fields": ["concern_level"],
        "signal_weights": {
            "concern_level": 1.0,
            "purchase_intent": 0.0,
            "avoidance_signals": 0.0,
        },
        "prompt_hint": (
            "Focus only on general concern level. Purchase intent and avoidance "
            "are not relevant; keep them near 0.0."
        ),
    },
}

DEFAULT_DOMAIN = "generic"

TOPIC_DOMAIN_RULES: list[tuple[list[str], str]] = [
    (
        [
            "ceny",
            "energie",
            "nafta",
            "benzin",
            "inflace",
            "nakup",
            "obchod",
            "trh",
            "ekonomika",
            "hdp",
            "spotreba",
        ],
        "commerce",
    ),
    (
        [
            "politika",
            "volby",
            "vlade",
            "parlament",
            "demokracie",
            "protest",
            "bezpecnost",
            "migrace",
            "eu",
        ],
        "civic",
    ),
    (
        [
            "zdravi",
            "nemoc",
            "covid",
            "vakcina",
            "nemocnice",
            "leky",
            "epidemie",
            "psychologie",
        ],
        "health",
    ),
]


def topic_to_domain(topic: str) -> str:
    normalized = topic.strip().lower()
    if not normalized:
        return DEFAULT_DOMAIN

    for keywords, domain in TOPIC_DOMAIN_RULES:
        if any(keyword in normalized for keyword in keywords):
            return domain
    return DEFAULT_DOMAIN


def get_domain_config(domain: str) -> dict[str, object]:
    return DOMAINS.get(domain, DOMAINS[DEFAULT_DOMAIN])
