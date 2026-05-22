"""Constants and shared exceptions."""

DEFAULT_MODELS = {
    "defender": "claude-haiku-4-5-20251001",
    "skeptic": "claude-haiku-4-5-20251001",
    "fact_checker": "claude-haiku-4-5-20251001",
    "evaluator": "claude-haiku-4-5-20251001",
    "summarizer": "claude-haiku-4-5-20251001",
    "researcher": "claude-haiku-4-5-20251001",
    "analyst": "claude-haiku-4-5-20251001",
}

DEFAULT_EFFORTS = {
    "defender": None,
    "skeptic": None,
    "fact_checker": None,
    "evaluator": None,
    "summarizer": None,
    "researcher": None,
    "analyst": None,
}

VERDICT_SCALE = [
    "CONFIRMED", "STRONGLY SUPPORTED", "LIKELY TRUE",
    "INCONCLUSIVE",
    "LIKELY FALSE", "STRONGLY REFUTED", "REFUTED",
]

CLAIM_CLASSES = (
    "FACTUAL_TESTABLE",
    "PREDICTIVE_FUTURE",
    "VALUE_NORMATIVE",
    "HARM_INDUCING_FALSEHOOD",
)


class AgentError(Exception):
    """Raised when a Claude CLI agent call fails."""
