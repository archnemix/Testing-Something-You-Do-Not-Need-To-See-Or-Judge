"""Small deterministic input grammar; this module does not generate language."""

import re

from aster.models import Fact, InputKind, ParsedInput

_FACT = re.compile(r"^(.+?)\s+(is|has)\s+(.+?)\s*[.!?]*$", re.IGNORECASE)
_IS_QUESTION = re.compile(r"^(?:what|who)\s+is\s+(.+?)\s*\??$", re.IGNORECASE)
_HAS_QUESTION = re.compile(r"^what\s+does\s+(.+?)\s+have\s*\??$", re.IGNORECASE)
_GOAL = re.compile(r"^(?:goal\s*:\s*|my goal is\s+)(.+?)\s*[.!]*$", re.IGNORECASE)


def parse_input(text: str) -> ParsedInput:
    """Classify supported commands, preserving names and fact wording."""
    cleaned = text.strip()
    folded = cleaned.casefold()

    if folded in {"quit", "exit", "q"}:
        return ParsedInput(InputKind.QUIT, cleaned)
    if folded in {"status", "what are you working on?"}:
        return ParsedInput(InputKind.STATUS, cleaned)
    if folded in {"help", "?"}:
        return ParsedInput(InputKind.HELP, cleaned)

    goal_match = _GOAL.match(cleaned)
    if goal_match:
        return ParsedInput(InputKind.GOAL, cleaned, subject=goal_match.group(1).strip())

    question_match = _IS_QUESTION.match(cleaned)
    if question_match:
        return ParsedInput(
            InputKind.QUESTION,
            cleaned,
            subject=question_match.group(1).strip().rstrip("?.!"),
            relation="is",
        )

    question_match = _HAS_QUESTION.match(cleaned)
    if question_match:
        return ParsedInput(
            InputKind.QUESTION,
            cleaned,
            subject=question_match.group(1).strip().rstrip("?.!"),
            relation="has",
        )

    fact_match = _FACT.match(cleaned)
    if fact_match:
        subject, relation, obj = (part.strip().rstrip(".!?") for part in fact_match.groups())
        if subject and obj:
            fact = Fact(subject, relation.casefold(), obj)
            return ParsedInput(InputKind.FACT, cleaned, fact=fact)

    return ParsedInput(InputKind.OTHER, cleaned)
