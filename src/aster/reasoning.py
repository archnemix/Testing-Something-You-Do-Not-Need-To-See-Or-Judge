"""Retrieval by spreading activation over the plastic network.

Answers are not looked up: activation spreads along learned pathways and
the concepts that fire win the answer slots. Direct links arrive at hop 1,
chains like transitive `is` arrive at later hops.
"""

from aster.memory import MemoryStore
from aster.models import Fact


def answers_with_hops_for(
    subject: str, relation: str, memory: MemoryStore
) -> list[tuple[str, int]]:
    return memory.network.retrieve(subject, relation)


def answers_for(subject: str, relation: str, memory: MemoryStore) -> list[str]:
    return [label for label, _ in answers_with_hops_for(subject, relation, memory)]


def render_fact(fact: Fact) -> str:
    return f"{fact.subject} {fact.relation} {fact.object}."
