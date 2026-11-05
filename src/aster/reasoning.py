"""Symbolic retrieval and a deliberately small transitive inference rule."""

from aster.memory import MemoryStore
from aster.models import Fact


def answers_for(subject: str, relation: str, memory: MemoryStore) -> list[str]:
    direct = memory.find_facts(subject, relation)
    values = [fact.object for fact in direct]
    if relation.casefold() == "is":
        answered = {value.casefold() for value in values}
        answered.add(subject.casefold())
        values.extend(_is_closure(subject, memory, answered))
    return values


def _is_closure(subject: str, memory: MemoryStore, answered: set[str]) -> list[str]:
    """Follow stored `is` links without looping on cyclic or repeated facts."""
    discovered: list[str] = []
    visited_nodes = {subject.casefold()}
    frontier = [subject]
    while frontier:
        current = frontier.pop(0)
        for fact in memory.find_facts(current, "is"):
            key = fact.object.casefold()
            if key not in answered:
                answered.add(key)
                discovered.append(fact.object)
            if key not in visited_nodes:
                visited_nodes.add(key)
                frontier.append(fact.object)
    return discovered


def render_fact(fact: Fact) -> str:
    return f"{fact.subject} {fact.relation} {fact.object}."
