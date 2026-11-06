"""Coordinates perception, memory, associative retrieval, and responses."""

from pathlib import Path

from aster.language import parse_input
from aster.memory import MemoryStore
from aster.models import InputKind
from aster.reasoning import answers_with_hops_for


class Aster:
    def __init__(
        self, memory_path: Path | None = None, max_neurons: int | None = None
    ) -> None:
        self.memory = MemoryStore(memory_path, max_neurons=max_neurons)

    def respond(self, text: str) -> str:
        parsed = parse_input(text)
        self.memory.network.tick(1)  # a moment passes: idle pathways fade a little
        if parsed.kind is InputKind.QUIT:
            return "Goodbye."
        if parsed.kind is InputKind.HELP:
            return (
                "Teach a fact with 'Ada is a mathematician' or 'Ada has a notebook'. "
                "Ask 'what is Ada?' or 'what does Ada have?'. Set a goal with "
                "'goal: learn about Ada', inspect it with 'status', or type 'quit'."
            )
        if parsed.kind is InputKind.FACT and parsed.fact is not None:
            added = self.memory.remember_fact(parsed.fact)
            if added:
                fact = parsed.fact
                return f"Remembered: {fact.subject} {fact.relation} {fact.object}."
            return "I already have that fact."
        if parsed.kind is InputKind.QUESTION and parsed.subject and parsed.relation:
            self.memory.record_input(text)
            results = answers_with_hops_for(parsed.subject, parsed.relation, self.memory)
            if not results:
                return f"I don't have a stored answer for {parsed.subject} {parsed.relation} yet."
            direct_answers = [label for label, hop in results if hop == 1]
            inferred_answers = [label for label, hop in results if hop > 1]
            sentences = [f"{parsed.subject} {parsed.relation} {value}." for value in direct_answers]
            for value in inferred_answers:
                sentences.append(
                    f"By following stored '{parsed.relation}' facts, "
                    f"{parsed.subject} is also {value}."
                )
            return " ".join(sentences)
        if parsed.kind is InputKind.GOAL and parsed.subject:
            self.memory.set_goal(parsed.subject)
            return f"Goal recorded: {parsed.subject}."
        if parsed.kind is InputKind.STATUS:
            active_goals = [goal.description for goal in self.memory.goals if goal.active]
            fact_count = self.memory.known_fact_count()
            if not active_goals:
                return f"No active goal. I have {fact_count} stored fact(s)."
            return f"Active goal: {active_goals[0]}. I have {fact_count} stored fact(s)."
        self.memory.record_input(text)
        return (
            "I don't have a reliable interpretation for that yet. I can store simple "
            "'subject is object' and 'subject has object' facts, answer supported "
            "questions, and track one active goal. Type 'help' to see examples."
        )
