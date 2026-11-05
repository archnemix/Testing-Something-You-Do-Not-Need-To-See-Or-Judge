"""Persistent long-term facts plus a bounded, recent working-memory window."""

import json
from collections import deque
from pathlib import Path
from typing import Any

from aster.models import Episode, Fact, Goal


class MemoryStore:
    def __init__(self, path: Path | None = None, working_capacity: int = 7) -> None:
        if working_capacity < 1:
            raise ValueError("working_capacity must be at least 1")
        self.path = path
        self.working_capacity = working_capacity
        self.facts: list[Fact] = []
        self.goals: list[Goal] = []
        self.episodes: deque[Episode] = deque(maxlen=working_capacity)
        if path is not None and path.exists():
            self._load()

    def remember_fact(self, fact: Fact) -> bool:
        """Store a fact once; return whether the store changed."""
        if any(existing.key() == fact.key() for existing in self.facts):
            return False
        self.facts.append(fact)
        self._record(f"{fact.subject} {fact.relation} {fact.object}", "fact")
        self.save()
        return True

    def find_facts(self, subject: str, relation: str) -> list[Fact]:
        subject_key = subject.casefold()
        relation_key = relation.casefold()
        return [
            fact
            for fact in self.facts
            if fact.subject.casefold() == subject_key and fact.relation.casefold() == relation_key
        ]

    def set_goal(self, description: str) -> None:
        self.goals = [Goal(description, active=True)]
        self._record(description, "goal")
        self.save()

    def record_input(self, text: str) -> None:
        self._record(text, "input")
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "facts": [
                {"subject": fact.subject, "relation": fact.relation, "object": fact.object}
                for fact in self.facts
            ],
            "goals": [
                {"description": goal.description, "active": goal.active}
                for goal in self.goals
            ],
            "episodes": [{"text": item.text, "kind": item.kind} for item in self.episodes],
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(self.path)

    def _record(self, text: str, kind: str) -> None:
        self.episodes.append(Episode(text=text, kind=kind))

    def _load(self) -> None:
        assert self.path is not None
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("memory file must contain a JSON object")
            if data.get("version") != 1:
                raise ValueError("unsupported memory file version")
            self.facts = [Fact(**item) for item in data.get("facts", [])]
            self.goals = [Goal(**item) for item in data.get("goals", [])]
            self.episodes.extend(Episode(**item) for item in data.get("episodes", []))
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            raise ValueError(f"Cannot read Aster memory at {self.path}: {error}") from error
