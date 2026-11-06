"""Long-term memory: a plastic neuron network, plus goals and a small
recent-conversation window.

Facts are not stored in a list or dictionary. Telling a fact strengthens
pathways between neuron ensembles in `PlasticNetwork`; asking spreads
activation along those pathways. Repeated use strengthens pathways further
and idle time weakens them. Goals and the bounded episode window are plain
persistent state (like a small notepad), not part of the learned substrate.
"""

import json
from collections import deque
from pathlib import Path
from typing import Any

from aster.cognition.plastic import PlasticNetwork
from aster.models import Episode, Fact, Goal


class MemoryStore:
    def __init__(
        self,
        path: Path | None = None,
        working_capacity: int = 7,
        max_neurons: int | None = None,
    ) -> None:
        if working_capacity < 1:
            raise ValueError("working_capacity must be at least 1")
        self.path = path
        self.working_capacity = working_capacity
        self.network = PlasticNetwork(max_neurons=max_neurons)
        self.goals: list[Goal] = []
        self.episodes: deque[Episode] = deque(maxlen=working_capacity)
        if path is not None and path.exists():
            self._load()

    def remember_fact(self, fact: Fact) -> bool:
        """Tell the network a fact; return whether it counts as new knowledge.

        Repetition is recognized (and still strengthens the pathway), but
        only new knowledge is recorded in the episode window.
        """
        is_new = self.network.observe(fact.subject, fact.relation, fact.object)
        if is_new:
            self._record(f"{fact.subject} {fact.relation} {fact.object}", "fact")
        self.save()
        return is_new

    def known_fact_count(self) -> int:
        return self.network.known_fact_count()

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
            "version": 2,
            "network": self.network.snapshot(),
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
            version = data.get("version")
            if version == 2:
                self.network.restore(data.get("network") or {})
            elif version == 1:
                self._consolidate_old_facts(data)
            else:
                raise ValueError("unsupported memory file version")
            self.goals = [Goal(**item) for item in data.get("goals", [])]
            self.episodes.extend(Episode(**item) for item in data.get("episodes", []))
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            raise ValueError(f"Cannot read Aster memory at {self.path}: {error}") from error

    def _consolidate_old_facts(self, data: dict[str, Any]) -> None:
        """Replay version-1 fact lists through the network (like re-telling)."""
        for item in data.get("facts", []):
            self.network.observe(item["subject"], item["relation"], item["object"])
