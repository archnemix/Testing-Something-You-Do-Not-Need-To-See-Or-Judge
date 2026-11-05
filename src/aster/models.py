"""Data structures shared by the cognitive subsystems."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class Fact:
    subject: str
    relation: str
    object: str

    def key(self) -> tuple[str, str, str]:
        return tuple(part.casefold() for part in (self.subject, self.relation, self.object))


@dataclass(frozen=True, slots=True)
class Goal:
    description: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class Episode:
    text: str
    kind: str


class InputKind(Enum):
    FACT = "fact"
    QUESTION = "question"
    GOAL = "goal"
    STATUS = "status"
    HELP = "help"
    QUIT = "quit"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ParsedInput:
    kind: InputKind
    text: str
    fact: Fact | None = None
    subject: str | None = None
    relation: str | None = None
