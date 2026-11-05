"""Typed state and action records for the Workshop environment."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True, order=True)
class Position:
    x: int
    y: int


class Action(Enum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    PICK_UP = "pick_up"
    INTERACT = "interact"
    WAIT = "wait"

    @property
    def movement(self) -> tuple[int, int] | None:
        return {
            Action.MOVE_NORTH: (0, -1),
            Action.MOVE_SOUTH: (0, 1),
            Action.MOVE_EAST: (1, 0),
            Action.MOVE_WEST: (-1, 0),
        }.get(self)


class GateKind(Enum):
    DOOR = "door"
    BRIDGE = "bridge"


@dataclass(frozen=True, slots=True)
class WorkshopLayout:
    width: int
    height: int
    walls: frozenset[Position]
    agent_start: Position
    target: Position
    gate_kind: GateKind
    gate: Position
    max_steps: int = 64
    key: Position | None = None
    switch: Position | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    width: int
    height: int
    walls: frozenset[Position]
    agent: Position
    target: Position
    gate_kind: GateKind
    gate: Position
    gate_open: bool
    key: Position | None
    has_key: bool
    switch: Position | None
    steps: int
    max_steps: int


@dataclass(frozen=True, slots=True)
class StepResult:
    previous_observation: Observation
    observation: Observation
    action: Action
    event: str
    reward: float
    done: bool
    invalid: bool
