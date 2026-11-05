"""Baseline policies for data collection and controlled comparisons."""

from collections import deque

from aster.environment.types import Action, Observation, Position, StepResult, WorkshopLayout
from aster.environment.workshop import Workshop

_MOVE_TO_POSITION = {
    (0, -1): Action.MOVE_NORTH,
    (0, 1): Action.MOVE_SOUTH,
    (1, 0): Action.MOVE_EAST,
    (-1, 0): Action.MOVE_WEST,
}


def choose_reactive_action(observation: Observation) -> Action:
    """Move toward the target while ignoring state-changing objects.

    This baseline never picks up keys and never operates switches or
    doors; it plans as if the gate were already passable. Gated tasks are
    therefore unsolvable for it, which isolates the value of modeling
    state-changing objects.
    """
    if observation.agent == observation.target:
        return Action.WAIT
    path = _path(
        observation,
        observation.agent,
        {observation.target},
        gate_open=True,
    )
    return path[0] if path else Action.WAIT


def collect_expert_episode(layout: WorkshopLayout) -> list[StepResult]:
    """Solve one layout using known rules; never use this policy at model runtime."""
    environment = Workshop(layout)
    route: list[Action] = []
    current = layout.agent_start

    if layout.key is not None:
        segment = _path(layout, current, {layout.key}, gate_open=False)
        route.extend(segment)
        current = _follow(current, segment)
        route.append(Action.PICK_UP)
        adjacent_cells = set(_neighbors(layout.gate))
        segment = _path(layout, current, adjacent_cells, gate_open=False)
        route.extend(segment)
        current = _follow(current, segment)
        route.append(Action.INTERACT)
    else:
        assert layout.switch is not None
        segment = _path(layout, current, {layout.switch}, gate_open=False)
        route.extend(segment)
        current = _follow(current, segment)
        route.append(Action.INTERACT)

    route.extend(_path(layout, current, {layout.target}, gate_open=True))
    return [environment.step(action) for action in route]


def _path(
    geometry: WorkshopLayout | Observation,
    start: Position,
    goals: set[Position],
    gate_open: bool,
) -> list[Action]:
    frontier = deque([start])
    previous: dict[Position, tuple[Position, Action] | None] = {start: None}

    while frontier:
        current = frontier.popleft()
        if current in goals:
            return _reconstruct_path(current, previous)
        for neighbor, action in _neighbors_with_actions(current):
            if neighbor in previous or not _is_passable(geometry, neighbor, gate_open):
                continue
            previous[neighbor] = (current, action)
            frontier.append(neighbor)

    raise ValueError("No route exists for this Workshop layout")


def _follow(start: Position, actions: list[Action]) -> Position:
    position = start
    for action in actions:
        movement = action.movement
        assert movement is not None
        position = Position(position.x + movement[0], position.y + movement[1])
    return position


def _reconstruct_path(
    destination: Position,
    previous: dict[Position, tuple[Position, Action] | None],
) -> list[Action]:
    actions: list[Action] = []
    current = destination
    while previous[current] is not None:
        step = previous[current]
        assert step is not None
        current, action = step
        actions.append(action)
    actions.reverse()
    return actions


def _is_passable(
    geometry: WorkshopLayout | Observation,
    position: Position,
    gate_open: bool,
) -> bool:
    if not (0 <= position.x < geometry.width and 0 <= position.y < geometry.height):
        return False
    if position in geometry.walls:
        return False
    return position != geometry.gate or gate_open


def _neighbors(position: Position) -> list[Position]:
    return [
        Position(position.x, position.y - 1),
        Position(position.x, position.y + 1),
        Position(position.x + 1, position.y),
        Position(position.x - 1, position.y),
    ]


def _neighbors_with_actions(position: Position) -> list[tuple[Position, Action]]:
    return [
        (Position(position.x + dx, position.y + dy), action)
        for (dx, dy), action in _MOVE_TO_POSITION.items()
    ]
