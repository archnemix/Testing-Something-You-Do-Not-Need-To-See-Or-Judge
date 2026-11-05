"""Grid-aware goal costs for the controlled Workshop task families."""

from collections import deque

from aster.environment import GateKind, Observation, Position


def remaining_cost(observation: Observation) -> float:
    """Estimate shortest valid actions to the target and required subgoals."""
    if observation.gate_open:
        return _distance(observation, observation.agent, observation.target, gate_open=True)

    after_gate = _distance(observation, observation.gate, observation.target, gate_open=True)
    if after_gate >= observation.max_steps:
        return float(observation.max_steps)

    if observation.gate_kind is GateKind.DOOR:
        approach_cost = _distance_to_gate_side(observation, observation.agent)
        if observation.has_key:
            return approach_cost + 1 + after_gate
        if observation.key is None:
            return float(observation.max_steps)
        to_key = _distance(observation, observation.agent, observation.key, gate_open=False)
        key_to_door = _distance_to_gate_side(observation, observation.key)
        if max(to_key, key_to_door) >= observation.max_steps:
            return float(observation.max_steps)
        return to_key + 1 + key_to_door + 1 + after_gate

    if observation.switch is None:
        return float(observation.max_steps)
    to_switch = _distance(observation, observation.agent, observation.switch, gate_open=False)
    after_switch = _distance(observation, observation.switch, observation.target, gate_open=True)
    if max(to_switch, after_switch) >= observation.max_steps:
        return float(observation.max_steps)
    return to_switch + 1 + after_switch


def _distance_to_gate_side(observation: Observation, position: Position) -> int:
    sides = {
        Position(observation.gate.x, observation.gate.y - 1),
        Position(observation.gate.x, observation.gate.y + 1),
        Position(observation.gate.x - 1, observation.gate.y),
        Position(observation.gate.x + 1, observation.gate.y),
    }
    distances = [
        distance
        for side in sides
        if _passable(observation, side, gate_open=False)
        and (distance := _distance(observation, position, side, gate_open=False))
        < observation.max_steps
    ]
    return min(distances) if distances else observation.max_steps


def _distance(
    observation: Observation,
    start: Position,
    target: Position,
    gate_open: bool,
) -> int:
    if start == target:
        return 0
    if start in observation.walls:
        return observation.max_steps

    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        position, distance = frontier.popleft()
        for neighbor in _neighbors(position):
            if neighbor in visited or not _passable(observation, neighbor, gate_open):
                continue
            if neighbor == target:
                return distance + 1
            visited.add(neighbor)
            frontier.append((neighbor, distance + 1))
    return observation.max_steps


def _passable(observation: Observation, position: Position, gate_open: bool) -> bool:
    if not (0 <= position.x < observation.width and 0 <= position.y < observation.height):
        return False
    if position in observation.walls:
        return False
    return position != observation.gate or gate_open


def _neighbors(position: Position) -> tuple[Position, ...]:
    return (
        Position(position.x, position.y - 1),
        Position(position.x, position.y + 1),
        Position(position.x + 1, position.y),
        Position(position.x - 1, position.y),
    )
