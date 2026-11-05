"""Short-horizon model-based planning with real feedback between actions."""

from dataclasses import dataclass

from aster.cognition.workshop_goal import remaining_cost
from aster.cognition.world_model import TransitionModel, predict_transitions
from aster.environment import (
    Action,
    GateKind,
    Observation,
    Position,
    available_actions,
    encode_observation,
)
from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES


@dataclass(frozen=True, slots=True)
class PlanningResult:
    actions: tuple[Action, ...]
    score: float
    predicted_success: bool
    expanded_nodes: int


@dataclass(frozen=True, slots=True)
class _BeamNode:
    observation: Observation
    actions: tuple[Action, ...]
    score: float
    visited: frozenset[tuple[object, ...]]


def plan(
    model: TransitionModel,
    initial: Observation,
    beam_width: int = 64,
    horizon: int = 18,
) -> PlanningResult | None:
    """Search predicted futures and return actions; execute only the first one."""
    if beam_width < 1 or horizon < 1:
        raise ValueError("beam_width and horizon must be positive")
    if initial.agent == initial.target:
        return PlanningResult((), 0.0, True, 0)

    initial_key = _state_key(initial)
    initial_score = -remaining_cost(initial)
    beam = [_BeamNode(initial, (), initial_score, frozenset({initial_key}))]
    best_node = beam[0]
    expanded_nodes = 0
    action_space = tuple(Action)

    for depth in range(1, horizon + 1):
        inputs: list[tuple[float, ...]] = []
        actions: list[Action] = []
        parents: list[_BeamNode] = []
        for parent in beam:
            for action in action_space:
                inputs.append(encode_observation(parent.observation))
                actions.append(action)
                parents.append(parent)

        predictions = predict_transitions(model, inputs, actions)
        candidates: list[_BeamNode] = []
        completed: list[_BeamNode] = []
        for parent, action, prediction in zip(parents, actions, predictions):
            expanded_nodes += 1
            next_values, reward, done_probability, invalid_probability = prediction
            if invalid_probability >= 0.5:
                continue
            next_observation = decode_prediction(next_values, parent.observation, action)
            key = _state_key(next_observation)
            if key in parent.visited:
                continue

            sequence = parent.actions + (action,)
            progress = remaining_cost(parent.observation) - remaining_cost(next_observation)
            score = parent.score + progress + reward - 0.01
            node = _BeamNode(
                observation=next_observation,
                actions=sequence,
                score=score,
                visited=parent.visited | {key},
            )
            if next_observation.agent == next_observation.target:
                completed.append(node)
            elif done_probability < 0.5:
                candidates.append(node)

        if completed:
            winner = max(completed, key=lambda node: node.score)
            return PlanningResult(winner.actions, winner.score, True, expanded_nodes)
        if not candidates:
            break

        candidates.sort(key=lambda node: node.score, reverse=True)
        beam = _deduplicate(candidates, beam_width)
        depth_best = max(beam, key=lambda node: node.score)
        if depth_best.score > best_node.score:
            best_node = depth_best

    if not best_node.actions or best_node.score <= initial_score:
        return None
    return PlanningResult(best_node.actions, best_node.score, False, expanded_nodes)


def decode_prediction(
    values: tuple[float, ...],
    template: Observation,
    action: Action | None = None,
) -> Observation:
    """Decode learned motion while enforcing known Workshop action preconditions."""
    cell_count = template.width * template.height
    expected_size = cell_count * len(CHANNELS) + len(GLOBAL_FEATURES)
    if len(values) != expected_size:
        raise ValueError(f"expected {expected_size} predicted features, received {len(values)}")

    agent = _position_for_channel(values, template.width, template.height, 1)
    movement = action.movement if action is not None else None
    if action in available_actions(template) and movement is not None:
        agent = Position(template.agent.x + movement[0], template.agent.y + movement[1])
    elif action is not None and movement is None:
        agent = template.agent

    has_key = template.has_key or (
        action is Action.PICK_UP and template.key == template.agent
    )
    key_position = None if has_key else template.key
    gate_open = template.gate_open
    if action is Action.INTERACT:
        adjacent_to_door = (
            abs(template.agent.x - template.gate.x)
            + abs(template.agent.y - template.gate.y)
            == 1
        )
        if template.gate_kind is GateKind.DOOR and has_key and adjacent_to_door:
            gate_open = True
        elif template.gate_kind is GateKind.BRIDGE and template.agent == template.switch:
            gate_open = not template.gate_open
    steps = min(template.steps + 1, template.max_steps)
    return Observation(
        width=template.width,
        height=template.height,
        walls=template.walls,
        agent=agent,
        target=template.target,
        gate_kind=template.gate_kind,
        gate=template.gate,
        gate_open=gate_open,
        key=key_position,
        has_key=has_key,
        switch=template.switch,
        steps=steps,
        max_steps=template.max_steps,
    )


def _position_for_channel(
    values: tuple[float, ...],
    width: int,
    height: int,
    channel: int,
    threshold: float | None = None,
) -> Position | None:
    cells = [
        (values[(y * width + x) * len(CHANNELS) + channel], Position(x, y))
        for y in range(height)
        for x in range(width)
    ]
    value, position = max(cells, key=lambda item: item[0])
    if threshold is not None and value < threshold:
        return None
    return position



def _state_key(observation: Observation) -> tuple[object, ...]:
    return (
        observation.agent,
        observation.key,
        observation.has_key,
        observation.gate_open,
    )


def _deduplicate(nodes: list[_BeamNode], beam_width: int) -> list[_BeamNode]:
    best_for_state: dict[tuple[object, ...], _BeamNode] = {}
    for node in nodes:
        key = _state_key(node.observation)
        current = best_for_state.get(key)
        if current is None or node.score > current.score:
            best_for_state[key] = node
    return sorted(best_for_state.values(), key=lambda node: node.score, reverse=True)[:beam_width]

