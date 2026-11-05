"""One-step, model-guided control with explicit subgoal structure and feedback."""

from dataclasses import dataclass

from aster.cognition.planner import decode_prediction
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


@dataclass(frozen=True, slots=True)
class ActionChoice:
    action: Action
    expected_observation: Observation
    predicted_reward: float
    expected_progress: float


def choose_action(
    model: TransitionModel,
    observation: Observation,
    excluded_actions: frozenset[Action] = frozenset(),
) -> ActionChoice | None:
    """Choose a single action from learned predictions, then reobserve and repeat."""
    actions = available_actions(observation)
    state = encode_observation(observation)
    predictions = predict_transitions(model, [state] * len(actions), actions)
    current_cost = remaining_cost(observation)
    candidates: list[tuple[tuple[float, int, float, float], ActionChoice]] = []

    for action, prediction in zip(actions, predictions):
        next_state, reward, _done_probability, invalid_probability = prediction
        if action in excluded_actions:
            continue
        predicted = decode_prediction(next_state, observation, action)
        next_cost = remaining_cost(predicted)
        progress = current_cost - next_cost
        if progress < 0:
            continue

        priority = _action_priority(action, observation)
        rank = (next_cost, priority, invalid_probability, -reward)
        candidates.append(
            (
                rank,
                ActionChoice(action, predicted, reward, progress),
            )
        )

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _action_priority(action: Action, observation: Observation) -> int:
    if action is Action.PICK_UP and observation.key == observation.agent:
        return 0
    if action is not Action.INTERACT or observation.gate_open:
        return 1
    if observation.gate_kind is GateKind.DOOR and observation.has_key:
        if _are_adjacent(observation.agent, observation.gate):
            return 0
    if observation.gate_kind is GateKind.BRIDGE and observation.agent == observation.switch:
        return 0
    return 1


def _are_adjacent(left: Position, right: Position) -> bool:
    return abs(left.x - right.x) + abs(left.y - right.y) == 1
