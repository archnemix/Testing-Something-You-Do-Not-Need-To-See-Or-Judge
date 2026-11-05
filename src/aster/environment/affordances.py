"""Observable action preconditions for the Workshop actuator."""

from aster.environment.types import Action, GateKind, Observation, Position


def available_actions(observation: Observation) -> tuple[Action, ...]:
    """Return actions whose known Workshop preconditions are satisfied."""
    actions: list[Action] = [Action.WAIT]
    for action in (Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST):
        delta = action.movement
        assert delta is not None
        destination = Position(observation.agent.x + delta[0], observation.agent.y + delta[1])
        if _can_enter(observation, destination):
            actions.append(action)

    if observation.key == observation.agent and not observation.has_key:
        actions.append(Action.PICK_UP)
    if _can_interact(observation):
        actions.append(Action.INTERACT)
    return tuple(actions)


def _can_enter(observation: Observation, position: Position) -> bool:
    if not (0 <= position.x < observation.width and 0 <= position.y < observation.height):
        return False
    if position in observation.walls:
        return False
    return position != observation.gate or observation.gate_open


def _can_interact(observation: Observation) -> bool:
    if observation.gate_open:
        return observation.gate_kind is GateKind.BRIDGE and observation.agent == observation.switch
    if observation.gate_kind is GateKind.DOOR:
        return observation.has_key and _adjacent(observation.agent, observation.gate)
    return observation.agent == observation.switch


def _adjacent(left: Position, right: Position) -> bool:
    return abs(left.x - right.x) + abs(left.y - right.y) == 1
