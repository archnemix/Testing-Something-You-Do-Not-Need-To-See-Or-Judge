"""Stable numeric encoding used by trainable models and baseline tests."""

from aster.environment.types import GateKind, Observation, Position

CHANNELS = ("wall", "agent", "target", "gate", "key", "switch")
GLOBAL_FEATURES = ("has_key", "gate_open", "gate_is_bridge", "time_remaining")


def encode_observation(observation: Observation) -> tuple[float, ...]:
    """Encode a structured Workshop observation in row-major cell order."""
    wall_cells = observation.walls
    key = observation.key
    switch = observation.switch
    values: list[float] = []

    for y in range(observation.height):
        for x in range(observation.width):
            position = Position(x, y)
            values.extend(
                (
                    float(position in wall_cells),
                    float(position == observation.agent),
                    float(position == observation.target),
                    float(position == observation.gate),
                    float(position == key),
                    float(position == switch),
                )
            )

    remaining_steps = max(observation.max_steps - observation.steps, 0)
    values.extend(
        (
            float(observation.has_key),
            float(observation.gate_open),
            float(observation.gate_kind is GateKind.BRIDGE),
            remaining_steps / observation.max_steps,
        )
    )
    return tuple(values)
