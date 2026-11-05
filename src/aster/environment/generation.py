"""Seeded layouts for repeatable Workshop scenarios."""

import random

from aster.environment.types import GateKind, Position, WorkshopLayout


def make_workshop_layout(
    gate_kind: GateKind,
    seed: int,
    width: int = 7,
    height: int = 5,
    max_steps: int = 64,
) -> WorkshopLayout:
    """Create a two-room task with one stateful crossing and a guaranteed route."""
    if width < 5 or height < 3:
        raise ValueError("Workshop layouts need width >= 5 and height >= 3")

    randomizer = random.Random(seed)
    gate_x = width // 2
    gate_y = randomizer.randrange(1, height - 1)
    available_rows = [row for row in range(height) if row != gate_y]
    key_or_switch_y = randomizer.choice(available_rows)
    target_y = randomizer.randrange(height)

    crossing = Position(gate_x, gate_y)
    barrier = frozenset(
        Position(gate_x, row)
        for row in range(height)
        if row != gate_y
    )
    gate_side_object = Position(1, key_or_switch_y)

    return WorkshopLayout(
        width=width,
        height=height,
        walls=barrier,
        agent_start=Position(1, gate_y),
        target=Position(width - 2, target_y),
        gate_kind=gate_kind,
        gate=crossing,
        max_steps=max_steps,
        key=gate_side_object if gate_kind is GateKind.DOOR else None,
        switch=gate_side_object if gate_kind is GateKind.BRIDGE else None,
    )
