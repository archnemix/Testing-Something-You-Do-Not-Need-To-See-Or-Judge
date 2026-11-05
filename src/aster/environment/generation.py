"""Seeded layouts for repeatable Workshop scenarios."""

import random

from aster.environment.types import GateKind, Position, WorkshopLayout

LAYOUT_SPACE_VERSION = 2


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
    gate_x = randomizer.randrange(2, width - 2)
    gate_y = randomizer.randrange(1, height - 1)
    start_y = randomizer.randrange(height)
    target_y = randomizer.randrange(height)
    object_x = randomizer.randrange(1, gate_x)
    object_rows = [
        row
        for row in range(height)
        if row != gate_y and not (object_x == 1 and row == start_y)
    ]
    object_y = randomizer.choice(object_rows)

    crossing = Position(gate_x, gate_y)
    barrier = frozenset(
        Position(gate_x, row)
        for row in range(height)
        if row != gate_y
    )
    gate_side_object = Position(object_x, object_y)

    return WorkshopLayout(
        width=width,
        height=height,
        walls=barrier,
        agent_start=Position(1, start_y),
        target=Position(width - 2, target_y),
        gate_kind=gate_kind,
        gate=crossing,
        max_steps=max_steps,
        key=gate_side_object if gate_kind is GateKind.DOOR else None,
        switch=gate_side_object if gate_kind is GateKind.BRIDGE else None,
    )
