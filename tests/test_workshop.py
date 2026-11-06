import unittest

from aster.environment import (
    Action,
    GateKind,
    Position,
    Workshop,
    WorkshopLayout,
    collect_expert_episode,
    encode_observation,
    expert_action,
    make_workshop_layout,
)
from aster.training.data import layout_id


class WorkshopTests(unittest.TestCase):
    def test_seeded_layouts_are_repeatable(self) -> None:
        first = make_workshop_layout(GateKind.DOOR, seed=17)
        second = make_workshop_layout(GateKind.DOOR, seed=17)
        self.assertEqual(first, second)

    def test_door_requires_key_and_explicit_interaction(self) -> None:
        layout = _explicit_layout(GateKind.DOOR, key=Position(1, 1))
        environment = Workshop(layout)
        environment.step(Action.MOVE_EAST)
        blocked = environment.step(Action.MOVE_EAST)
        self.assertTrue(blocked.invalid)
        self.assertEqual(blocked.observation.agent, Position(layout.gate.x - 1, layout.gate.y))

        assert layout.key is not None
        to_key = _moves_to(environment.agent, layout.key)
        for action in to_key:
            environment.step(action)
        picked_up = environment.step(Action.PICK_UP)
        self.assertEqual(picked_up.event, "picked_up_key")
        self.assertTrue(picked_up.observation.has_key)

        to_door = _moves_to(environment.agent, Position(layout.gate.x - 1, layout.gate.y))
        for action in to_door:
            environment.step(action)
        opened = environment.step(Action.INTERACT)
        self.assertEqual(opened.event, "door_opened")
        self.assertTrue(opened.observation.gate_open)

    def test_switch_opens_bridge_before_crossing(self) -> None:
        layout = _explicit_layout(GateKind.BRIDGE, switch=Position(1, 1))
        environment = Workshop(layout)
        environment.step(Action.MOVE_EAST)
        blocked = environment.step(Action.MOVE_EAST)
        self.assertTrue(blocked.invalid)

        assert layout.switch is not None
        to_switch = _moves_to(environment.agent, layout.switch)
        for action in to_switch:
            environment.step(action)
        opened = environment.step(Action.INTERACT)
        self.assertEqual(opened.event, "bridge_opened")

        to_bridge = _moves_to(environment.agent, layout.gate)
        crossing = None
        for action in to_bridge:
            crossing = environment.step(action)
        self.assertIsNotNone(crossing)
        self.assertEqual(crossing.observation.agent, layout.gate)

    def test_layout_space_supports_large_held_out_samples(self) -> None:
        for gate_kind in GateKind:
            with self.subTest(gate_kind=gate_kind):
                unique = {
                    layout_id(make_workshop_layout(gate_kind, seed=seed))
                    for seed in range(300)
                }
                self.assertGreaterEqual(len(unique), 100)

    def test_expert_coverage_policy_solves_seeded_layouts(self) -> None:
        for gate_kind in GateKind:
            for seed in range(30):
                with self.subTest(gate_kind=gate_kind, seed=seed):
                    layout = make_workshop_layout(gate_kind, seed=seed)
                    episode = collect_expert_episode(layout)
                    self.assertTrue(episode[-1].done)
                    self.assertFalse(any(result.invalid for result in episode))

    def test_encoder_has_stable_shape_and_marks_global_state(self) -> None:
        layout = make_workshop_layout(GateKind.DOOR, seed=2)
        environment = Workshop(layout)
        observation = environment.observe()
        encoded = encode_observation(observation)
        expected_size = layout.width * layout.height * 6 + 4
        self.assertEqual(len(encoded), expected_size)
        self.assertEqual(encoded[-4:], (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(encoded, encode_observation(observation))

    def test_episode_ends_at_its_step_limit(self) -> None:
        layout = make_workshop_layout(GateKind.DOOR, seed=4, max_steps=1)
        result = Workshop(layout).step(Action.WAIT)
        self.assertTrue(result.done)
        self.assertEqual(result.event, "time_limit_reached")
        self.assertEqual(result.reward, -1.0)

    def test_counterfactual_restoration_does_not_change_the_recorded_state(self) -> None:
        layout = make_workshop_layout(GateKind.BRIDGE, seed=11)
        original = Workshop(layout)
        observation = original.observe()
        restored = Workshop.from_observation(layout, observation)
        restored.step(Action.MOVE_NORTH)
        self.assertEqual(original.observe(), observation)
        self.assertEqual(restored.steps, 1)

    def test_restart_recovery_preserves_progress_at_every_interruption(self) -> None:
        for gate_kind in GateKind:
            for seed in range(4):
                with self.subTest(gate_kind=gate_kind, seed=seed):
                    layout = make_workshop_layout(gate_kind, seed=seed)
                    reference = Workshop(layout)
                    prefix: list[Action] = []
                    while not reference.done:
                        prefix.append(expert_action(reference.observe()))
                        reference.step(prefix[-1])

                    for interruption in range(len(prefix)):
                        live = Workshop(layout)
                        for action in prefix[:interruption]:
                            live.step(action)
                        snapshot = live.observe()
                        restored = Workshop.from_observation(layout, snapshot)
                        self.assertEqual(restored.observe(), snapshot)
                        self.assertEqual(restored.has_key, snapshot.has_key)
                        self.assertEqual(restored.gate_open, snapshot.gate_open)
                        self.assertEqual(restored.steps, snapshot.steps)

                        while not restored.done:
                            restored.step(expert_action(restored.observe()))
                        self.assertEqual(
                            restored.agent,
                            reference.agent,
                            "restart changed the outcome",
                        )
                        self.assertEqual(restored.steps, reference.steps)

    def test_rejects_too_small_generated_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "width >= 5"):
            make_workshop_layout(GateKind.DOOR, seed=1, width=4)

    def test_rejects_nonpositive_step_limits(self) -> None:
        layout = make_workshop_layout(GateKind.DOOR, seed=1, max_steps=0)
        with self.assertRaisesRegex(ValueError, "max_steps must be positive"):
            Workshop(layout)


def _explicit_layout(
    gate_kind: GateKind,
    key: Position | None = None,
    switch: Position | None = None,
) -> WorkshopLayout:
    """Canonical fixture: gate column at x=3, agent left of it, target right."""
    return WorkshopLayout(
        width=7,
        height=5,
        walls=frozenset(Position(3, row) for row in range(5) if row != 2),
        agent_start=Position(1, 2),
        target=Position(5, 2),
        gate_kind=gate_kind,
        gate=Position(3, 2),
        max_steps=64,
        key=key,
        switch=switch,
    )


def _moves_to(start: Position, target: Position) -> list[Action]:
    actions: list[Action] = []
    current = start
    while current.y != target.y:
        if current.y < target.y:
            actions.append(Action.MOVE_SOUTH)
            current = Position(current.x, current.y + 1)
        else:
            actions.append(Action.MOVE_NORTH)
            current = Position(current.x, current.y - 1)
    while current.x != target.x:
        if current.x < target.x:
            actions.append(Action.MOVE_EAST)
            current = Position(current.x + 1, current.y)
        else:
            actions.append(Action.MOVE_WEST)
            current = Position(current.x - 1, current.y)
    return actions


if __name__ == "__main__":
    unittest.main()
