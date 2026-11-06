import unittest

from aster.cognition.plastic import PlasticNetwork
from aster.environment.physics import PhysicsPlayground, learn_from_trials


class PhysicsGroundingTests(unittest.TestCase):
    """Causal knowledge formed from experience, never from being told.

    The physics playground is the world model's teacher: pushing moves
    things, letting go drops them, walls stop them. The network only ever
    sees action-outcome pairings; the causal pathways form from repetition
    and can be named afterwards ("that is called gravity").
    """

    def test_falling_is_learned_from_experience_not_telling(self) -> None:
        net = PlasticNetwork()
        playground = PhysicsPlayground()
        self.assertEqual(net.retrieve("letting go", "causes"), [])
        learn_from_trials(net, playground, "letting go", 6)
        self.assertEqual(
            [label for label, _ in net.retrieve("letting go", "causes")],
            ["falling", "landing"],
        )

    def test_the_world_not_a_prior_decides_what_is_learned(self) -> None:
        open_net = PlasticNetwork()
        walled_net = PlasticNetwork()
        open_yard = PhysicsPlayground(yard_width=3, height=2, start_x=1)
        walled_yard = PhysicsPlayground(yard_width=2, height=2, start_x=1)
        learn_from_trials(open_net, open_yard, "pushing right", 5)
        learn_from_trials(walled_net, walled_yard, "pushing right", 5)
        self.assertEqual(
            [label for label, _ in open_net.retrieve("pushing right", "causes")],
            ["moving right"],
        )
        self.assertEqual(
            [label for label, _ in walled_net.retrieve("pushing right", "causes")],
            ["staying still"],
        )

    def test_discovery_precedes_naming_and_the_name_is_grounded(self) -> None:
        net = PlasticNetwork()
        playground = PhysicsPlayground()
        learn_from_trials(net, playground, "letting go", 4)
        # before anyone names it, the word "gravity" means nothing here
        self.assertEqual(net.retrieve("gravity", "means"), [])
        # hearing "that is called gravity" binds the word to the experienced
        # concept -- the same pathway the experience already built
        net.observe("gravity", "means", "falling")
        named = [label for label, _ in net.retrieve("gravity", "means")]
        experienced = [label for label, _ in net.retrieve("letting go", "causes")]
        self.assertEqual(named, ["falling"])
        self.assertEqual(named[0], experienced[0])
        self.assertGreater(net.path_strength("gravity", "means", "falling"), 0.5)

    def test_unexperienced_actions_are_not_invented(self) -> None:
        net = PlasticNetwork()
        playground = PhysicsPlayground()
        learn_from_trials(net, playground, "letting go", 3)
        self.assertEqual(net.retrieve("shaking", "causes"), [])

    def test_skills_get_better_with_practice_over_long_engagement(self) -> None:
        net = PlasticNetwork()
        playground = PhysicsPlayground()
        learn_from_trials(net, playground, "letting go", 3)
        before = net.path_strength("letting go", "causes", "falling")
        actions = ["pushing right", "pushing left", "lifting"]
        for round_index in range(200):
            learn_from_trials(net, playground, actions[round_index % 3], 1)
            if round_index % 10 == 0:
                learn_from_trials(net, playground, "letting go", 1)
        after = net.path_strength("letting go", "causes", "falling")
        self.assertGreaterEqual(after, before)
        self.assertEqual(
            [label for label, _ in net.retrieve("letting go", "causes")],
            ["falling", "landing"],
        )


if __name__ == "__main__":
    unittest.main()
