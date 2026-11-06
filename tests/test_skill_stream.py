import unittest

from aster.agent import Aster
from aster.environment.physics import PhysicsPlayground
from aster.skills import Say, Skill, Stream, Trial, build_curriculum, run_protocol


class SkillStreamTests(unittest.TestCase):
    """Skills arrive through conversation and are tested through questions."""

    def test_one_skill_taught_by_talking_is_learned_and_testable(self) -> None:
        stream = Stream()
        kitchen = next(s for s in build_curriculum() if s.name == "kitchen")
        stream.teach(kitchen)
        result = stream.quiz(kitchen)
        self.assertEqual(result["accuracy"], 1.0)

    def test_second_skill_talked_in_mid_conversation_lands_and_first_survives(
        self,
    ) -> None:
        stream = Stream()
        words = next(s for s in build_curriculum() if s.name == "words")
        kitchen = next(s for s in build_curriculum() if s.name == "kitchen")
        stream.teach(words)
        first = stream.quiz(words)
        self.assertEqual(first["accuracy"], 1.0)
        stream.chat("hello")  # the conversation just keeps going
        stream.teach(kitchen)
        second = stream.quiz(kitchen)
        again = stream.quiz(words)
        self.assertEqual(second["accuracy"], 1.0)  # the new skill was learned
        self.assertEqual(again["accuracy"], 1.0)  # and the old one is intact

    def test_experience_and_talk_shape_one_shared_brain(self) -> None:
        stream = Stream(agent=Aster(), playground=PhysicsPlayground())
        yard = next(s for s in build_curriculum() if s.name == "yard")
        kitchen = next(s for s in build_curriculum() if s.name == "kitchen")
        stream.teach(yard)
        stream.teach(kitchen)
        # what the world showed it ...
        self.assertIn("falling", stream.agent.respond("what does letting go cause?"))
        # ... and what conversation named, answered by one and the same network
        self.assertIn("falling", stream.agent.respond("what does gravity mean?"))
        self.assertIn("metal", stream.agent.respond("what is the pot?"))
        self.assertEqual(stream.quiz(yard)["accuracy"], 1.0)
        self.assertEqual(stream.quiz(kitchen)["accuracy"], 1.0)

    def test_protocol_produces_an_accuracy_matrix(self) -> None:
        report = run_protocol(max_neurons=3000)
        matrix = report["matrix"]
        self.assertEqual(sorted(matrix), ["chores", "kitchen", "places", "trades", "words", "yard"])
        # every skill answers its own questions right after being taught
        for skill, checkpoints in matrix.items():
            self.assertGreaterEqual(checkpoints[skill], 0.85, skill)

    def test_a_fallen_skill_relearns_quickly_in_a_tight_brain(self) -> None:
        from aster.skills import run_reteach

        report = run_reteach(max_neurons=400)
        fallen = report["reteached"]
        before = report["before_reteach"][fallen]
        after = report["after_reteach"][fallen]
        self.assertLess(before, 1.0)  # the tight brain really did lose something
        self.assertGreaterEqual(after, before)  # one re-showing brings it back
        self.assertGreaterEqual(after, 0.85)


if __name__ == "__main__":
    unittest.main()
