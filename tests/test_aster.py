import tempfile
import unittest
from pathlib import Path

from aster.agent import Aster
from aster.language import parse_input
from aster.memory import MemoryStore
from aster.models import Fact, InputKind
from aster.reasoning import answers_for


class InputParsingTests(unittest.TestCase):
    def test_parses_fact_and_question(self) -> None:
        fact = parse_input("Ada has a notebook.")
        question = parse_input("what does Ada have?")
        self.assertEqual(fact.kind, InputKind.FACT)
        self.assertEqual(fact.fact, Fact("Ada", "has", "a notebook"))
        self.assertEqual((question.subject, question.relation), ("Ada", "has"))

    def test_unrecognized_text_stays_unrecognized(self) -> None:
        self.assertEqual(parse_input("Tell me a story").kind, InputKind.OTHER)


class MemoryTests(unittest.TestCase):
    def test_deduplicates_without_case_sensitivity(self) -> None:
        memory = MemoryStore()
        self.assertTrue(memory.remember_fact(Fact("Ada", "is", "a mathematician")))
        self.assertFalse(memory.remember_fact(Fact("ada", "IS", "A MATHEMATICIAN")))
        self.assertEqual(len(memory.facts), 1)

    def test_round_trips_persistent_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = MemoryStore(path)
            memory.remember_fact(Fact("Ada", "is", "a mathematician"))
            memory.set_goal("learn about geometry")

            restored = MemoryStore(path)
            self.assertEqual(restored.facts, memory.facts)
            self.assertEqual(restored.goals, memory.goals)

    def test_restart_recovers_goal_history_and_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            agent = Aster(memory_path=path)
            agent.respond("Ada is a mathematician.")
            agent.respond("goal: learn about geometry")
            agent.respond("what is Ada?")
            agent.respond("what is Grace?")

            restarted = Aster(memory_path=path)
            self.assertEqual(
                restarted.respond("what is Ada?"),
                "Ada is a mathematician.",
            )
            status = restarted.respond("status")
            self.assertIn("learn about geometry", status)
            history = [episode.text for episode in restarted.memory.episodes]
            self.assertIn("learn about geometry", history)
            self.assertIn("what is Grace?", history)
            kinds = [episode.kind for episode in restarted.memory.episodes]
            self.assertIn("fact", kinds)
            self.assertIn("goal", kinds)

    def test_working_memory_is_bounded(self) -> None:
        memory = MemoryStore(working_capacity=2)
        memory.record_input("one")
        memory.record_input("two")
        memory.record_input("three")
        self.assertEqual([episode.text for episode in memory.episodes], ["two", "three"])

    def test_rejects_a_memory_file_with_the_wrong_root_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                MemoryStore(path)


class ReasoningTests(unittest.TestCase):
    def test_transitive_is_facts_are_inferred_without_cycles(self) -> None:
        memory = MemoryStore()
        memory.remember_fact(Fact("Ada", "is", "a mathematician"))
        memory.remember_fact(Fact("a mathematician", "is", "a person"))
        memory.remember_fact(Fact("a person", "is", "Ada"))
        self.assertEqual(
            answers_for("Ada", "is", memory),
            ["a mathematician", "a person"],
        )

    def test_agent_learns_queries_and_reports_unknowns(self) -> None:
        agent = Aster()
        self.assertIn("Remembered", agent.respond("Ada is a mathematician."))
        self.assertEqual(agent.respond("what is Ada?"), "Ada is a mathematician.")
        self.assertIn("don't have a stored answer", agent.respond("what is Grace?"))

    def test_goal_is_persistent_state_not_emotion(self) -> None:
        agent = Aster()
        agent.respond("goal: understand arithmetic")
        self.assertIn("Active goal: understand arithmetic", agent.respond("status"))


if __name__ == "__main__":
    unittest.main()
