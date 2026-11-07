import tempfile
import unittest
from pathlib import Path

from aster.case_study import (
    ENGLISH,
    MUSIC,
    PYTHON,
    lexicon_skill,
    load_lexicon,
    run_case_study,
)
from aster.language import parse_input
from aster.models import InputKind
from aster.skills import Say


class CaseStudyTests(unittest.TestCase):
    """English -> Python -> reload -> music -> reload, in miniature."""

    def test_every_teaching_line_is_a_teachable_fact(self) -> None:
        lessons = []
        for skill in (ENGLISH, PYTHON, MUSIC, lexicon_skill(load_lexicon(40))):
            lessons.extend(skill.lessons)
        for lesson in lessons:
            self.assertIsInstance(lesson, Say)
            parsed = parse_input(lesson.text)
            self.assertEqual(parsed.kind, InputKind.FACT, lesson.text)
            self.assertIsNotNone(parsed.fact, lesson.text)

    def test_english_python_music_survive_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_case_study(
                max_neurons=5000,
                lexicon_size=30,
                memory_path=Path(tmp) / "memory.json",
            )
        acc = report["accuracies"]
        self.assertGreaterEqual(acc["english"], 0.9)
        self.assertGreaterEqual(acc["python"], 0.9)
        self.assertGreaterEqual(acc["music"], 0.9)
        self.assertTrue(report["permanent"])  # strengths identical across reload
        self.assertGreaterEqual(acc["english_after_reload"], 0.9)
        self.assertGreaterEqual(acc["python_after_reload"], 0.9)
        self.assertGreaterEqual(acc["music_final"], 0.9)
        phases = {item["phase"] for item in report["phases"]}
        self.assertIn("teach english", phases)
        self.assertIn("save, unload, reload", phases)
        self.assertTrue(any("seconds" in item for item in report["phases"]))
        self.assertTrue(report["transcript"])
        self.assertTrue(report["neurons"] > 0)


if __name__ == "__main__":
    unittest.main()
