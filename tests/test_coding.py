"""Second-level learning tests: a skill learned by doing, in an environment."""

import unittest

from aster.coding import CodingStudent, PythonLab


class EnvironmentTests(unittest.TestCase):
    def test_the_lab_runs_code_and_reports_real_errors(self) -> None:
        lab = PythonLab()
        self.assertEqual(lab.run("print('hi')"), {"ok": True, "output": "hi", "error": ""})
        broken = lab.run("print('hi'")
        self.assertFalse(broken["ok"])
        self.assertEqual(broken["error"], "SyntaxError")
        self.assertIn("syntaxerror", lab.lookup("SyntaxError").casefold())


class SecondLearningTests(unittest.TestCase):
    def test_it_learns_from_running_code_not_from_being_told(self) -> None:
        student = CodingStudent()
        transcript = student.solve("print the text welcome", "welcome")
        first_try = transcript[1]  # after the study entry
        self.assertEqual(first_try["outcome"], "wrong output")  # it tried what it knew
        self.assertEqual(transcript[-1]["outcome"], "solved")  # the loop got it there
        self.assertTrue(any(entry.get("lookup") for entry in transcript))

    def test_the_fix_lives_in_the_pathways(self) -> None:
        student = CodingStudent()
        student.solve("print the text welcome", "welcome")
        labels = [label for label, _ in student.net.successors("welcome")]
        self.assertIn("'welcome'", labels)  # the word is wired to its code form

    def test_it_re_runs_to_verify_the_fix(self) -> None:
        student = CodingStudent()
        transcript = student.solve("print the text goodbye", "goodbye")
        last = transcript[-1]
        self.assertEqual(last["outcome"], "solved")
        self.assertEqual(last["output"], "goodbye")  # verified by the environment

    def test_success_rewards_the_path_that_produced_it(self) -> None:
        student = CodingStudent()
        before = student.net.successors("print")[0][1]
        student.solve("print the text sunshine", "sunshine")
        after = student.net.successors("print")[0][1]
        self.assertGreaterEqual(after, before)  # use strengthens, never destroys

    def test_it_studies_before_writing_code(self) -> None:
        student = CodingStudent()
        transcript = student.solve("print the text welcome", "welcome")
        self.assertEqual(transcript[0]["stage"], "study")
        self.assertTrue(transcript[0]["understood"])  # looked it up first

    def test_it_rewrites_with_another_code_when_stuck(self) -> None:
        student = CodingStudent()
        # expected output the first shape cannot produce: must rewrite
        transcript = student.solve("print the text welcome", "WELCOME")
        self.assertTrue(
            any(entry["stage"] == "rewrite" for entry in transcript)
        )

    def test_new_information_lives_in_the_synapses_and_survives_reload(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        student = CodingStudent()
        student.solve("print the text welcome", "welcome")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "student.json")
            student.save(path)
            revived = CodingStudent()
            revived.load(path)
        labels = [label for label, _ in revived.net.successors("welcome")]
        self.assertIn("'welcome'", labels)  # the learning survived a restart


if __name__ == "__main__":
    unittest.main()
