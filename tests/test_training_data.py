import json
import tempfile
import unittest
from pathlib import Path

from aster.training.data import collect_examples, split_by_layout, write_splits


class TrainingDataTests(unittest.TestCase):
    def test_collection_is_reproducible_and_has_both_gate_families(self) -> None:
        first = collect_examples(range(4), random_episodes_per_layout=1, max_steps=20)
        second = collect_examples(range(4), random_episodes_per_layout=1, max_steps=20)
        self.assertEqual(first, second)
        self.assertEqual({row.gate_kind for row in first}, {"door", "bridge"})
        self.assertTrue(any(row.episode_kind == "expert" for row in first))
        self.assertTrue(any(row.episode_kind == "coverage" for row in first))
        self.assertTrue(any(row.episode_kind.startswith("random") for row in first))
        self.assertTrue(all(len(row.state) == len(row.next_state) for row in first))

    def test_splits_keep_duplicate_layouts_together(self) -> None:
        examples = collect_examples(range(10), random_episodes_per_layout=0)
        splits = split_by_layout(examples, seed=9)
        train = set(splits.train_layout_ids)
        validation = set(splits.validation_layout_ids)
        test = set(splits.test_layout_ids)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(
            train | validation | test,
            {row.layout_id for row in examples},
        )
        for rows, layout_ids in (
            (splits.train, train),
            (splits.validation, validation),
            (splits.test, test),
        ):
            self.assertEqual({row.layout_id for row in rows}, layout_ids)

    def test_writes_readable_splits_and_manifest(self) -> None:
        splits = split_by_layout(collect_examples(range(4), random_episodes_per_layout=0))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "data"
            write_splits(splits, output_dir)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["environment_version"], 1)
            self.assertEqual(manifest["data_schema_version"], 2)
            self.assertEqual(manifest["counts"]["train"], len(splits.train))
            self.assertEqual(len(manifest["split_sha256"]["train"]), 64)
            self.assertTrue((output_dir / "test.jsonl").read_text(encoding="utf-8").strip())

    def test_requires_three_distinct_layouts(self) -> None:
        examples = collect_examples(range(2), random_episodes_per_layout=0)
        one_layout = tuple(row for row in examples if row.layout_id == examples[0].layout_id)
        with self.assertRaisesRegex(ValueError, "at least three"):
            split_by_layout(one_layout)


if __name__ == "__main__":
    unittest.main()
