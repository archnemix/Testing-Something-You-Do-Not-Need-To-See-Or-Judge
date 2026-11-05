import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from aster.training.data import collect_examples, split_by_layout

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from aster.cognition.evaluation import evaluate_controller
    from aster.cognition.policy_evaluation import evaluate_reactive_policy_layouts
    from aster.cognition.world_model import TransitionModel
    from aster.environment import Action, GateKind, Workshop, make_workshop_layout
    from aster.environment.baselines import choose_reactive_action
    from aster.training.baselines import evaluate_frequency_baseline
    from aster.training.cli import train_from_config
    from aster.training.engine import (
        TrainingSettings,
        evaluate_rollout_error,
        evaluate_world_model,
        train_world_model,
    )


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "install the optional training extra to test the neural model",
)
class WorldModelTests(unittest.TestCase):
    def test_hand_built_reference_controller_does_not_require_a_trained_model(self) -> None:
        torch.manual_seed(41)
        untrained_model = TransitionModel(feature_size=214, hidden_size=16)
        metrics = evaluate_controller(untrained_model, seeds=(0, 1, 2))
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["invalid_actions"], 0.0)

    def test_model_trains_from_random_initialization_and_evaluates(self) -> None:
        examples = collect_examples(range(6), random_episodes_per_layout=0, max_steps=32)
        splits = split_by_layout(examples, seed=3)
        result = train_world_model(
            splits.train,
            splits.validation,
            TrainingSettings(
                epochs=3,
                batch_size=32,
                hidden_size=16,
                patience=2,
                seed=3,
            ),
        )
        metrics = evaluate_world_model(result.model, splits.test)
        self.assertGreater(result.best_validation_loss, 0.0)
        self.assertGreaterEqual(metrics["agent_position_accuracy"], 0.0)
        self.assertLessEqual(metrics["agent_position_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["gate_state_accuracy"], 0.0)

    def test_frequency_baseline_is_deterministic_and_stateless(self) -> None:
        examples = collect_examples(range(4), random_episodes_per_layout=1, max_steps=24)
        splits = split_by_layout(examples, seed=1)
        first = evaluate_frequency_baseline(splits.train, splits.test)
        second = evaluate_frequency_baseline(splits.train, splits.test)
        self.assertEqual(first, second)
        self.assertIn("dynamic_state_bit_accuracy", first)
        self.assertIn("reward_mae", first)
        self.assertGreaterEqual(first["invalid_action_accuracy"], 0.0)

    def test_rollout_error_reports_free_running_horizons(self) -> None:
        examples = collect_examples(range(6), random_episodes_per_layout=1, max_steps=32)
        splits = split_by_layout(examples, seed=3)
        result = train_world_model(
            splits.train,
            splits.validation,
            TrainingSettings(
                epochs=2,
                batch_size=32,
                hidden_size=16,
                patience=2,
                seed=3,
            ),
        )
        metrics = evaluate_rollout_error(result.model, splits.test, horizons=(1, 3))
        self.assertIn("rollout_mae_1", metrics)
        self.assertIn("rollout_dynamic_bit_accuracy_1", metrics)
        self.assertIn("rollout_agent_position_accuracy_1", metrics)
        self.assertGreater(metrics["rollout_count_1"], 0.0)
        self.assertGreaterEqual(metrics["rollout_mae_1"], 0.0)

    def test_reactive_baseline_never_uses_state_changing_actions(self) -> None:
        for gate in GateKind:
            for seed in range(4):
                environment = Workshop(make_workshop_layout(gate, seed=seed, max_steps=24))
                while not environment.done:
                    action = choose_reactive_action(environment.observe())
                    self.assertNotIn(action, (Action.PICK_UP, Action.INTERACT))
                    environment.step(action)
                self.assertNotEqual(environment.agent, environment.layout.target)
        metrics = evaluate_reactive_policy_layouts(
            tuple((seed, gate.value) for gate in GateKind for seed in range(3)),
            max_steps=24,
        )
        self.assertEqual(metrics["success_rate"], 0.0)

    def test_configured_run_saves_model_data_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "seed": 2,
                        "seed_count": 6,
                        "random_episodes_per_layout": 0,
                        "max_steps": 32,
                        "epochs": 2,
                        "batch_size": 32,
                        "hidden_size": 16,
                        "patience": 2,
                        "output_dir": str(Path(directory) / "runs"),
                        "evaluation_seed_start": 20,
                        "evaluation_seed_count": 12,
                    }
                ),
                encoding="utf-8",
            )
            run_dir = train_from_config(config_path)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "policy.pt").is_file())
            self.assertTrue((run_dir / "data" / "manifest.json").is_file())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["reference_controller_gate_passed"])
            self.assertIn("does not measure learned planning", metrics["controller_description"])
            self.assertEqual(metrics["closed_loop_metrics"]["success_rate"], 1.0)
            self.assertEqual(metrics["closed_loop_metrics"]["invalid_actions"], 0.0)
            self.assertIn("agent_position_accuracy", metrics["test_metrics"])
            self.assertIn("expert_action_accuracy", metrics["policy_test_metrics"])
            self.assertIn("success_rate", metrics["learned_policy_metrics"])
            self.assertIn("success_rate", metrics["random_legal_policy_metrics"])
            self.assertIn("dynamic_state_bit_accuracy", metrics["frequency_baseline_metrics"])
            self.assertIn("frequency_baseline_gate_passed", metrics)
            self.assertIn("rollout_mae_1", metrics["rollout_error_metrics"])
            self.assertEqual(metrics["reactive_policy_metrics"]["success_rate"], 0.0)
            large_sample = metrics["large_sample_metrics"]
            self.assertGreater(large_sample["tasks"], 0.0)
            self.assertIn("oracle_metrics", large_sample)
            self.assertIn("learned_policy_metrics", large_sample)
            self.assertIn("oracle_gap_to_learned", large_sample)
            self.assertFalse(metrics["policy_gate_sample_sufficient"])


if __name__ == "__main__":
    unittest.main()
