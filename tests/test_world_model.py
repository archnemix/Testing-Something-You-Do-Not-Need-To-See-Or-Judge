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
    from aster.cognition.policy import ActionPolicy
    from aster.cognition.policy_evaluation import evaluate_reactive_policy_layouts
    from aster.cognition.world_model import TransitionModel
    from aster.environment import Action, GateKind, Workshop, expert_action, make_workshop_layout
    from aster.environment.baselines import choose_reactive_action
    from aster.training.baselines import evaluate_frequency_baseline
    from aster.training.cli import train_from_config
    from aster.training.ensemble import evaluate_uncertainty, train_uncertainty_ensemble
    from aster.training.policy import collect_aggregation_examples
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
        self.assertEqual(metrics["failure_key_never_collected"], 3.0)
        self.assertEqual(metrics["failure_gate_never_opened"], 3.0)
        self.assertEqual(metrics["failure_opened_never_crossed"], 0.0)

    def test_expert_action_solves_from_reachable_states(self) -> None:
        for gate in GateKind:
            for seed in range(4):
                layout = make_workshop_layout(gate, seed=seed)
                environment = Workshop(layout)
                for _ in range(3):
                    if environment.done:
                        break
                    environment.step(expert_action(environment.observe()))
                while not environment.done:
                    environment.step(expert_action(environment.observe()))
                self.assertEqual(environment.agent, environment.layout.target)

    def test_aggregation_labels_policy_visited_states(self) -> None:
        policy = ActionPolicy(feature_size=214, hidden_size=16)
        cases = tuple((seed, gate.value) for gate in GateKind for seed in range(2))
        rows = collect_aggregation_examples(policy, cases, max_steps=24)
        self.assertTrue(rows)
        self.assertTrue(all(row.episode_kind == "expert-aggregation" for row in rows))
        self.assertTrue(all(0 <= row.action < len(Action) for row in rows))
        self.assertEqual(len({(row.state, row.action) for row in rows}), len(rows))

    def test_uncertainty_ensemble_scores_familiar_and_unfamiliar_sets(self) -> None:
        examples = collect_examples(range(6), random_episodes_per_layout=1, max_steps=24)
        splits = split_by_layout(examples, seed=3)
        settings = TrainingSettings(
            epochs=2,
            batch_size=32,
            hidden_size=16,
            patience=2,
            seed=3,
        )
        base = train_world_model(splits.train, splits.validation, settings)
        members = train_uncertainty_ensemble(
            base.model,
            splits.train,
            splits.validation,
            settings,
            2,
        )
        self.assertEqual(len(members), 2)
        metrics = evaluate_uncertainty(members, splits.train, splits.test)
        for name in (
            "familiar_mean_uncertainty",
            "unfamiliar_mean_uncertainty",
            "uncertainty_shift_ratio",
            "error_auroc",
            "error_auroc_defined",
            "confidently_wrong_rate",
        ):
            self.assertIn(name, metrics)
        self.assertGreaterEqual(metrics["familiar_mean_uncertainty"], 0.0)
        self.assertGreater(metrics["uncertainty_shift_ratio"], 0.0)
        self.assertGreaterEqual(metrics["error_auroc"], 0.0)
        self.assertLessEqual(metrics["error_auroc"], 1.0)
        self.assertGreaterEqual(metrics["confidently_wrong_rate"], 0.0)
        self.assertLessEqual(metrics["confidently_wrong_rate"], 1.0)

    def test_error_flags_use_logits_when_probabilities_saturate(self) -> None:
        from aster.training.ensemble import _error_flags

        feature_size = 214
        true_agent_cell = 5
        wrong_tie_cell = 3
        target = torch.zeros(1, feature_size)
        target[0, true_agent_cell * 6 + 1] = 1.0
        mean_logits = torch.zeros(1, feature_size)
        mean_logits[0, true_agent_cell * 6 + 1] = 30.0
        mean_logits[0, wrong_tie_cell * 6 + 1] = 20.0
        mean_prediction = torch.sigmoid(mean_logits)
        self.assertEqual(
            mean_prediction[0, true_agent_cell * 6 + 1].item(),
            mean_prediction[0, wrong_tie_cell * 6 + 1].item(),
            "both confident cells must saturate to the same probability",
        )
        errors = _error_flags(mean_prediction, mean_logits, target)
        self.assertFalse(bool(errors[0]))

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
                        "policy_aggregation_rounds": 1,
                        "uncertainty_ensemble_size": 2,
                    }
                ),
                encoding="utf-8",
            )
            run_dir = train_from_config(config_path, seed_override=5)
            self.assertTrue(run_dir.name.endswith("-5"))
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
            learned = metrics["learned_policy_metrics"]
            failure_total = (
                learned["successes"]
                + learned["failure_key_never_collected"]
                + learned["failure_key_no_open"]
                + learned["failure_gate_never_opened"]
                + learned["failure_opened_never_crossed"]
                + learned["failure_crossed_never_reached"]
            )
            self.assertEqual(failure_total, learned["tasks"])
            self.assertIn("failure_breakdown_description", metrics)
            self.assertEqual(metrics["policy_aggregation_rounds"], 1.0)
            self.assertEqual(len(metrics["policy_aggregation_examples_per_round"]), 1)
            self.assertIn("Dataset aggregation: 1", metrics["behavior_cloning_description"])
            self.assertIn("AUROC", metrics["uncertainty_gate_rule"])
            self.assertIn("confidently_wrong_rate", metrics["uncertainty_metrics"])
            self.assertIn("uncertainty_gate_passed", metrics)
            large_sample = metrics["large_sample_metrics"]
            self.assertGreater(large_sample["tasks"], 0.0)
            self.assertIn("oracle_metrics", large_sample)
            self.assertIn("learned_policy_metrics", large_sample)
            self.assertIn("oracle_gap_to_learned", large_sample)
            self.assertFalse(metrics["policy_gate_sample_sufficient"])


if __name__ == "__main__":
    unittest.main()
