"""Command-line entry point for the first from-scratch world-model run."""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from aster.cognition.evaluation import evaluate_controller_layouts
from aster.cognition.policy import save_policy_checkpoint
from aster.cognition.policy_evaluation import (
    evaluate_action_policy_layouts,
    evaluate_oracle_policy_layouts,
    evaluate_random_legal_policy_layouts,
    evaluate_reactive_policy_layouts,
)
from aster.environment import Action
from aster.training.baselines import evaluate_frequency_baseline
from aster.training.data import (
    DATA_SCHEMA_VERSION,
    EvaluationPool,
    collect_examples,
    split_by_layout,
    unseen_layout_cases,
    write_splits,
)
from aster.training.engine import (
    TrainingSettings,
    evaluate_rollout_error,
    evaluate_world_model,
    train_world_model,
)
from aster.training.policy import evaluate_policy_examples, train_action_policy


_DEFAULT_CONFIG = {
    "seed": 17,
    "seed_count": 60,
    "random_episodes_per_layout": 2,
    "max_steps": 64,
    "train_fraction": 0.7,
    "validation_fraction": 0.15,
    "epochs": 40,
    "batch_size": 128,
    "hidden_size": 128,
    "learning_rate": 0.001,
    "patience": 6,
    "device": "cpu",
    "output_dir": "runs",
    "evaluation_seed_start": 100,
    "evaluation_seed_count": 200,
}

_FREQUENCY_GATE_ACCURACIES = (
    "dynamic_state_bit_accuracy",
    "agent_position_accuracy",
    "gate_state_accuracy",
    "key_state_accuracy",
    "done_accuracy",
)
_FREQUENCY_GATE_MAES = (
    "time_remaining_mae",
    "invalid_action_mae",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aster's Workshop world model from scratch.")
    parser.add_argument("--config", type=Path, required=True, help="JSON training configuration")
    arguments = parser.parse_args()
    run_dir = train_from_config(arguments.config)
    print(f"Training run saved to {run_dir}")


def train_from_config(config_path: Path) -> Path:
    config = _read_config(config_path)
    seed_count = int(config["seed_count"])
    if seed_count < 3:
        raise ValueError("seed_count must be at least three for train/validation/test splits")

    seed = int(config["seed"])
    max_steps = int(config["max_steps"])
    examples = collect_examples(
        seeds=range(seed_count),
        random_episodes_per_layout=int(config["random_episodes_per_layout"]),
        max_steps=max_steps,
    )
    splits = split_by_layout(
        examples,
        seed=seed,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = Path(config["output_dir"]) / f"workshop-{timestamp}-{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = {
        **config,
        "config_path": str(config_path.resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "environment_version": 1,
        "data_schema_version": DATA_SCHEMA_VERSION,
    }
    (run_dir / "config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n",
        encoding="utf-8",
    )
    write_splits(splits, run_dir / "data")

    settings = TrainingSettings(
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        hidden_size=int(config["hidden_size"]),
        learning_rate=float(config["learning_rate"]),
        patience=int(config["patience"]),
        seed=seed,
        device=str(config["device"]),
    )
    result = train_world_model(splits.train, splits.validation, settings)
    test_metrics = evaluate_world_model(result.model, splits.test)
    frequency_baseline_metrics = evaluate_frequency_baseline(splits.train, splits.test)
    rollout_metrics = evaluate_rollout_error(result.model, splits.test)
    controller_metrics = evaluate_controller_layouts(
        result.model,
        splits.test_layouts,
        max_steps=max_steps,
    )
    policy_result = train_action_policy(splits.train, splits.validation, settings)
    policy_test_metrics = evaluate_policy_examples(policy_result.model, splits.test)
    learned_policy_metrics = evaluate_action_policy_layouts(
        policy_result.model,
        splits.test_layouts,
        max_steps=max_steps,
    )
    random_policy_metrics = evaluate_random_legal_policy_layouts(
        splits.test_layouts,
        seed=seed,
        max_steps=max_steps,
    )
    reactive_policy_metrics = evaluate_reactive_policy_layouts(
        splits.test_layouts,
        max_steps=max_steps,
    )
    large_sample_metrics = _evaluate_large_sample(
        config,
        splits,
        result.model,
        policy_result.model,
        max_steps,
    )

    frequency_baseline_gate_passed = all(
        test_metrics[name] > frequency_baseline_metrics[name]
        for name in _FREQUENCY_GATE_ACCURACIES
    ) and all(
        test_metrics[name] <= frequency_baseline_metrics[name]
        for name in _FREQUENCY_GATE_MAES
    ) and test_metrics["invalid_action_accuracy"] >= (
        frequency_baseline_metrics["invalid_action_accuracy"]
    )
    reference_controller_gate_passed = (
        controller_metrics["door_success_rate"] >= 0.8
        and controller_metrics["bridge_success_rate"] >= 0.6
    )
    large_learned = large_sample_metrics["learned_policy_metrics"]
    policy_gate_sample_sufficient = (
        large_sample_metrics["door_tasks"] >= 100
        and large_sample_metrics["bridge_tasks"] >= 100
    )
    policy_gate_passed = (
        policy_gate_sample_sufficient
        and large_learned["door_success_rate"] >= 0.8
        and large_learned["bridge_success_rate"] >= 0.6
    )
    learned_policy_beats_baselines = (
        large_sample_metrics["tasks"] > 0
        and large_learned["success_rate"]
        > large_sample_metrics["random_legal_policy_metrics"]["success_rate"]
        and large_learned["success_rate"]
        > large_sample_metrics["reactive_policy_metrics"]["success_rate"]
    )
    oracle_rate = large_sample_metrics["oracle_metrics"]["success_rate"]
    large_sample_metrics["oracle_gap_to_learned"] = (
        oracle_rate - large_learned["success_rate"]
    )

    checkpoint = {
        "format_version": 2,
        "model_state": result.model.state_dict(),
        "feature_size": result.model.feature_size,
        "hidden_size": settings.hidden_size,
        "action_names": [action.value for action in Action],
        "training_seed": seed,
        "environment_version": 1,
    }
    torch.save(checkpoint, run_dir / "best.pt")
    save_policy_checkpoint(policy_result.model, run_dir / "policy.pt", seed)

    report: dict[str, Any] = {
        "best_validation_loss": result.best_validation_loss,
        "epochs_completed": len(result.history),
        "train_examples": len(splits.train),
        "validation_examples": len(splits.validation),
        "test_examples": len(splits.test),
        "test_metrics": test_metrics,
        "frequency_baseline_description": (
            "Training-frequency baseline: per-action modal next-state fields and mean "
            "outcomes, ignoring the current state. The model must beat it on held-out "
            "transitions to show state-dependent dynamics learning."
        ),
        "frequency_baseline_metrics": frequency_baseline_metrics,
        "frequency_baseline_rule": (
            "Gate passes when the model is strictly better on dynamic_state_bit_accuracy, "
            "agent_position_accuracy, gate_state_accuracy, key_state_accuracy, and "
            "done_accuracy, and not worse on time_remaining_mae, invalid_action_mae, or "
            "invalid_action_accuracy."
        ),
        "frequency_baseline_reward_note": (
            "reward_mae is reported but does not gate: predicting a constant per-action "
            "outcome is near-optimal under MAE for the Workshop's skewed reward signal, "
            "so reward_mae cannot discriminate state-dependent learning. The rule was "
            "revised after the first completed evaluation for this reason."
        ),
        "frequency_baseline_gate_passed": frequency_baseline_gate_passed,
        "rollout_error_metrics": rollout_metrics,
        "closed_loop_metrics": controller_metrics,
        "controller_description": (
            "Hybrid reference: explicit Workshop affordances and goal costs are combined "
            "with model predictions; this does not measure learned planning alone."
        ),
        "behavior_cloning_description": (
            "The separate action policy is a neural behavior-cloning baseline trained only "
            "on expert actions; execution masks illegal actions and has no goal-distance "
            "heuristic."
        ),
        "policy_test_metrics": policy_test_metrics,
        "learned_policy_metrics": learned_policy_metrics,
        "random_legal_policy_metrics": random_policy_metrics,
        "reactive_baseline_description": (
            "Reactive baseline: greedy path toward the target that ignores keys, "
            "switches, and doors; gated tasks are unsolvable for it."
        ),
        "reactive_policy_metrics": reactive_policy_metrics,
        "large_sample_metrics": large_sample_metrics,
        "learned_policy_beats_random_and_reactive": learned_policy_beats_baselines,
        "policy_benchmark_thresholds_passed": policy_gate_passed,
        "policy_gate_sample_sufficient": policy_gate_sample_sufficient,
        "reference_controller_gate_passed": reference_controller_gate_passed,
        "history": list(result.history),
        "note": (
            "This checkpoint was trained and evaluated only in Workshop. A passed "
            "gate is not evidence of general intelligence or safe real-world planning."
        ),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _evaluate_large_sample(
    config: dict[str, Any],
    splits: Any,
    model: torch.nn.Module,
    policy_model: torch.nn.Module,
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate on unique layouts that no data split has ever contained."""
    seed_start = int(config["evaluation_seed_start"])
    seed_count = int(config["evaluation_seed_count"])
    excluded = set(splits.train_layout_ids) | set(splits.validation_layout_ids)
    excluded |= set(splits.test_layout_ids)
    pool: EvaluationPool = unseen_layout_cases(
        range(seed_start, seed_start + seed_count),
        excluded,
        max_steps=max_steps,
    )
    sample: dict[str, Any] = {
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "tasks": float(len(pool.cases)),
        "door_tasks": float(sum(1 for _, gate in pool.cases if gate == "door")),
        "bridge_tasks": float(sum(1 for _, gate in pool.cases if gate == "bridge")),
        "unique_layouts": float(len(pool.layout_ids)),
        "duplicates_dropped": float(pool.duplicates_dropped),
        "excluded_dropped": float(pool.excluded_dropped),
    }
    if not pool.cases:
        sample["note"] = (
            "No unseen layouts remained outside the data splits; large-sample "
            "gates cannot be assessed in this run."
        )
        sample["oracle_metrics"] = {"success_rate": 0.0}
        sample["hybrid_controller_metrics"] = {"success_rate": 0.0}
        sample["learned_policy_metrics"] = {"success_rate": 0.0}
        sample["random_legal_policy_metrics"] = {"success_rate": 0.0}
        sample["reactive_policy_metrics"] = {"success_rate": 0.0}
        return sample

    evaluations: dict[str, Any] = {}
    for name, evaluate in (
        ("oracle_metrics", lambda: evaluate_oracle_policy_layouts(pool.cases, max_steps)),
        (
            "hybrid_controller_metrics",
            lambda: evaluate_controller_layouts(model, pool.cases, max_steps=max_steps),
        ),
        (
            "learned_policy_metrics",
            lambda: evaluate_action_policy_layouts(policy_model, pool.cases, max_steps=max_steps),
        ),
        (
            "random_legal_policy_metrics",
            lambda: evaluate_random_legal_policy_layouts(pool.cases, seed=0, max_steps=max_steps),
        ),
        (
            "reactive_policy_metrics",
            lambda: evaluate_reactive_policy_layouts(pool.cases, max_steps=max_steps),
        ),
    ):
        started = time.perf_counter()
        metrics = evaluate()
        elapsed = time.perf_counter() - started
        total_steps = metrics.get("total_steps", 0.0)
        metrics["wall_seconds"] = elapsed
        metrics["seconds_per_decision"] = (
            elapsed / total_steps if total_steps else 0.0
        )
        evaluations[name] = metrics
    sample.update(evaluations)
    return sample


def _read_config(path: Path) -> dict[str, Any]:
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read training config at {path}: {error}") from error
    if not isinstance(supplied, dict):
        raise ValueError("training config must be a JSON object")

    unknown = set(supplied) - set(_DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown training config field(s): {', '.join(sorted(unknown))}")
    config = {**_DEFAULT_CONFIG, **supplied}
    if int(config["seed"]) < 0:
        raise ValueError("seed cannot be negative")
    if int(config["evaluation_seed_start"]) < 0:
        raise ValueError("evaluation_seed_start cannot be negative")
    for name in (
        "seed_count",
        "max_steps",
        "epochs",
        "batch_size",
        "hidden_size",
        "patience",
        "evaluation_seed_count",
    ):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(config["random_episodes_per_layout"]) < 0:
        raise ValueError("random_episodes_per_layout cannot be negative")
    return config
