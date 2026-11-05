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
    evaluate_random_legal_policy_layouts,
)
from aster.environment import Action
from aster.training.data import (
    DATA_SCHEMA_VERSION,
    collect_examples,
    split_by_layout,
    write_splits,
)
from aster.training.engine import TrainingSettings, evaluate_world_model, train_world_model
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
}


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
    controller_metrics = evaluate_controller_layouts(result.model, splits.test_layouts)
    policy_result = train_action_policy(splits.train, splits.validation, settings)
    policy_test_metrics = evaluate_policy_examples(policy_result.model, splits.test)
    learned_policy_metrics = evaluate_action_policy_layouts(
        policy_result.model,
        splits.test_layouts,
    )
    random_policy_metrics = evaluate_random_legal_policy_layouts(
        splits.test_layouts,
        seed=seed,
    )
    policy_gate_passed = (
        learned_policy_metrics["door_success_rate"] >= 0.8
        and learned_policy_metrics["bridge_success_rate"] >= 0.6
    )
    reference_controller_gate_passed = (
        controller_metrics["door_success_rate"] >= 0.8
        and controller_metrics["bridge_success_rate"] >= 0.6
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
        "history": list(result.history),
        "reference_controller_gate_passed": reference_controller_gate_passed,
        "policy_benchmark_thresholds_passed": policy_gate_passed,
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
    for name in ("seed_count", "max_steps", "epochs", "batch_size", "hidden_size", "patience"):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(config["random_episodes_per_layout"]) < 0:
        raise ValueError("random_episodes_per_layout cannot be negative")
    return config
