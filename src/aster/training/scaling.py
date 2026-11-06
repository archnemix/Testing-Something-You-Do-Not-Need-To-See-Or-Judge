"""Architecture scaling study: the project MLP vs a Transformer baseline.

Both families are trained on identical Workshop data, splits, optimizer,
and stopping rules at three parameter scales (tiny, about 5M, about 10M),
then scored with the same one-step metrics and rollout error. This tests
the architecture, not the brain-inspired claim: the Transformer is a
yardstick.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from aster.cognition.transformer import TransformerTransitionModel
from aster.cognition.world_model import TransitionModel, examples_to_tensors
from aster.training.data import collect_examples, split_by_layout
from aster.training.engine import (
    TrainingSettings,
    evaluate_rollout_error,
    evaluate_world_model,
    fit_transition_model,
)

_TARGETS = {
    "tiny": 0,
    "5m": 5_000_000,
    "10m": 10_000_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated cell names to run (default: all six)",
    )
    arguments = parser.parse_args()
    selected = {name.strip() for name in arguments.only.split(",") if name.strip()}
    existing: dict[str, Any] | None = None
    if (arguments.output_dir / "report.json").exists():
        # merge mode: rerun missing cells into an existing study directory
        run_dir = arguments.output_dir
        existing = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        run_dir = arguments.output_dir / f"scaling-study-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
    if arguments.threads > 0:
        torch.set_num_threads(arguments.threads)
    report = run_study(
        run_dir,
        arguments.epochs,
        arguments.patience,
        arguments.seed,
        selected,
        existing=existing,
        threads=arguments.threads,
    )
    print(json.dumps(_summary_table(report), indent=2))
    print(f"SCALING STUDY COMPLETE -> {run_dir}")


def run_study(
    run_dir: Path,
    epochs: int,
    patience: int,
    seed: int,
    selected: set[str],
    existing: dict[str, Any] | None = None,
    threads: int = 0,
) -> dict[str, Any]:
    examples = collect_examples(
        seeds=range(60),
        random_episodes_per_layout=2,
        max_steps=64,
    )
    splits = split_by_layout(examples, seed=seed, train_fraction=0.7, validation_fraction=0.15)
    feature_size = len(splits.train[0].state)
    train_count = len(splits.train)
    validation_count = len(splits.validation)
    test_examples = splits.test
    settings = TrainingSettings(
        epochs=epochs,
        batch_size=128,
        hidden_size=128,
        learning_rate=0.001,
        patience=patience,
        seed=seed,
    )
    device = torch.device(settings.device)
    training_tensors = examples_to_tensors(splits.train, device)
    validation_tensors = examples_to_tensors(splits.validation, device)
    del examples, splits  # free the raw rows; only tensors and held-out test rows stay resident
    report: dict[str, Any] = {
        "description": (
            "Same data, splits, optimizer, and stopping rules for every cell; "
            "the Transformer is a comparison yardstick, not part of the "
            "brain-inspired claim."
        ),
        "data": {
            "seed_count": 60,
            "split_seed": seed,
            "train_examples": train_count,
            "validation_examples": validation_count,
            "test_examples": len(test_examples),
        },
        "settings": {
            "epochs": epochs,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "patience": patience,
        },
        "cells": [],
    }
    cells_by_name: dict[str, dict[str, Any]] = {}
    if existing:
        for cell in existing.get("cells", []):
            cells_by_name[cell["name"]] = cell
        report["settings"] = existing.get("settings", report["settings"])
        report["data"] = existing.get("data", report["data"])
    report.setdefault("invocations", [])
    if existing:
        report["invocations"] = list(existing.get("invocations", []))
    report["invocations"].append(
        {
            "selected": sorted(selected) if selected else "all",
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "threads": threads,
            "started_utc": time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        }
    )
    report["cells"] = list(cells_by_name.values())
    for name, build in (
        ("mlp-tiny", lambda: _mlp_for_budget(feature_size, 0)),
        ("mlp-5m", lambda: _mlp_for_budget(feature_size, _TARGETS["5m"])),
        ("mlp-10m", lambda: _mlp_for_budget(feature_size, _TARGETS["10m"])),
        ("transformer-tiny", lambda: _transformer_for_budget(feature_size, 0)),
        ("transformer-5m", lambda: _transformer_for_budget(feature_size, _TARGETS["5m"])),
        ("transformer-10m", lambda: _transformer_for_budget(feature_size, _TARGETS["10m"])),
    ):
        if selected and name not in selected:
            continue
        torch.manual_seed(seed)
        model, config = build()
        parameters = sum(p.numel() for p in model.parameters())
        print(f"[{name}] training with {parameters} parameters ...", flush=True)
        started = time.perf_counter()
        result = fit_transition_model(model, training_tensors, validation_tensors, settings)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        test_metrics = evaluate_world_model(result.model, test_examples)
        rollout = evaluate_rollout_error(result.model, test_examples)
        inference_seconds = _time_inference(result.model, test_examples)
        cell = {
            "name": name,
            "family": "mlp" if name.startswith("mlp") else "transformer",
            "parameters": parameters,
            "config": config,
            "epochs_completed": len(result.history),
            "best_validation_loss": result.best_validation_loss,
            "final_train_loss": result.history[-1]["train_loss"],
            "val_loss_by_epoch": [round(epoch["loss"], 5) for epoch in result.history],
            "test_metrics": test_metrics,
            "rollout_error_metrics": rollout,
            "train_wall_seconds": train_seconds,
            "inference_seconds_per_example": inference_seconds,
        }
        cells_by_name[name] = cell
        report["cells"] = list(cells_by_name.values())
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{name}] val_loss={result.best_validation_loss:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"agent_acc={test_metrics['agent_position_accuracy']:.4f} "
            f"({train_seconds:.0f}s)",
            flush=True,
        )
        del model, result
        gc.collect()
    return report


def _mlp_for_budget(feature_size: int, target: int) -> tuple[TransitionModel, dict[str, int]]:
    """Width-scaled project MLP whose parameter count is nearest the target."""
    if target <= 0:
        return TransitionModel(feature_size, 128), {"hidden_size": 128, "hidden_layers": 2}
    hidden = int((-440 + (440**2 + 4 * (target - 217)) ** 0.5) / 2)
    candidates = range(max(8, hidden - 4), hidden + 5)
    best = min(
        candidates,
        key=lambda width: abs(_mlp_parameters(feature_size, width) - target),
    )
    model = TransitionModel(feature_size, best)
    return model, {"hidden_size": best, "hidden_layers": 2}


def _mlp_parameters(feature_size: int, hidden: int) -> int:
    return hidden * hidden + (feature_size + 214 + 3 + 4) * hidden + 214 + 3


def _transformer_for_budget(
    feature_size: int,
    target: int,
) -> tuple[TransformerTransitionModel, dict[str, int]]:
    """Nearest cell-token Transformer to the target over a small config grid."""
    if target <= 0:
        model = TransformerTransitionModel(feature_size, d_model=64, layers=2, heads=2)
        return model, {"d_model": 64, "layers": 2, "heads": 2}

    grid: set[tuple[int, int, int]] = set()
    for layers in (1, 2, 4, 6, 8):
        width = int((target / (12 * layers)) ** 0.5)
        for d_model in {max(32, width - 32), width, width + 32}:
            d_model -= d_model % 32
            for heads in (2, 4, 8):
                if d_model % heads == 0:
                    grid.add((layers, max(d_model, 32), heads))

    best_model: TransformerTransitionModel | None = None
    best_config: tuple[int, int, int] | None = None
    best_distance = float("inf")
    for layers, d_model, heads in grid:
        model = TransformerTransitionModel(
            feature_size,
            d_model=d_model,
            layers=layers,
            heads=heads,
        )
        distance = abs(sum(p.numel() for p in model.parameters()) - target)
        if distance < best_distance:
            best_distance = distance
            best_model = model
            best_config = (layers, d_model, heads)
    assert best_model is not None and best_config is not None
    layers, d_model, heads = best_config
    return best_model, {"d_model": d_model, "layers": layers, "heads": heads}


def _time_inference(model: torch.nn.Module, examples: Any) -> float:
    """Seconds per example for batched forward passes over the held-out states."""
    device = next(model.parameters()).device
    states = torch.tensor([row.state for row in examples], dtype=torch.float32, device=device)
    actions = torch.tensor([row.action for row in examples], dtype=torch.long, device=device)
    batch_size = 512
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(states[:2], actions[:2])
            started = time.perf_counter()
            for start in range(0, len(examples), batch_size):
                end = min(start + batch_size, len(examples))
                model(states[start:end], actions[start:end])
            elapsed = time.perf_counter() - started
    finally:
        model.train(was_training)
    return elapsed / len(examples)


def _summary_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": cell["name"],
            "parameters": cell["parameters"],
            "epochs_completed": cell["epochs_completed"],
            "best_validation_loss": round(cell["best_validation_loss"], 4),
            "test_loss": round(cell["test_metrics"]["loss"], 4),
            "agent_position_accuracy": round(
                cell["test_metrics"]["agent_position_accuracy"], 4
            ),
            "rollout_mae_10": round(cell["rollout_error_metrics"]["rollout_mae_10"], 4),
            "train_wall_seconds": round(cell["train_wall_seconds"], 1),
        }
        for cell in report["cells"]
    ]


if __name__ == "__main__":
    main()
