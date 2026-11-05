"""Reproducible supervised fitting and held-out evaluation for world dynamics."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from aster.cognition.world_model import (
    TransitionModel,
    examples_to_tensors,
    transition_loss,
)
from aster.training.data import TransitionExample
from aster.training.metrics import (
    agent_position_indices,
    dynamic_state_indices,
    field_metrics,
)


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 40
    batch_size: int = 128
    hidden_size: int = 128
    learning_rate: float = 0.001
    patience: int = 6
    seed: int = 17
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: TransitionModel
    history: tuple[dict[str, float], ...]
    best_validation_loss: float


def train_world_model(
    training_examples: tuple[TransitionExample, ...],
    validation_examples: tuple[TransitionExample, ...],
    settings: TrainingSettings,
) -> TrainingResult:
    _validate_settings(settings)
    if not training_examples or not validation_examples:
        raise ValueError("training and validation splits must both contain examples")

    device = torch.device(settings.device)
    torch.manual_seed(settings.seed)
    feature_size = len(training_examples[0].state)
    if any(len(row.state) != feature_size for row in training_examples + validation_examples):
        raise ValueError("all observations must have the same feature size")

    training = examples_to_tensors(training_examples, device)
    validation = examples_to_tensors(validation_examples, device)
    model = TransitionModel(feature_size, settings.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    random_generator = torch.Generator(device=device).manual_seed(settings.seed)

    best_loss = float("inf")
    best_weights: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(training[0]), generator=random_generator, device=device)
        total_loss = 0.0
        for indices in order.split(settings.batch_size):
            batch = tuple(tensor[indices] for tensor in training)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1])
            loss = transition_loss(
                prediction,
                batch[2],
                batch[3],
                batch[4],
                batch[5],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)

        train_loss = total_loss / len(training_examples)
        validation_metrics = evaluate_world_model(model, validation_examples, device=device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, **validation_metrics})

        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_weights = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                break

    if best_weights is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_weights)
    return TrainingResult(model, tuple(history), best_loss)


def evaluate_world_model(
    model: TransitionModel,
    examples: tuple[TransitionExample, ...],
    device: torch.device | None = None,
    batch_size: int = 512,
) -> dict[str, float]:
    if not examples:
        raise ValueError("cannot evaluate an empty dataset split")
    evaluation_device = device or next(model.parameters()).device
    tensors = examples_to_tensors(examples, evaluation_device)
    was_training = model.training
    model.eval()

    weighted_loss = 0.0
    predicted_states: list[Tensor] = []
    predicted_rewards: list[Tensor] = []
    predicted_dones: list[Tensor] = []
    predicted_invalids: list[Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                end = min(start + batch_size, len(examples))
                batch = tuple(tensor[start:end] for tensor in tensors)
                state_logits, reward_prediction, done_logits, invalid_logits = model(
                    batch[0],
                    batch[1],
                )
                loss = transition_loss(
                    (state_logits, reward_prediction, done_logits, invalid_logits),
                    batch[2],
                    batch[3],
                    batch[4],
                    batch[5],
                )
                weighted_loss += float(loss) * (end - start)
                predicted_states.append(_decode_state_logits(state_logits))
                predicted_rewards.append(torch.tanh(reward_prediction))
                predicted_dones.append(torch.sigmoid(done_logits))
                predicted_invalids.append(torch.sigmoid(invalid_logits))
    finally:
        model.train(was_training)

    metrics = field_metrics(
        torch.cat(predicted_states),
        torch.cat(predicted_rewards),
        torch.cat(predicted_dones),
        torch.cat(predicted_invalids),
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
    )
    return {"loss": weighted_loss / len(examples), **metrics}


def evaluate_rollout_error(
    model: TransitionModel,
    examples: tuple[TransitionExample, ...],
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    batch_size: int = 512,
) -> dict[str, float]:
    """Free-running rollout error: feed predictions back for h recorded actions.

    Trajectories are maximal runs of transitions where each recorded next
    state is exactly the following transition's start state. Horizons with
    no covering trajectory are omitted and counted as zero.
    """
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be positive")
    trajectories = _chained_trajectories(examples)
    if not trajectories:
        raise ValueError("no chained trajectories available for rollout")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    try:
        with torch.no_grad():
            for horizon in sorted(set(horizons)):
                jobs: list[tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...]]] = []
                for trajectory in trajectories:
                    for index in range(0, len(trajectory) - horizon + 1):
                        actions = tuple(
                            trajectory[index + offset].action
                            for offset in range(horizon)
                        )
                        jobs.append(
                            (
                                trajectory[index].state,
                                actions,
                                trajectory[index + horizon - 1].next_state,
                            )
                        )
                if not jobs:
                    metrics[f"rollout_mae_{horizon}"] = 0.0
                    metrics[f"rollout_dynamic_bit_accuracy_{horizon}"] = 0.0
                    metrics[f"rollout_agent_position_accuracy_{horizon}"] = 0.0
                    metrics[f"rollout_count_{horizon}"] = 0.0
                    continue

                absolute_error = 0.0
                bit_correct = 0
                bit_count = 0
                agent_correct = 0
                for start in range(0, len(jobs), batch_size):
                    batch = jobs[start : start + batch_size]
                    states = torch.tensor(
                        [item[0] for item in batch],
                        dtype=torch.float32,
                        device=device,
                    )
                    actions = torch.tensor(
                        [item[1] for item in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    targets = torch.tensor(
                        [item[2] for item in batch],
                        dtype=torch.float32,
                        device=device,
                    )
                    for offset in range(horizon):
                        state_logits, _, _, _ = model(states, actions[:, offset])
                        states = _decode_state_logits(state_logits)
                    absolute_error += float((states - targets).abs().sum())
                    feature_size = states.shape[1]
                    predicted_bits = states > 0.5
                    target_bits = targets > 0.5
                    dynamic = dynamic_state_indices(feature_size)
                    bit_correct += int(
                        (predicted_bits[:, dynamic] == target_bits[:, dynamic]).sum()
                    )
                    bit_count += target_bits[:, dynamic].numel()
                    agents = agent_position_indices(feature_size)
                    agent_correct += int(
                        (
                            states[:, agents].argmax(dim=1)
                            == targets[:, agents].argmax(dim=1)
                        ).sum()
                    )

                count = len(jobs)
                metrics[f"rollout_mae_{horizon}"] = absolute_error / (count * model.feature_size)
                metrics[f"rollout_dynamic_bit_accuracy_{horizon}"] = bit_correct / bit_count
                metrics[f"rollout_agent_position_accuracy_{horizon}"] = agent_correct / count
                metrics[f"rollout_count_{horizon}"] = float(count)
    finally:
        model.train(was_training)
    return metrics


def _decode_state_logits(state_logits: Tensor) -> Tensor:
    """Map next-state logits to encoded-feature space for scoring and rollouts."""
    predicted_state = torch.sigmoid(state_logits)
    agents = agent_position_indices(state_logits.shape[1])
    agent_logits = state_logits[:, agents]
    locations = agent_logits.argmax(dim=1, keepdim=True)
    predicted_state[:, agents] = torch.zeros_like(agent_logits).scatter_(
        1, locations, 1.0
    )
    return predicted_state


def _chained_trajectories(
    examples: tuple[TransitionExample, ...],
) -> list[list[TransitionExample]]:
    """Split transitions into maximal runs of exactly connected states."""
    grouped: dict[tuple[object, ...], list[TransitionExample]] = {}
    for row in examples:
        key = (row.layout_id, row.seed, row.gate_kind, row.episode_kind)
        grouped.setdefault(key, []).append(row)

    trajectories: list[list[TransitionExample]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (row.step, row.action))
        chain: list[TransitionExample] = []
        for row in rows:
            if chain and chain[-1].next_state != row.state:
                trajectories.append(chain)
                chain = []
            chain.append(row)
        if chain:
            trajectories.append(chain)
    return trajectories


def _validate_settings(settings: TrainingSettings) -> None:
    if settings.epochs < 1 or settings.batch_size < 1 or settings.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if settings.hidden_size < 8:
        raise ValueError("hidden_size must be at least eight")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
