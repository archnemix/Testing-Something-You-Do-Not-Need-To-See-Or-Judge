"""Reproducible supervised fitting and held-out evaluation for world dynamics."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from aster.cognition.world_model import TransitionModel, examples_to_tensors, transition_loss
from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES
from aster.training.data import TransitionExample


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
    binary_correct = 0
    binary_count = 0
    agent_position_correct = 0
    gate_state_correct = 0
    key_state_correct = 0
    time_absolute_error = 0.0
    reward_absolute_error = 0.0
    done_correct = 0
    invalid_correct = 0
    invalid_absolute_error = 0.0

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            end = min(start + batch_size, len(examples))
            batch = tuple(tensor[start:end] for tensor in tensors)
            state_logits, reward_prediction, done_logits, invalid_logits = model(batch[0], batch[1])
            loss = transition_loss(
                (state_logits, reward_prediction, done_logits, invalid_logits),
                batch[2],
                batch[3],
                batch[4],
                batch[5],
            )
            count = end - start
            weighted_loss += float(loss) * count
            predicted_bits = state_logits > 0
            target_bits = batch[2] > 0.5
            cell_feature_count = len(batch[0][0]) - len(GLOBAL_FEATURES)
            cell_count = cell_feature_count // len(CHANNELS)
            dynamic_indices = [
                cell * len(CHANNELS) + channel
                for cell in range(cell_count)
                for channel in (1, 3, 4)
            ]
            global_offset = cell_feature_count
            dynamic_indices.extend((global_offset, global_offset + 1))
            binary_correct += int(
                (predicted_bits[:, dynamic_indices] == target_bits[:, dynamic_indices]).sum()
            )
            binary_count += target_bits[:, dynamic_indices].numel()

            agent_indices = [
                cell * len(CHANNELS) + 1
                for cell in range(cell_count)
            ]
            agent_position_correct += int(
                (
                    state_logits[:, agent_indices].argmax(dim=1)
                    == batch[2][:, agent_indices].argmax(dim=1)
                ).sum()
            )
            gate_index = global_offset + GLOBAL_FEATURES.index("gate_open")
            key_index = global_offset + GLOBAL_FEATURES.index("has_key")
            gate_state_correct += int(
                (predicted_bits[:, gate_index] == target_bits[:, gate_index]).sum()
            )
            key_state_correct += int(
                (predicted_bits[:, key_index] == target_bits[:, key_index]).sum()
            )
            predicted_time = torch.sigmoid(state_logits[:, -1])
            time_absolute_error += float((predicted_time - batch[2][:, -1]).abs().sum())
            predicted_reward = torch.tanh(reward_prediction)
            reward_absolute_error += float((predicted_reward - batch[3]).abs().sum())
            predicted_done = done_logits > 0
            done_correct += int((predicted_done == (batch[4] > 0.5)).sum())
            predicted_invalid = torch.sigmoid(invalid_logits)
            invalid_correct += int(
                ((predicted_invalid > 0.5) == (batch[5] > 0.5)).sum()
            )
            invalid_absolute_error += float((predicted_invalid - batch[5]).abs().sum())

    model.train(was_training)
    count = len(examples)
    return {
        "loss": weighted_loss / count,
        "dynamic_state_bit_accuracy": binary_correct / binary_count,
        "agent_position_accuracy": agent_position_correct / count,
        "gate_state_accuracy": gate_state_correct / count,
        "key_state_accuracy": key_state_correct / count,
        "time_remaining_mae": time_absolute_error / count,
        "reward_mae": reward_absolute_error / count,
        "done_accuracy": done_correct / count,
        "invalid_action_accuracy": invalid_correct / count,
        "invalid_action_mae": invalid_absolute_error / count,
    }


def _validate_settings(settings: TrainingSettings) -> None:
    if settings.epochs < 1 or settings.batch_size < 1 or settings.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if settings.hidden_size < 8:
        raise ValueError("hidden_size must be at least eight")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
