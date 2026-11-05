"""Behavior-cloning training for the experimental Workshop action policy."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from aster.cognition.policy import ActionPolicy
from aster.training.data import TransitionExample
from aster.training.engine import TrainingSettings


@dataclass(frozen=True, slots=True)
class PolicyTrainingResult:
    model: ActionPolicy
    history: tuple[dict[str, float], ...]
    best_validation_loss: float


def train_action_policy(
    training_examples: tuple[TransitionExample, ...],
    validation_examples: tuple[TransitionExample, ...],
    settings: TrainingSettings,
) -> PolicyTrainingResult:
    """Fit on expert demonstrations only; random and counterfactual actions are not labels."""
    _validate_settings(settings)
    training = _expert_examples(training_examples)
    validation = _expert_examples(validation_examples)
    if not training or not validation:
        raise ValueError("policy training needs expert examples in both splits")
    feature_size = len(training[0].state)
    if any(len(row.state) != feature_size for row in training + validation):
        raise ValueError("all policy observations must have the same feature size")

    device = torch.device(settings.device)
    torch.manual_seed(settings.seed + 1)
    train_states, train_actions = _to_tensors(training, device)
    validation_tensors = _to_tensors(validation, device)
    model = ActionPolicy(feature_size, settings.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator(device=device).manual_seed(settings.seed + 1)
    best_loss = float("inf")
    best_weights: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(train_states), generator=generator, device=device)
        for indices in order.split(settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                model(train_states[indices]),
                train_actions[indices],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics = evaluate_policy_examples(model, validation, device=device)
        history.append({"epoch": float(epoch), **metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    if best_weights is None:
        raise RuntimeError("policy training did not produce a valid checkpoint")
    model.load_state_dict(best_weights)
    return PolicyTrainingResult(model, tuple(history), best_loss)


def evaluate_policy_examples(
    model: ActionPolicy,
    examples: tuple[TransitionExample, ...],
    device: torch.device | None = None,
    batch_size: int = 512,
) -> dict[str, float]:
    """Report held-out expert action cross-entropy and exact-action accuracy."""
    rows = _expert_examples(examples)
    if not rows:
        raise ValueError("cannot evaluate a policy without expert examples")
    evaluation_device = device or next(model.parameters()).device
    states, actions = _to_tensors(rows, evaluation_device)
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            logits = model(states[start:end])
            labels = actions[start:end]
            loss_sum += float(
                functional.cross_entropy(logits, labels, reduction="sum")
            )
            correct += int((logits.argmax(dim=1) == labels).sum())
    model.train(was_training)
    return {"loss": loss_sum / len(rows), "expert_action_accuracy": correct / len(rows)}


def _expert_examples(
    examples: tuple[TransitionExample, ...],
) -> tuple[TransitionExample, ...]:
    return tuple(row for row in examples if row.episode_kind == "expert")


def _to_tensors(
    examples: tuple[TransitionExample, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    states = torch.tensor([row.state for row in examples], dtype=torch.float32, device=device)
    actions = torch.tensor([row.action for row in examples], dtype=torch.long, device=device)
    return states, actions


def _validate_settings(settings: TrainingSettings) -> None:
    if settings.epochs < 1 or settings.batch_size < 1 or settings.patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if settings.learning_rate <= 0 or settings.hidden_size < 8:
        raise ValueError("learning_rate must be positive and hidden_size at least eight")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
