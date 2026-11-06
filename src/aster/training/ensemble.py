"""Seed-ensemble predictive uncertainty for Workshop transition models.

A deterministic network cannot express uncertainty about its own
predictions. Training a small committee from different random
initializations turns their disagreement into a measurable uncertainty
estimate: familiar states should produce little disagreement, unfamiliar
states more, and errors should concentrate where disagreement is high.
Confident wrong predictions are a gating failure.
"""

from dataclasses import replace

import torch
from torch import Tensor

from aster.cognition.world_model import TransitionModel, examples_to_tensors
from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES
from aster.training.data import TransitionExample
from aster.training.engine import TrainingSettings, train_world_model
from aster.training.metrics import dynamic_state_indices

_FAMILIAR_SAMPLE_LIMIT = 2000
_CONFIDENT_QUANTILE = 0.25


def train_uncertainty_ensemble(
    base_model: TransitionModel,
    training_examples: tuple[TransitionExample, ...],
    validation_examples: tuple[TransitionExample, ...],
    settings: TrainingSettings,
    size: int,
) -> tuple[TransitionModel, ...]:
    """Return a committee: the trained base model plus differently seeded members."""
    if size < 2:
        raise ValueError("ensemble size must be at least two")
    members: list[TransitionModel] = [base_model]
    for index in range(1, size):
        member_settings = replace(settings, seed=settings.seed + 997 * index)
        result = train_world_model(
            training_examples,
            validation_examples,
            member_settings,
        )
        members.append(result.model)
    return tuple(members)


def evaluate_uncertainty(
    members: tuple[TransitionModel, ...],
    familiar_examples: tuple[TransitionExample, ...],
    unfamiliar_examples: tuple[TransitionExample, ...],
) -> dict[str, float]:
    """Compare committee disagreement on familiar vs unfamiliar states.

    ``familiar_examples`` are training-distribution states; ``unfamiliar_examples``
    are states from layouts no split member trained on. An example counts as
    an error when the committee's mean prediction gets the agent position, the
    gate flag, or the key flag wrong. The error-detection AUROC is undefined
    when only one error class is present; ``error_auroc_defined`` reports that.
    """
    if len(members) < 2:
        raise ValueError("uncertainty needs at least two ensemble members")
    if not familiar_examples or not unfamiliar_examples:
        raise ValueError("both familiar and unfamiliar examples are required")

    familiar_sample = _even_sample(familiar_examples, _FAMILIAR_SAMPLE_LIMIT)
    familiar_scores, familiar_errors, familiar_confident_cut = _score_set(
        members, familiar_sample
    )
    unfamiliar_scores, unfamiliar_errors, _ = _score_set(members, unfamiliar_examples)

    pooled_scores = familiar_scores + unfamiliar_scores
    pooled_errors = familiar_errors + unfamiliar_errors
    auroc, auroc_defined = _auroc(pooled_scores, pooled_errors)
    confidently_wrong = sum(
        1
        for score, error in zip(pooled_scores, pooled_errors)
        if error and score <= familiar_confident_cut
    )
    familiar_mean = sum(familiar_scores) / len(familiar_scores)
    unfamiliar_mean = sum(unfamiliar_scores) / len(unfamiliar_scores)

    return {
        "familiar_examples": float(len(familiar_scores)),
        "unfamiliar_examples": float(len(unfamiliar_scores)),
        "familiar_mean_uncertainty": familiar_mean,
        "unfamiliar_mean_uncertainty": unfamiliar_mean,
        "uncertainty_shift_ratio": (
            unfamiliar_mean / familiar_mean if familiar_mean > 0 else 1.0
        ),
        "familiar_error_rate": sum(familiar_errors) / len(familiar_errors),
        "unfamiliar_error_rate": sum(unfamiliar_errors) / len(unfamiliar_errors),
        "error_auroc": auroc,
        "error_auroc_defined": 1.0 if auroc_defined else 0.0,
        "confidently_wrong_rate": confidently_wrong / len(pooled_scores),
    }


def _score_set(
    members: tuple[TransitionModel, ...],
    examples: tuple[TransitionExample, ...],
) -> tuple[list[float], list[bool], float]:
    """Per-example disagreement scores and error flags for one state set."""
    probabilities, logits = _member_outputs(members, examples)
    mean_prediction = probabilities.mean(dim=0)
    mean_logits = logits.mean(dim=0)
    spread = probabilities.std(dim=0, unbiased=False)
    indices = dynamic_state_indices(mean_prediction.shape[1])
    uncertainty = spread[:, indices].mean(dim=1).tolist()

    target = torch.tensor(
        [row.next_state for row in examples],
        dtype=torch.float32,
    )
    errors = _error_flags(mean_prediction, mean_logits, target).tolist()
    ordered = sorted(uncertainty)
    cut_index = int(len(ordered) * _CONFIDENT_QUANTILE)
    confident_cut = ordered[min(cut_index, len(ordered) - 1)]
    return uncertainty, [bool(flag) for flag in errors], confident_cut


def _member_outputs(
    members: tuple[TransitionModel, ...],
    examples: tuple[TransitionExample, ...],
) -> tuple[Tensor, Tensor]:
    """Sigmoid probabilities and raw logits from every member, stacked per member."""
    device = next(members[0].parameters()).device
    tensors = examples_to_tensors(examples, device)
    probability_rows: list[Tensor] = []
    logit_rows: list[Tensor] = []
    for model in members:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                state_logits, _, _, _ = model(tensors[0], tensors[1])
            probability_rows.append(torch.sigmoid(state_logits))
            logit_rows.append(state_logits)
        finally:
            model.train(was_training)
    return torch.stack(probability_rows), torch.stack(logit_rows)


def _error_flags(mean_prediction: Tensor, mean_logits: Tensor, target: Tensor) -> Tensor:
    """Hard errors: wrong agent position, wrong gate flag, or wrong key flag.

    The agent position is compared in logit space: sigmoid saturates for
    confident predictions, and argmax over saturated probabilities is a
    tie broken by cell index rather than by the model's belief.
    """
    feature_size = mean_prediction.shape[1]
    cell_feature_count = feature_size - len(GLOBAL_FEATURES)
    cell_count = cell_feature_count // len(CHANNELS)
    agent_indices = [cell * len(CHANNELS) + 1 for cell in range(cell_count)]
    agent_wrong = (
        mean_logits[:, agent_indices].argmax(dim=1)
        != target[:, agent_indices].argmax(dim=1)
    )
    gate_index = cell_feature_count + GLOBAL_FEATURES.index("gate_open")
    key_index = cell_feature_count + GLOBAL_FEATURES.index("has_key")
    gate_wrong = (mean_prediction[:, gate_index] > 0.5) != (target[:, gate_index] > 0.5)
    key_wrong = (mean_prediction[:, key_index] > 0.5) != (target[:, key_index] > 0.5)
    return agent_wrong | gate_wrong | key_wrong


def _auroc(scores: list[float], labels: list[bool]) -> tuple[float, bool]:
    """Probability a wrong example scores above a correct one (ties count half)."""
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5, False
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives)), True


def _even_sample(
    examples: tuple[TransitionExample, ...],
    limit: int,
) -> tuple[TransitionExample, ...]:
    if len(examples) <= limit:
        return examples
    stride = len(examples) / limit
    return tuple(examples[int(index * stride)] for index in range(limit))
