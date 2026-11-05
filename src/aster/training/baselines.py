"""Training-frequency baseline: per-action most common next outcome.

This predictor memorizes, for each action, the modal next-state fields and
modal outcomes observed in training data. It ignores the current state, so
a trained model must beat it to show state-dependent dynamics learning.
The continuous time field uses the per-action mean because a mode is not
well defined for it.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from aster.training.data import TransitionExample
from aster.training.metrics import field_metrics


class FrequencyBaseline:
    """Predict the per-action most common outcome from training frequencies."""

    def __init__(self, training_examples: Sequence[TransitionExample]) -> None:
        if not training_examples:
            raise ValueError("frequency baseline needs training examples")
        self.feature_size = len(training_examples[0].state)
        self._states: dict[int, tuple[float, ...]] = {}
        self._rewards: dict[int, float] = {}
        self._dones: dict[int, float] = {}
        self._invalids: dict[int, float] = {}

        by_action: dict[int, list[TransitionExample]] = {}
        for row in training_examples:
            if len(row.state) != self.feature_size:
                raise ValueError("all observations must have the same feature size")
            by_action.setdefault(row.action, []).append(row)
        for action, rows in by_action.items():
            self._states[action] = _modal_next_state(rows, self.feature_size)
            self._rewards[action] = _modal_value([row.reward for row in rows])
            self._dones[action] = _modal_value([float(row.done) for row in rows])
            self._invalids[action] = _modal_value([float(row.invalid) for row in rows])

    def predict(
        self,
        examples: Sequence[TransitionExample],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return modal predictions as outcome-space tensors."""
        if not examples:
            raise ValueError("cannot predict an empty dataset split")
        fallback = next(iter(self._states.values()))
        fallback_reward = next(iter(self._rewards.values()))
        fallback_done = next(iter(self._dones.values()))
        fallback_invalid = next(iter(self._invalids.values()))
        states = [self._states.get(row.action, fallback) for row in examples]
        rewards = [self._rewards.get(row.action, fallback_reward) for row in examples]
        dones = [self._dones.get(row.action, fallback_done) for row in examples]
        invalids = [
            self._invalids.get(row.action, fallback_invalid) for row in examples
        ]
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(invalids, dtype=torch.float32),
        )


def evaluate_frequency_baseline(
    training_examples: tuple[TransitionExample, ...],
    evaluation_examples: tuple[TransitionExample, ...],
) -> dict[str, float]:
    """Score the per-action frequency predictor with the model's field metrics."""
    baseline = FrequencyBaseline(training_examples)
    predicted_state, predicted_reward, predicted_done, predicted_invalid = (
        baseline.predict(evaluation_examples)
    )
    next_state = torch.tensor(
        [row.next_state for row in evaluation_examples],
        dtype=torch.float32,
    )
    reward = torch.tensor([row.reward for row in evaluation_examples], dtype=torch.float32)
    done = torch.tensor([float(row.done) for row in evaluation_examples], dtype=torch.float32)
    invalid = torch.tensor(
        [float(row.invalid) for row in evaluation_examples],
        dtype=torch.float32,
    )
    return field_metrics(
        predicted_state,
        predicted_reward,
        predicted_done,
        predicted_invalid,
        next_state,
        reward,
        done,
        invalid,
    )


def _modal_next_state(
    rows: Sequence[TransitionExample],
    feature_size: int,
) -> tuple[float, ...]:
    """Element-wise mode of binary fields and mean of the continuous time field."""
    count = len(rows)
    sums = [0.0] * feature_size
    for row in rows:
        for index, value in enumerate(row.next_state):
            sums[index] += value
    return tuple(
        round(sums[index] / count) if index < feature_size - 1 else sums[index] / count
        for index in range(feature_size)
    )


def _modal_value(values: Sequence[float]) -> float:
    """Most common value; ties resolve to the smallest candidate."""
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))
