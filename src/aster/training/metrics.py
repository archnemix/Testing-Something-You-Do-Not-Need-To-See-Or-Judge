"""Shared field-level metrics for one-step transition predictions.

Both the trained model and frequency baselines are scored with these
formulas so their numbers are directly comparable.
"""

from torch import Tensor

from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES


def dynamic_state_indices(feature_size: int) -> list[int]:
    """Indices of state fields that change during an episode.

    Walls, the target marker, and the gate-family marker are static, so
    only dynamic cell channels and the mutable global flags are counted.
    """
    cell_feature_count = feature_size - len(GLOBAL_FEATURES)
    cell_count = cell_feature_count // len(CHANNELS)
    indices = [
        cell * len(CHANNELS) + channel
        for cell in range(cell_count)
        for channel in (1, 3, 4)
    ]
    global_offset = cell_feature_count
    indices.extend((global_offset, global_offset + 1))
    return indices


def agent_position_indices(feature_size: int) -> list[int]:
    """Cell-channel indices holding the one-hot agent position."""
    cell_feature_count = feature_size - len(GLOBAL_FEATURES)
    cell_count = cell_feature_count // len(CHANNELS)
    return [cell * len(CHANNELS) + 1 for cell in range(cell_count)]


def field_metrics(
    predicted_state: Tensor,
    predicted_reward: Tensor,
    predicted_done: Tensor,
    predicted_invalid: Tensor,
    next_state: Tensor,
    reward: Tensor,
    done: Tensor,
    invalid: Tensor,
) -> dict[str, float]:
    """Score predictions given in outcome space (values in [0, 1]).

    ``predicted_state`` and ``next_state`` hold full encoded observations,
    ``predicted_reward`` and ``reward`` are on the environment reward scale,
    and done/invalid predictions are probabilities against 0/1 targets.
    """
    feature_size = predicted_state.shape[1]
    if next_state.shape != predicted_state.shape:
        raise ValueError("predicted and target states must have the same shape")
    count = predicted_state.shape[0]
    if count == 0:
        raise ValueError("cannot score an empty batch")

    predicted_bits = predicted_state > 0.5
    target_bits = next_state > 0.5
    dynamic = dynamic_state_indices(feature_size)
    binary_correct = int(
        (predicted_bits[:, dynamic] == target_bits[:, dynamic]).sum()
    )
    binary_count = target_bits[:, dynamic].numel()

    agent_indices = agent_position_indices(feature_size)
    agent_correct = int(
        (
            predicted_state[:, agent_indices].argmax(dim=1)
            == next_state[:, agent_indices].argmax(dim=1)
        ).sum()
    )

    cell_feature_count = feature_size - len(GLOBAL_FEATURES)
    gate_index = cell_feature_count + GLOBAL_FEATURES.index("gate_open")
    key_index = cell_feature_count + GLOBAL_FEATURES.index("has_key")
    gate_correct = int(
        (predicted_bits[:, gate_index] == target_bits[:, gate_index]).sum()
    )
    key_correct = int(
        (predicted_bits[:, key_index] == target_bits[:, key_index]).sum()
    )

    time_error = float((predicted_state[:, -1] - next_state[:, -1]).abs().sum())
    reward_error = float((predicted_reward - reward).abs().sum())
    done_correct = int(
        ((predicted_done > 0.5) == (done > 0.5)).sum()
    )
    invalid_correct = int(
        ((predicted_invalid > 0.5) == (invalid > 0.5)).sum()
    )
    invalid_error = float((predicted_invalid - invalid).abs().sum())

    return {
        "dynamic_state_bit_accuracy": binary_correct / binary_count,
        "agent_position_accuracy": agent_correct / count,
        "gate_state_accuracy": gate_correct / count,
        "key_state_accuracy": key_correct / count,
        "time_remaining_mae": time_error / count,
        "reward_mae": reward_error / count,
        "done_accuracy": done_correct / count,
        "invalid_action_accuracy": invalid_correct / count,
        "invalid_action_mae": invalid_error / count,
    }
