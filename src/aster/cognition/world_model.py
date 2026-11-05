"""A small action-conditioned neural transition model trained from experience."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from aster.environment import Action
from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES
from aster.training.data import TransitionExample


class TransitionModel(nn.Module):
    """Predict the next encoded world state and the outcome of one action."""

    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        if feature_size < 2:
            raise ValueError("feature_size must be at least two")
        if hidden_size < 8:
            raise ValueError("hidden_size must be at least eight")
        self.feature_size = feature_size
        self.action_count = len(Action)
        self.layers = nn.Sequential(
            nn.Linear(feature_size + self.action_count, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, feature_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)
        self.invalid_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_vector = functional.one_hot(actions.long(), self.action_count).to(state.dtype)
        hidden = self.layers(torch.cat((state, action_vector), dim=-1))
        return (
            self.next_state_head(hidden),
            self.reward_head(hidden).squeeze(-1),
            self.done_head(hidden).squeeze(-1),
            self.invalid_head(hidden).squeeze(-1),
        )


def transition_loss(
    prediction: tuple[Tensor, Tensor, Tensor, Tensor],
    next_state: Tensor,
    reward: Tensor,
    done: Tensor,
    invalid: Tensor,
) -> Tensor:
    """Combine dynamics, outcome, terminal, and action-validity losses."""
    state_logits, reward_prediction, done_logits, invalid_logits = prediction
    cell_feature_count = state_logits.shape[1] - len(GLOBAL_FEATURES)
    cell_count = cell_feature_count // len(CHANNELS)
    agent_index_values = [
        cell * len(CHANNELS) + 1
        for cell in range(cell_count)
    ]
    agent_indices = torch.tensor(agent_index_values, device=state_logits.device)
    agent_index_set = set(agent_index_values)
    binary_indices = [
        index
        for index in range(cell_feature_count)
        if index not in agent_index_set
    ]
    binary_indices.extend(range(cell_feature_count, state_logits.shape[1] - 1))
    binary_state_loss = functional.binary_cross_entropy_with_logits(
        state_logits[:, binary_indices],
        next_state[:, binary_indices],
    )
    next_agent_cells = next_state[:, agent_indices].argmax(dim=1)
    agent_location_loss = functional.cross_entropy(
        state_logits[:, agent_indices],
        next_agent_cells,
    )
    time_prediction = torch.sigmoid(state_logits[:, -1])
    time_loss = functional.smooth_l1_loss(time_prediction, next_state[:, -1])
    reward_loss = functional.smooth_l1_loss(torch.tanh(reward_prediction), reward)
    done_loss = functional.binary_cross_entropy_with_logits(done_logits, done)
    invalid_loss = functional.binary_cross_entropy_with_logits(invalid_logits, invalid)
    return (
        binary_state_loss
        + agent_location_loss
        + time_loss
        + reward_loss
        + done_loss
        + invalid_loss
    )


def predict_transition(
    model: TransitionModel,
    state: Sequence[float],
    action: Action,
) -> tuple[tuple[float, ...], float, float, float]:
    """Predict a single transition without changing the model's training state."""
    return predict_transitions(model, [state], [action])[0]


def predict_transitions(
    model: TransitionModel,
    states: Sequence[Sequence[float]],
    actions: Sequence[Action],
) -> list[tuple[tuple[float, ...], float, float, float]]:
    """Predict a batch of transitions for efficient beam-search expansion."""
    if len(states) != len(actions):
        raise ValueError("states and actions must have the same number of items")
    if not states:
        return []
    if any(len(state) != model.feature_size for state in states):
        raise ValueError("every state must match the model feature size")

    device = next(model.parameters()).device
    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    action_ids = {action: index for index, action in enumerate(Action)}
    action_tensor = torch.tensor(
        [action_ids[action] for action in actions],
        dtype=torch.long,
        device=device,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state_logits, reward_prediction, done_logits, invalid_logits = model(
                state_tensor,
                action_tensor,
            )
            predicted_state = torch.sigmoid(state_logits)
            cell_feature_count = model.feature_size - len(GLOBAL_FEATURES)
            cell_count = cell_feature_count // len(CHANNELS)
            agent_indices = [cell * len(CHANNELS) + 1 for cell in range(cell_count)]
            agent_logits = state_logits[:, agent_indices]
            agent_locations = agent_logits.argmax(dim=1, keepdim=True)
            agent_one_hot = torch.zeros_like(agent_logits).scatter_(1, agent_locations, 1.0)
            predicted_state[:, agent_indices] = agent_one_hot
            predicted_rewards = torch.tanh(reward_prediction)
            predicted_done = torch.sigmoid(done_logits)
            predicted_invalid = torch.sigmoid(invalid_logits)
    finally:
        model.train(was_training)

    return [
        (
            tuple(float(value) for value in predicted_state[index].cpu()),
            float(predicted_rewards[index].cpu()),
            float(predicted_done[index].cpu()),
            float(predicted_invalid[index].cpu()),
        )
        for index in range(len(states))
    ]


def load_world_model(path: str | Path, device: str = "cpu") -> TransitionModel:
    """Load a locally trained checkpoint after validating its interface version."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_actions = [action.value for action in Action]
    if checkpoint.get("format_version") != 2:
        raise ValueError("unsupported world-model checkpoint version")
    if checkpoint.get("environment_version") != 1:
        raise ValueError("checkpoint was trained for a different environment version")
    if checkpoint.get("action_names") != expected_actions:
        raise ValueError("checkpoint action vocabulary does not match this runtime")

    model = TransitionModel(
        feature_size=int(checkpoint["feature_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def examples_to_tensors(
    examples: Sequence[TransitionExample],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not examples:
        raise ValueError("cannot build tensors from an empty dataset split")
    states = torch.tensor([row.state for row in examples], dtype=torch.float32, device=device)
    actions = torch.tensor([row.action for row in examples], dtype=torch.long, device=device)
    next_states = torch.tensor(
        [row.next_state for row in examples],
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.tensor([row.reward for row in examples], dtype=torch.float32, device=device)
    dones = torch.tensor([row.done for row in examples], dtype=torch.float32, device=device)
    invalid = torch.tensor([row.invalid for row in examples], dtype=torch.float32, device=device)
    return states, actions, next_states, rewards, dones, invalid
