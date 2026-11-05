"""A small learned action policy for Workshop behavior-cloning experiments."""

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from aster.environment import Action, Observation, available_actions, encode_observation


class ActionPolicy(nn.Module):
    """Map an encoded observation to unmasked action logits."""

    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        if feature_size < 2:
            raise ValueError("feature_size must be at least two")
        if hidden_size < 8:
            raise ValueError("hidden_size must be at least eight")
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, len(Action)),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)

    def choose(self, observation: Observation) -> Action:
        """Choose the highest-scoring currently legal action, with no goal heuristic."""
        state_values = encode_observation(observation)
        device = next(self.parameters()).device
        state = torch.tensor([state_values], dtype=torch.float32, device=device)
        legal_actions = available_actions(observation)
        action_ids = {action: index for index, action in enumerate(Action)}
        with torch.no_grad():
            scores = self(state)[0]
        selected = max(legal_actions, key=lambda action: float(scores[action_ids[action]]))
        return selected


def save_policy_checkpoint(
    model: ActionPolicy,
    path: str | Path,
    training_seed: int,
) -> None:
    """Save a local policy checkpoint with its action vocabulary and version."""
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "feature_size": model.feature_size,
            "hidden_size": model.hidden_size,
            "action_names": [action.value for action in Action],
            "training_seed": training_seed,
            "environment_version": 1,
        },
        path,
    )


def load_policy_checkpoint(path: str | Path, device: str = "cpu") -> ActionPolicy:
    """Load a policy after validating its feature and action interfaces."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported action-policy checkpoint version")
    if checkpoint.get("environment_version") != 1:
        raise ValueError("checkpoint was trained for a different environment version")
    if checkpoint.get("action_names") != [action.value for action in Action]:
        raise ValueError("checkpoint action vocabulary does not match this runtime")

    model = ActionPolicy(
        feature_size=int(checkpoint["feature_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
