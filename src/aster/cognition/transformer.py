"""Cell-token Transformer baseline for architecture comparisons.

This is a yardstick, not part of the brain-inspired claim: it predicts the
same next-state features, reward, terminal status, and action invalidity
as `TransitionModel`, so both architectures can be trained and scored on
identical data. Each grid cell is one token; a final global token carries
the map-level features and the action.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional

from aster.environment import Action
from aster.environment.encoding import CHANNELS, GLOBAL_FEATURES


class TransformerTransitionModel(nn.Module):
    """Encoder-only Transformer with the ``TransitionModel`` output contract."""

    def __init__(
        self,
        feature_size: int,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        cell_features = feature_size - len(GLOBAL_FEATURES)
        if cell_features <= 0 or cell_features % len(CHANNELS) != 0:
            raise ValueError("feature_size must hold whole cell channel groups")
        if d_model < 8 or layers < 1:
            raise ValueError("d_model must be at least eight and layers at least one")
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")

        self.feature_size = feature_size
        self.action_count = len(Action)
        self.cell_count = cell_features // len(CHANNELS)
        self.cell_projection = nn.Linear(len(CHANNELS), d_model)
        self.global_projection = nn.Linear(
            len(GLOBAL_FEATURES) + self.action_count, d_model
        )
        self.positions = nn.Parameter(torch.zeros(self.cell_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cell_head = nn.Linear(d_model, len(CHANNELS))
        self.global_head = nn.Linear(d_model, len(GLOBAL_FEATURES) + 3)

    def forward(
        self,
        state: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = state.shape[0]
        cells = state[:, : self.cell_count * len(CHANNELS)].reshape(
            batch_size, self.cell_count, len(CHANNELS)
        )
        globals_ = state[:, self.cell_count * len(CHANNELS) :]
        action_vector = functional.one_hot(
            actions.long(), self.action_count
        ).to(state.dtype)

        cell_tokens = self.cell_projection(cells) + self.positions.unsqueeze(0)
        global_token = self.global_projection(
            torch.cat((globals_, action_vector), dim=-1)
        ).unsqueeze(1)
        encoded = self.encoder(torch.cat((cell_tokens, global_token), dim=1))

        cell_logits = self.cell_head(encoded[:, : self.cell_count]).reshape(
            batch_size, self.cell_count * len(CHANNELS)
        )
        global_output = self.global_head(encoded[:, self.cell_count])
        state_logits = torch.cat(
            (cell_logits, global_output[:, : len(GLOBAL_FEATURES)]), dim=-1
        )
        reward = global_output[:, len(GLOBAL_FEATURES)]
        done = global_output[:, len(GLOBAL_FEATURES) + 1]
        invalid = global_output[:, len(GLOBAL_FEATURES) + 2]
        return state_logits, reward, done, invalid
