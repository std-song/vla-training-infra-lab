"""Sequence-conditioned future-state model for LIBERO's mismatched state/action spaces."""

from __future__ import annotations

import torch
from torch import nn


class ActionSequenceStatePredictor(nn.Module):
    """Predict normalized future state from current state and a planned action prefix."""

    def __init__(self, state_dim: int, action_dim: int, max_horizon: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.step_embedding = nn.Embedding(max_horizon, hidden_dim)
        self.action_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + state_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        current_state_normalized: torch.Tensor,
        action_prefix_normalized: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted future state in normalized state coordinates."""
        batch_size, sequence_length, _ = action_prefix_normalized.shape
        if sequence_length != self.max_horizon:
            raise ValueError(f"Expected {self.max_horizon} actions, got {sequence_length}")
        steps = torch.arange(self.max_horizon, device=horizon.device)
        action_tokens = self.action_encoder(action_prefix_normalized)
        action_tokens = action_tokens + self.step_embedding(steps).unsqueeze(0)
        initial_hidden = self.state_encoder(current_state_normalized).unsqueeze(0)
        outputs, _ = self.action_gru(action_tokens, initial_hidden)
        selected = outputs[torch.arange(batch_size, device=horizon.device), horizon - 1]
        horizon_feature = horizon.unsqueeze(1).to(current_state_normalized.dtype) / self.max_horizon
        delta = self.state_decoder(torch.cat((selected, current_state_normalized, horizon_feature), dim=1))
        return current_state_normalized + delta


def predict_future_state(
    model: ActionSequenceStatePredictor,
    current_state: torch.Tensor,
    action_prefix: torch.Tensor,
    horizon: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    state_mean = stats["state_mean"].to(current_state)
    state_std = stats["state_std"].to(current_state)
    action_mean = stats["action_mean"].to(action_prefix)
    action_std = stats["action_std"].to(action_prefix)
    state_normalized = (current_state - state_mean) / state_std
    action_normalized = (action_prefix - action_mean) / action_std
    prediction_normalized = model(state_normalized, action_normalized, horizon)
    return prediction_normalized * state_std + state_mean
