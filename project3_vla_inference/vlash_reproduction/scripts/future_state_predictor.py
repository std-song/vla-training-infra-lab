"""Lightweight residual future-state predictor for VLASH action chunks."""

from __future__ import annotations

import torch
from torch import nn


class ResidualStatePredictor(nn.Module):
    """Predict the correction from the VLASH endpoint-action proxy to future state."""

    def __init__(self, state_dim: int, action_dim: int, max_horizon: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        input_dim = state_dim + max_horizon * action_dim + max_horizon + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        current_state_normalized: torch.Tensor,
        action_prefix_normalized: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        """Return a normalized-state residual over the endpoint-action proxy."""
        steps = torch.arange(self.max_horizon, device=horizon.device).unsqueeze(0)
        mask = steps < horizon.unsqueeze(1)
        masked_actions = action_prefix_normalized * mask.unsqueeze(-1)
        features = torch.cat(
            (
                current_state_normalized,
                masked_actions.flatten(1),
                mask.to(current_state_normalized.dtype),
                horizon.unsqueeze(1).to(current_state_normalized.dtype) / self.max_horizon,
            ),
            dim=1,
        )
        return self.network(features)


def predict_future_state(
    model: ResidualStatePredictor,
    current_state: torch.Tensor,
    action_prefix: torch.Tensor,
    horizon: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict future state in the original state coordinate system."""
    state_mean = stats["state_mean"].to(current_state)
    state_std = stats["state_std"].to(current_state)
    action_mean = stats["action_mean"].to(action_prefix)
    action_std = stats["action_std"].to(action_prefix)
    state_normalized = (current_state - state_mean) / state_std
    actions_normalized = (action_prefix - action_mean) / action_std
    residual = model(state_normalized, actions_normalized, horizon)
    endpoint = action_prefix[torch.arange(action_prefix.shape[0], device=horizon.device), horizon - 1]
    endpoint_as_state = (endpoint - state_mean) / state_std
    return (endpoint_as_state + residual) * state_std + state_mean
