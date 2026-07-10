from __future__ import annotations

import torch
from torch import nn


class TinyLowDimVLAPolicy(nn.Module):
    """Small MLP policy used to validate LeRobot batch plumbing.

    It is deliberately not SmolVLA. Its job is to provide a cheap trainable
    target while the Nanotron adapter and data pipeline are being assembled.
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 14, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean((prediction - target) ** 2)
    loss = (prediction - target) ** 2
    return loss.masked_select(mask).mean()
