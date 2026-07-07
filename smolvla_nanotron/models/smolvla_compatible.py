from __future__ import annotations

import re
import zlib

import torch
from torch import nn

from smolvla_nanotron.data.collator import VLABatch
from smolvla_nanotron.models.tiny_policy import masked_mse_loss

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hash_tokens(texts: list[str], vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids: list[int] = []
    offsets: list[int] = []
    for text in texts:
        offsets.append(len(token_ids))
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            tokens = ["<empty>"]
        token_ids.extend([zlib.crc32(token.encode("utf-8")) % vocab_size for token in tokens])
    return torch.tensor(token_ids, dtype=torch.long, device=device), torch.tensor(offsets, dtype=torch.long, device=device)


class TaskTextHashEncoder(nn.Module):
    """Small dependency-free text encoder for task strings.

    This is a practical smoke-test substitute for a full language model. It keeps
    the VLA batch contract explicit while avoiding heavyweight tokenizer/model
    dependencies in early infrastructure validation.
    """

    def __init__(self, vocab_size: int = 4096, embed_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")

    def forward(self, task_text: list[str], device: torch.device) -> torch.Tensor:
        token_ids, offsets = _hash_tokens(task_text, self.vocab_size, device)
        return self.embedding(token_ids, offsets)


class MultiCameraVisionEncoder(nn.Module):
    """Shared image encoder for LeRobot multi-camera observations."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.SiLU(),
        )

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for camera in sorted(images):
            image = images[camera]
            if image.dtype == torch.uint8:
                image = image.float().div_(255.0)
            else:
                image = image.float()
            features.append(self.encoder(image))
        return torch.stack(features, dim=0).mean(dim=0)


class SmolVLACompatiblePolicy(nn.Module):
    """Compact VLA policy wrapper used before full SmolVLA integration.

    The wrapper consumes the same multimodal batch fields expected by a SmolVLA
    fine-tuning path: multi-camera images, robot state, task text, and action
    targets. It is intentionally lightweight so checkpoint/resume and data
    pipeline behavior can be validated before introducing the official model.
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        vision_dim: int = 128,
        task_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.vision_encoder = MultiCameraVisionEncoder(out_dim=vision_dim)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 128), nn.SiLU())
        self.task_encoder = TaskTextHashEncoder(embed_dim=task_dim)
        self.action_head = nn.Sequential(
            nn.Linear(vision_dim + 128 + task_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, batch: VLABatch) -> torch.Tensor:
        if batch.images is None:
            raise ValueError("SmolVLACompatiblePolicy requires batch.images. Run with include_images=True.")
        device = batch.state.device
        vision_features = self.vision_encoder(batch.images)
        state_features = self.state_encoder(batch.state.float())
        task_features = self.task_encoder(batch.task_text, device)
        features = torch.cat([vision_features, state_features, task_features], dim=-1)
        return self.action_head(features)

    def loss(self, batch: VLABatch) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.forward(batch)
        loss = masked_mse_loss(prediction, batch.action.float(), batch.action_mask)
        return loss, prediction
