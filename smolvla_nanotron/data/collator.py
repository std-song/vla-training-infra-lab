from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VLABatch:
    state: torch.Tensor
    effort: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor
    episode_index: torch.Tensor
    frame_index: torch.Tensor
    timestamp: torch.Tensor
    done: torch.Tensor
    task_index: torch.Tensor
    task_text: list[str]
    images: dict[str, torch.Tensor] | None = None


def collate_lerobot_lowdim(samples: list[dict[str, Any]]) -> VLABatch:
    state = torch.stack([sample["state"] for sample in samples], dim=0)
    effort = torch.stack([sample["effort"] for sample in samples], dim=0)
    action = torch.stack([sample["action"] for sample in samples], dim=0)

    images = None
    if "images" in samples[0]:
        cameras = samples[0]["images"].keys()
        images = {camera: torch.stack([sample["images"][camera] for sample in samples], dim=0) for camera in cameras}

    return VLABatch(
        state=state,
        effort=effort,
        action=action,
        action_mask=torch.ones_like(action, dtype=torch.bool),
        episode_index=torch.tensor([sample["episode_index"] for sample in samples], dtype=torch.long),
        frame_index=torch.tensor([sample["frame_index"] for sample in samples], dtype=torch.long),
        timestamp=torch.tensor([sample["timestamp"] for sample in samples], dtype=torch.float32),
        done=torch.tensor([sample["done"] for sample in samples], dtype=torch.bool),
        task_index=torch.tensor([sample["task_index"] for sample in samples], dtype=torch.long),
        task_text=[sample["task_text"] for sample in samples],
        images=images,
    )


def describe_batch(batch: VLABatch) -> str:
    lines = [
        "VLABatch(",
        f"  state={tuple(batch.state.shape)} {batch.state.dtype}",
        f"  effort={tuple(batch.effort.shape)} {batch.effort.dtype}",
        f"  action={tuple(batch.action.shape)} {batch.action.dtype}",
        f"  action_mask={tuple(batch.action_mask.shape)} {batch.action_mask.dtype}",
        f"  episode_index={tuple(batch.episode_index.shape)} {batch.episode_index.dtype}",
        f"  frame_index={tuple(batch.frame_index.shape)} {batch.frame_index.dtype}",
        f"  timestamp={tuple(batch.timestamp.shape)} {batch.timestamp.dtype}",
        f"  done={tuple(batch.done.shape)} {batch.done.dtype}",
        f"  task_index={tuple(batch.task_index.shape)} {batch.task_index.dtype}",
        f"  task_text[0]={batch.task_text[0] if batch.task_text else ''!r}",
    ]
    if batch.images is not None:
        for camera, image_batch in batch.images.items():
            lines.append(f"  images[{camera}]={tuple(image_batch.shape)} {image_batch.dtype}")
    lines.append(")")
    return "\n".join(lines)
