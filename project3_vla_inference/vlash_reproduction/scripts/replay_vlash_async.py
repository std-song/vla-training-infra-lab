"""Offline ALOHA replay that delegates action scheduling to upstream VLASH."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vlash.policies.factory import get_policy_class
from vlash.run import VLASHAsyncManager


@dataclass
class ReplayRobot:
    robot_type: str = "aloha"
    action_features: list[str] | None = None

    def __post_init__(self) -> None:
        self.action_features = [f"action_{index}" for index in range(14)]


def image_to_hwc_uint8(value: torch.Tensor) -> np.ndarray:
    image = value.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.numpy()
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return array


def observation_from_sample(sample: dict) -> dict[str, np.ndarray]:
    observation = {"observation.state": sample["observation.state"].detach().cpu().numpy().astype(np.float32)}
    for key, value in sample.items():
        if key.startswith("observation.images."):
            observation[key] = image_to_hwc_uint8(value)
    return observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--overlap-steps", type=int, default=0)
    parser.add_argument("--quant-ratio", type=int, default=1)
    parser.add_argument("--task", default="Open the top cabinet, store the pot inside it then close the cabinet.")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        "lerobot/aloha_mobile_cabinet", root=args.dataset_root, episodes=[0], video_backend="torchcodec"
    )
    policy = get_policy_class("pi05").from_pretrained(args.policy_path)
    policy.eval()
    manager = VLASHAsyncManager(policy, ReplayRobot(), args.task, args.overlap_steps * args.quant_ratio)

    rows: list[dict[str, float | int]] = []
    for step in range(args.steps):
        sample = dataset[step]
        observation = observation_from_sample(sample)
        fetch = int(manager.should_fetch_observation())
        started = time.perf_counter()
        action = manager.get_action(observation)
        latency_ms = (time.perf_counter() - started) * 1000
        sent = int((step + 1) % args.quant_ratio == 0)
        rows.append({
            "step": step,
            "fetch_observation": fetch,
            "latency_ms": latency_ms,
            "chunk_index": manager.chunk_index,
            "queue_active": int(manager.is_running()),
            "sent_action": sent,
            "action_l2": float(np.linalg.norm(list(action.values()))),
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({"steps": args.steps, "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 3), "out": str(output)})


if __name__ == "__main__":
    main()
