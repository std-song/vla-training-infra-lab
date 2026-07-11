"""Evaluate a normal or VLASH Pi0.5 policy on held-out delayed action targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torch.utils.data import DataLoader
from vlash.datasets import SharedObservationVLASHDataset
from vlash.policies.factory import get_policy_class


DEFAULT_TASK = "Open the top cabinet, store the pot inside it then close the cabinet."


def mean_metric(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, count: int) -> tuple[float, float]:
    mask = valid[:, :count]
    if not bool(mask.any()):
        return float("nan"), float("nan")
    delta = prediction[:, :count][mask] - target[:, :count][mask]
    return float(delta.abs().mean()), float(delta.square().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--mode", choices=("sync", "vlash"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=102)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    validation_ids = [int(value) for value in split["validation_episode_ids"]]
    max_delay = max(args.delays)

    policy = get_policy_class("pi05").from_pretrained(args.policy_path)
    policy.eval().to("cuda")
    device = next(policy.parameters()).device
    metadata = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet", root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = SharedObservationVLASHDataset(
        "lerobot/aloha_mobile_cabinet",
        root=args.dataset_root,
        episodes=validation_ids,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        max_delay_steps=max_delay,
    )

    sample_count = min(args.max_samples, len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, sample_count, dtype=int)).tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, float | int | str]] = []
    metrics: dict[int, list[dict[str, float]]] = defaultdict(list)

    for batch_number, batch in enumerate(dataloader):
        states = batch["observation.state"]
        targets = batch["action"]
        padding = batch["action_is_pad"]
        image_keys = [key for key in batch if key.startswith("observation.images.")]
        for delay in args.delays:
            state_index = 0 if args.mode == "sync" else delay
            model_batch = {key: batch[key].to(device) for key in image_keys}
            model_batch["observation.state"] = states[:, state_index].to(device)
            model_batch["task"] = batch["task"] if args.task is None else [args.task] * states.shape[0]
            torch.manual_seed(args.seed + batch_number * 97 + delay)
            torch.cuda.manual_seed_all(args.seed + batch_number * 97 + delay)
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(model_batch).detach().float().cpu()
            target = targets[:, delay].detach().float().cpu()
            valid = ~padding[:, delay].detach().cpu().bool()
            if prediction.shape != target.shape:
                raise RuntimeError(f"Prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
            first_mae, first_mse = mean_metric(prediction, target, valid, 1)
            first4_mae, first4_mse = mean_metric(prediction, target, valid, 4)
            chunk_mae, chunk_mse = mean_metric(prediction, target, valid, prediction.shape[0])
            for row_index in range(states.shape[0]):
                row = {
                    "mode": args.mode,
                    "dataset_index": int(indices[batch_number * args.batch_size + row_index]),
                    "episode_index": int(batch["episode_index"][row_index].item()),
                    "delay": delay,
                    "valid_action_steps": int(valid[row_index].sum()),
                    "first_action_mae": float((prediction[row_index, :1][valid[row_index, :1]] - target[row_index, :1][valid[row_index, :1]]).abs().mean()),
                    "first_action_mse": float((prediction[row_index, :1][valid[row_index, :1]] - target[row_index, :1][valid[row_index, :1]]).square().mean()),
                    "first4_mae": float((prediction[row_index, :4][valid[row_index, :4]] - target[row_index, :4][valid[row_index, :4]]).abs().mean()),
                    "first4_mse": float((prediction[row_index, :4][valid[row_index, :4]] - target[row_index, :4][valid[row_index, :4]]).square().mean()),
                    "chunk_mae": float((prediction[row_index][valid[row_index]] - target[row_index][valid[row_index]]).abs().mean()),
                    "chunk_mse": float((prediction[row_index][valid[row_index]] - target[row_index][valid[row_index]]).square().mean()),
                }
                rows.append(row)
                metrics[delay].append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for delay, delay_rows in sorted(metrics.items()):
        summary = {"mode": args.mode, "delay": delay, "samples": len(delay_rows)}
        for key in ("first_action_mae", "first_action_mse", "first4_mae", "first4_mse", "chunk_mae", "chunk_mse"):
            summary[key] = float(np.nanmean([row[key] for row in delay_rows]))
        summary_rows.append(summary)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
