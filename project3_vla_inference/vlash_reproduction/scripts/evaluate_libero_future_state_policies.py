"""Held-out action-alignment evaluation for LIBERO future-state conditioning."""

from __future__ import annotations

import argparse
import csv
import gc
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


METRICS = ("first_action_mae", "first_action_mse", "first4_mae", "first4_mse", "chunk_mae", "chunk_mse")


def sample_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    result = {}
    for name, count in (("first_action", 1), ("first4", 4), ("chunk", prediction.shape[0])):
        mask = valid[:count]
        delta = prediction[:count][mask] - target[:count][mask]
        result[f"{name}_mae"] = float(delta.abs().mean()) if delta.numel() else float("nan")
        result[f"{name}_mse"] = float(delta.square().mean()) if delta.numel() else float("nan")
    return result


def evaluate_mode(
    policy,
    policy_label: str,
    state_mode: str,
    result_label: str,
    dataset_root: str,
    validation_ids: list[int],
    indices: list[int],
    predictor_path: str | None,
    batch_size: int,
    delays: list[int],
    seed: int,
) -> list[dict[str, float | int | str]]:
    metadata = LeRobotDatasetMetadata("lerobot/libero", root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = SharedObservationVLASHDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=validation_ids,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        max_delay_steps=max(delays),
        future_state_mode=state_mode,
        future_state_predictor_path=predictor_path,
    )
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    device = next(policy.parameters()).device
    rows = []
    for batch_number, batch in enumerate(loader):
        image_keys = [key for key in batch if key.startswith("observation.images.")]
        for delay in delays:
            model_batch = {key: batch[key].to(device) for key in image_keys}
            model_batch["observation.state"] = batch["observation.state"][:, delay].to(device)
            model_batch["task"] = batch["task"]
            torch.manual_seed(seed + batch_number * 97 + delay)
            torch.cuda.manual_seed_all(seed + batch_number * 97 + delay)
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(model_batch).detach().float().cpu()
            target = batch["action"][:, delay].detach().float().cpu()
            valid = ~batch["action_is_pad"][:, delay].detach().cpu().bool()
            for row_index in range(prediction.shape[0]):
                rows.append({
                    "result": result_label,
                    "policy": policy_label,
                    "state_mode": state_mode,
                    "dataset_index": int(indices[batch_number * batch_size + row_index]),
                    "episode_index": int(batch["episode_index"][row_index].item()),
                    "delay": delay,
                    "delay_ms": delay * 100,
                    "valid_action_steps": int(valid[row_index].sum()),
                    **sample_metrics(prediction[row_index], target[row_index], valid[row_index]),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stale-policy", required=True)
    parser.add_argument("--learned-policy", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    validation_ids = [int(value) for value in split["validation_episode_ids"]]
    metadata = LeRobotDatasetMetadata("lerobot/libero", root=args.dataset_root)
    probe_policy = get_policy_class("pi05").from_pretrained(args.stale_policy)
    probe_delta = resolve_delta_timestamps(probe_policy.config, metadata)
    probe_dataset = SharedObservationVLASHDataset(
        "lerobot/libero", root=args.dataset_root, episodes=validation_ids,
        delta_timestamps=probe_delta, video_backend="torchcodec", max_delay_steps=max(args.delays),
        future_state_mode="stale",
    )
    sample_count = min(args.max_samples, len(probe_dataset))
    indices = np.unique(np.linspace(0, len(probe_dataset) - 1, sample_count, dtype=int)).tolist()
    del probe_policy, probe_dataset
    gc.collect()

    rows = []
    policy_specs = [
        (args.stale_policy, "stale_policy", [("stale", "stale_baseline", None)]),
        (
            args.learned_policy,
            "learned_policy",
            [
                ("learned", "learned_dynamics", args.predictor_path),
                ("oracle", "oracle_future_state", None),
                ("stale", "learned_policy_stale_input", None),
            ],
        ),
    ]
    for policy_path, policy_label, modes in policy_specs:
        policy = get_policy_class("pi05").from_pretrained(policy_path).eval().to("cuda")
        for state_mode, result_label, predictor_path in modes:
            rows.extend(evaluate_mode(
                policy, policy_label, state_mode, result_label, args.dataset_root,
                validation_ids, indices, predictor_path, args.batch_size, args.delays, args.seed,
            ))
        del policy
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["result"]), int(row["delay"]))].append(row)
    summary_rows = []
    for (result, delay), values in sorted(grouped.items()):
        summary = {"result": result, "delay": delay, "delay_ms": delay * 100, "samples": len(values)}
        for metric in METRICS:
            summary[metric] = float(np.nanmean([row[metric] for row in values]))
        summary_rows.append(summary)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
