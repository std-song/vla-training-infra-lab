"""Compare endpoint, learned, and oracle future-state inputs on one VLASH policy."""

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

from future_state_predictor import ResidualStatePredictor, predict_future_state


DEFAULT_TASK = "Open the top cabinet, store the pot inside it then close the cabinet."
METRIC_KEYS = ("first_action_mae", "first_action_mse", "first4_mae", "first4_mse", "chunk_mae", "chunk_mse")


def sample_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    result = {}
    for name, count in (("first_action", 1), ("first4", 4), ("chunk", prediction.shape[0])):
        mask = valid[:count]
        delta = prediction[:count][mask] - target[:count][mask]
        result[f"{name}_mae"] = float(delta.abs().mean()) if delta.numel() else float("nan")
        result[f"{name}_mse"] = float(delta.square().mean()) if delta.numel() else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    validation_ids = [int(value) for value in split["validation_episode_ids"]]
    max_delay = max(args.delays)
    device = torch.device("cuda")

    policy = get_policy_class("pi05").from_pretrained(args.policy_path).eval().to(device)
    checkpoint = torch.load(args.predictor_path, map_location=device, weights_only=False)
    predictor = ResidualStatePredictor(**checkpoint["config"]).to(device)
    predictor.load_state_dict(checkpoint["model"])
    predictor.eval()
    stats = checkpoint["stats"]

    metadata = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet", root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    common = dict(
        repo_id="lerobot/aloha_mobile_cabinet", root=args.dataset_root, episodes=validation_ids,
        delta_timestamps=delta_timestamps, video_backend="torchcodec", max_delay_steps=max_delay,
    )
    proxy_dataset = SharedObservationVLASHDataset(**common, use_state_ground_truth=False)
    oracle_dataset = SharedObservationVLASHDataset(**common, use_state_ground_truth=True)
    sample_count = min(args.max_samples, len(proxy_dataset))
    indices = np.unique(np.linspace(0, len(proxy_dataset) - 1, sample_count, dtype=int)).tolist()
    proxy_loader = DataLoader(torch.utils.data.Subset(proxy_dataset, indices), batch_size=args.batch_size, num_workers=0)
    oracle_loader = DataLoader(torch.utils.data.Subset(oracle_dataset, indices), batch_size=args.batch_size, num_workers=0)

    rows = []
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for batch_number, (batch, oracle_batch) in enumerate(zip(proxy_loader, oracle_loader)):
        image_keys = [key for key in batch if key.startswith("observation.images.")]
        current_state = batch["observation.state"][:, 0].to(device)
        current_actions = batch["action"][:, 0, :max_delay].to(device)
        for delay in args.delays:
            horizon = torch.full((current_state.shape[0],), delay, dtype=torch.long, device=device)
            learned_state = predict_future_state(predictor, current_state, current_actions, horizon, stats)
            state_inputs = {
                "endpoint_proxy": batch["observation.state"][:, delay].to(device),
                "learned_proxy": learned_state,
                "oracle_state": oracle_batch["observation.state"][:, delay].to(device),
            }
            target = batch["action"][:, delay].detach().float().cpu()
            valid = ~batch["action_is_pad"][:, delay].detach().cpu().bool()
            for mode, state_input in state_inputs.items():
                model_batch = {key: batch[key].to(device) for key in image_keys}
                model_batch["observation.state"] = state_input
                model_batch["task"] = [args.task] * current_state.shape[0]
                torch.manual_seed(args.seed + batch_number * 97 + delay)
                torch.cuda.manual_seed_all(args.seed + batch_number * 97 + delay)
                with torch.inference_mode():
                    prediction = policy.predict_action_chunk(model_batch).detach().float().cpu()
                for row_index in range(current_state.shape[0]):
                    row = {
                        "mode": mode,
                        "dataset_index": int(indices[batch_number * args.batch_size + row_index]),
                        "episode_index": int(batch["episode_index"][row_index].item()),
                        "delay": delay,
                        "valid_action_steps": int(valid[row_index].sum()),
                        **sample_metrics(prediction[row_index], target[row_index], valid[row_index]),
                    }
                    rows.append(row)
                    grouped[(mode, delay)].append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for (mode, delay), values in sorted(grouped.items()):
        summary = {"mode": mode, "delay": delay, "samples": len(values)}
        for key in METRIC_KEYS:
            summary[key] = float(np.nanmean([value[key] for value in values]))
        summary_rows.append(summary)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
