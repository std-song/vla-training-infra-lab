"""Train and evaluate a sequence-conditioned LIBERO future-state predictor."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from libero_state_predictor import ActionSequenceStatePredictor, predict_future_state


def load_dataset(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    table = pa.concat_tables([
        pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        for path in files
    ])
    states = torch.from_numpy(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
    actions = torch.from_numpy(np.asarray(table["action"].to_pylist(), dtype=np.float32))
    episodes = torch.from_numpy(np.asarray(table["episode_index"].to_numpy(), dtype=np.int64))
    return states, actions, episodes


def valid_starts(episodes: torch.Tensor, episode_ids: list[int], max_horizon: int) -> torch.Tensor:
    starts = []
    for episode_id in episode_ids:
        indices = torch.nonzero(episodes == episode_id, as_tuple=False).flatten()
        if len(indices) > max_horizon:
            starts.append(indices[:-max_horizon])
    return torch.cat(starts)


def gather(
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    horizons: torch.Tensor,
    max_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.arange(max_horizon, device=starts.device).unsqueeze(0)
    action_prefix = actions[starts.unsqueeze(1) + offsets]
    return states[starts], action_prefix, states[starts + horizons]


@torch.inference_mode()
def evaluate(
    model: ActionSequenceStatePredictor,
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    delays: list[int],
    stats: dict[str, torch.Tensor],
    batch_size: int,
) -> list[dict[str, float | int]]:
    model.eval()
    rows = []
    for delay in delays:
        stale_sse = learned_sse = stale_sae = learned_sae = 0.0
        count = 0
        for begin in range(0, len(starts), batch_size):
            selected = starts[begin : begin + batch_size]
            horizons = torch.full((len(selected),), delay, dtype=torch.long, device=starts.device)
            current, action_prefix, target = gather(states, actions, selected, horizons, model.max_horizon)
            learned = predict_future_state(model, current, action_prefix, horizons, stats)
            stale_delta = current - target
            learned_delta = learned - target
            stale_sse += stale_delta.square().sum().item()
            learned_sse += learned_delta.square().sum().item()
            stale_sae += stale_delta.abs().sum().item()
            learned_sae += learned_delta.abs().sum().item()
            count += target.numel()
        rows.append({
            "delay": delay,
            "delay_ms": delay * 100,
            "samples": len(starts),
            "stale_state_mae": stale_sae / count,
            "stale_state_mse": stale_sse / count,
            "learned_state_mae": learned_sae / count,
            "learned_state_mse": learned_sse / count,
            "mse_reduction_percent": 100 * (stale_sse - learned_sse) / stale_sse,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    states_cpu, actions_cpu, episodes_cpu = load_dataset(args.dataset_root)
    train_mask = torch.isin(episodes_cpu, torch.tensor(split["train_episode_ids"]))
    stats = {
        "state_mean": states_cpu[train_mask].mean(0).to(device),
        "state_std": states_cpu[train_mask].std(0).clamp_min(1e-6).to(device),
        "action_mean": actions_cpu[train_mask].mean(0).to(device),
        "action_std": actions_cpu[train_mask].std(0).clamp_min(1e-6).to(device),
    }
    states, actions = states_cpu.to(device), actions_cpu.to(device)
    train_starts = valid_starts(episodes_cpu, split["train_episode_ids"], args.max_horizon).to(device)
    validation_starts = valid_starts(episodes_cpu, split["validation_episode_ids"], args.max_horizon).to(device)
    model = ActionSequenceStatePredictor(
        states.shape[1], actions.shape[1], args.max_horizon, args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    log_rows = []

    for step in range(1, args.steps + 1):
        model.train()
        starts = train_starts[torch.randint(len(train_starts), (args.batch_size,), device=device)]
        horizons = torch.randint(1, args.max_horizon + 1, (args.batch_size,), device=device)
        current, action_prefix, target = gather(states, actions, starts, horizons, args.max_horizon)
        state_normalized = (current - stats["state_mean"]) / stats["state_std"]
        actions_normalized = (action_prefix - stats["action_mean"]) / stats["action_std"]
        target_normalized = (target - stats["state_mean"]) / stats["state_std"]
        prediction = model(state_normalized, actions_normalized, horizons)
        loss = F.smooth_l1_loss(prediction, target_normalized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            metrics = evaluate(
                model, states, actions, validation_starts, [1, 2, 4], stats, args.batch_size,
            )
            validation_mse = float(np.mean([row["learned_state_mse"] for row in metrics]))
            row = {
                "step": step,
                "train_loss": float(loss),
                "grad_norm": float(grad_norm),
                "validation_state_mse": validation_mse,
            }
            log_rows.append(row)
            print(json.dumps(row), flush=True)
            if validation_mse < best_validation:
                best_validation = validation_mse
                torch.save({
                    "model": model.state_dict(),
                    "stats": {key: value.cpu() for key, value in stats.items()},
                    "config": {
                        "state_dim": states.shape[1], "action_dim": actions.shape[1],
                        "max_horizon": args.max_horizon, "hidden_dim": args.hidden_dim,
                    },
                    "step": step,
                    "validation_state_mse": validation_mse,
                }, args.out_dir / "best.pt")

    checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_rows = evaluate(model, states, actions, validation_starts, [1, 2, 4], stats, args.batch_size)
    with (args.out_dir / "state_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    with (args.out_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    print(json.dumps(final_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
