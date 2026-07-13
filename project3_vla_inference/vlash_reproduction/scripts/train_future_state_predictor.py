"""Train a residual action-to-state predictor on the existing ALOHA episode split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from future_state_predictor import ResidualStatePredictor, predict_future_state


def load_dataset(parquet_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = pq.read_table(parquet_path, columns=["observation.state", "action", "episode_index"])
    states = torch.from_numpy(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
    actions = torch.from_numpy(np.asarray(table["action"].to_pylist(), dtype=np.float32))
    episodes = torch.from_numpy(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1))
    return states, actions, episodes


def valid_starts(episodes: torch.Tensor, episode_ids: list[int], max_horizon: int) -> torch.Tensor:
    starts = []
    for episode_id in episode_ids:
        indices = torch.nonzero(episodes == episode_id, as_tuple=False).flatten()
        if len(indices) > max_horizon:
            starts.append(indices[:-max_horizon])
    return torch.cat(starts)


def gather_examples(
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    horizons: torch.Tensor,
    max_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.arange(max_horizon, device=starts.device).unsqueeze(0)
    action_prefix = actions[starts.unsqueeze(1) + offsets]
    future_state = states[starts + horizons]
    return states[starts], action_prefix, future_state


@torch.inference_mode()
def evaluate(
    model: ResidualStatePredictor,
    states: torch.Tensor,
    actions: torch.Tensor,
    starts: torch.Tensor,
    horizons: list[int],
    stats: dict[str, torch.Tensor],
    max_horizon: int,
    batch_size: int,
) -> list[dict[str, float | int]]:
    model.eval()
    rows = []
    for delay in horizons:
        proxy_sse = learned_sse = proxy_sae = learned_sae = 0.0
        count = 0
        for begin in range(0, len(starts), batch_size):
            batch_starts = starts[begin : begin + batch_size]
            horizon = torch.full((len(batch_starts),), delay, device=starts.device, dtype=torch.long)
            current, prefix, target = gather_examples(states, actions, batch_starts, horizon, max_horizon)
            proxy = prefix[:, delay - 1]
            learned = predict_future_state(model, current, prefix, horizon, stats)
            proxy_delta = proxy - target
            learned_delta = learned - target
            proxy_sse += proxy_delta.square().sum().item()
            learned_sse += learned_delta.square().sum().item()
            proxy_sae += proxy_delta.abs().sum().item()
            learned_sae += learned_delta.abs().sum().item()
            count += target.numel()
        rows.append(
            {
                "delay": delay,
                "samples": len(starts),
                "proxy_state_mae": proxy_sae / count,
                "proxy_state_mse": proxy_sse / count,
                "learned_state_mae": learned_sae / count,
                "learned_state_mse": learned_sse / count,
                "mse_reduction_percent": 100.0 * (proxy_sse - learned_sse) / proxy_sse,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
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
    states_cpu, actions_cpu, episodes_cpu = load_dataset(args.parquet)
    train_mask = torch.isin(episodes_cpu, torch.tensor(split["train_episode_ids"]))
    state_mean = states_cpu[train_mask].mean(0)
    state_std = states_cpu[train_mask].std(0).clamp_min(1e-6)
    action_mean = actions_cpu[train_mask].mean(0)
    action_std = actions_cpu[train_mask].std(0).clamp_min(1e-6)
    stats = {
        "state_mean": state_mean.to(device),
        "state_std": state_std.to(device),
        "action_mean": action_mean.to(device),
        "action_std": action_std.to(device),
    }
    states = states_cpu.to(device)
    actions = actions_cpu.to(device)
    train_starts = valid_starts(episodes_cpu, split["train_episode_ids"], args.max_horizon).to(device)
    validation_starts = valid_starts(episodes_cpu, split["validation_episode_ids"], args.max_horizon).to(device)

    model = ResidualStatePredictor(states.shape[1], actions.shape[1], args.max_horizon, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_validation = float("inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []

    for step in range(1, args.steps + 1):
        model.train()
        selected = train_starts[torch.randint(len(train_starts), (args.batch_size,), device=device)]
        horizons = torch.randint(1, args.max_horizon + 1, (args.batch_size,), device=device)
        current, prefix, target = gather_examples(states, actions, selected, horizons, args.max_horizon)
        state_normalized = (current - stats["state_mean"]) / stats["state_std"]
        action_normalized = (prefix - stats["action_mean"]) / stats["action_std"]
        endpoint = prefix[torch.arange(args.batch_size, device=device), horizons - 1]
        endpoint_as_state = (endpoint - stats["state_mean"]) / stats["state_std"]
        target_residual = (target - stats["state_mean"]) / stats["state_std"] - endpoint_as_state
        prediction = model(state_normalized, action_normalized, horizons)
        loss = F.smooth_l1_loss(prediction, target_residual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            validation_rows = evaluate(
                model, states, actions, validation_starts, [1, 2, 4, 8], stats,
                args.max_horizon, args.batch_size,
            )
            validation_mse = float(np.mean([row["learned_state_mse"] for row in validation_rows]))
            row = {"step": step, "train_loss": float(loss), "validation_state_mse": validation_mse}
            log_rows.append(row)
            print(json.dumps(row), flush=True)
            if validation_mse < best_validation:
                best_validation = validation_mse
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stats": {key: value.cpu() for key, value in stats.items()},
                        "config": {
                            "state_dim": states.shape[1], "action_dim": actions.shape[1],
                            "max_horizon": args.max_horizon, "hidden_dim": args.hidden_dim,
                        },
                        "step": step,
                        "validation_state_mse": validation_mse,
                    },
                    args.out_dir / "best.pt",
                )

    checkpoint = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_rows = evaluate(
        model, states, actions, validation_starts, [1, 2, 4, 8], stats,
        args.max_horizon, args.batch_size,
    )
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
