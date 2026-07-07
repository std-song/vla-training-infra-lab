from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from smolvla_nanotron.data.collator import collate_lerobot_lowdim, describe_batch
from smolvla_nanotron.data.lerobot_parquet_dataset import LeRobotParquetDataset
from smolvla_nanotron.models.tiny_policy import TinyLowDimVLAPolicy, masked_mse_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run LeRobot low-dimensional VLA batches.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/aloha_mobile_cabinet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LeRobotParquetDataset(args.cache_root, repo_id=args.repo_id, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lerobot_lowdim)
    batch = next(iter(loader))

    print("=== dataset ===")
    print("repo_id:", dataset.schema.repo_id)
    print("snapshot:", dataset.paths.snapshot_dir)
    print("data:", dataset.paths.data_path)
    print("frames loaded:", len(dataset))
    print("fps:", dataset.schema.fps)
    print("robot_type:", dataset.schema.robot_type)
    print("features:")
    for name, feature in dataset.schema.features.items():
        print(f"  {name}: dtype={feature.get('dtype')} shape={feature.get('shape')}")

    print("\n=== first batch ===")
    print(describe_batch(batch))

    device = torch.device(args.device)
    policy = TinyLowDimVLAPolicy(state_dim=batch.state.shape[-1], action_dim=batch.action.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    print("\n=== tiny policy smoke train ===")
    start = time.perf_counter()
    steps = 0
    for step, train_batch in enumerate(loader, start=1):
        if step > args.train_steps:
            break
        state = train_batch.state.to(device)
        action = train_batch.action.to(device)
        mask = train_batch.action_mask.to(device)
        prediction = policy(state)
        loss = masked_mse_loss(prediction, action, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        print(f"step={step} loss={loss.item():.6f}")

    elapsed = time.perf_counter() - start
    frames = steps * args.batch_size
    print("\n=== summary ===")
    print(f"steps={steps}")
    print(f"frames={frames}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"frames_per_sec={frames / elapsed:.2f}")
    if device.type == "cuda":
        print(f"cuda_max_allocated_mib={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")


if __name__ == "__main__":
    main()
