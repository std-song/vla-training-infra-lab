from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from smolvla_nanotron.data.collator import VLABatch, collate_lerobot_lowdim, describe_batch
from smolvla_nanotron.data.lerobot_parquet_dataset import LeRobotParquetDataset
from smolvla_nanotron.models.smolvla_compatible import SmolVLACompatiblePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight SmolVLA-compatible policy wrapper.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/aloha_mobile_cabinet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch_to_device(batch: VLABatch, device: torch.device) -> VLABatch:
    images = None
    if batch.images is not None:
        images = {camera: tensor.to(device, non_blocking=True) for camera, tensor in batch.images.items()}
    return VLABatch(
        state=batch.state.to(device, non_blocking=True),
        effort=batch.effort.to(device, non_blocking=True),
        action=batch.action.to(device, non_blocking=True),
        action_mask=batch.action_mask.to(device, non_blocking=True),
        episode_index=batch.episode_index.to(device, non_blocking=True),
        frame_index=batch.frame_index.to(device, non_blocking=True),
        timestamp=batch.timestamp.to(device, non_blocking=True),
        done=batch.done.to(device, non_blocking=True),
        task_index=batch.task_index.to(device, non_blocking=True),
        task_text=batch.task_text,
        images=images,
    )


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = LeRobotParquetDataset(
        args.cache_root,
        repo_id=args.repo_id,
        limit=args.limit,
        start_index=args.start_index,
        include_images=True,
        image_size=args.image_size,
    )
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_lerobot_lowdim,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **kwargs)


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": _serializable_args(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args)
    first_batch = next(iter(loader))
    print("=== first batch ===")
    print(describe_batch(first_batch))

    model = SmolVLACompatiblePolicy(state_dim=first_batch.state.shape[-1], action_dim=first_batch.action.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer, device)
        print(f"resumed_from={args.resume_from} step={start_step}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    global_step = start_step
    start_time = time.perf_counter()
    iterator = iter(loader)
    while global_step < args.train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)
        loss, prediction = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1
        print(
            f"step={global_step} loss={loss.item():.6f} grad_norm={float(grad_norm):.4f} "
            f"pred_mean={prediction.detach().mean().item():.4f} target_mean={batch.action.mean().item():.4f}"
        )

        if args.save_every and global_step % args.save_every == 0:
            ckpt = args.output_dir / f"step-{global_step}.pt"
            save_checkpoint(ckpt, model, optimizer, global_step, args)
            print(f"saved={ckpt}")

    elapsed = time.perf_counter() - start_time
    final_ckpt = args.output_dir / "latest.pt"
    save_checkpoint(final_ckpt, model, optimizer, global_step, args)
    print("=== summary ===")
    print(f"steps={global_step - start_step}")
    print(f"final_step={global_step}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"steps_per_sec={(global_step - start_step) / elapsed:.3f}")
    print(f"checkpoint={final_ckpt}")
    if device.type == "cuda":
        print(f"cuda_max_allocated_mib={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")


if __name__ == "__main__":
    main()
