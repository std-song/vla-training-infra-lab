from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from smolvla_nanotron.data.collator import VLABatch, collate_lerobot_lowdim, describe_batch
from smolvla_nanotron.data.lerobot_parquet_dataset import LeRobotParquetDataset
from smolvla_nanotron.models.smolvla_compatible import SmolVLACompatiblePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nanotron-style DP/DDP train for the SmolVLA-compatible policy wrapper.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/aloha_mobile_cabinet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank batch size.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2, help="Per-rank dataloader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--backend", default=None, choices=["nccl", "gloo", None])
    return parser.parse_args()


def distributed_env() -> tuple[bool, int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1, rank, local_rank, world_size


def init_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    is_distributed, rank, local_rank, world_size = distributed_env()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = args.backend or "gloo"

    if is_distributed:
        dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def log_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


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


def make_loader(args: argparse.Namespace, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
    dataset = LeRobotParquetDataset(
        args.cache_root,
        repo_id=args.repo_id,
        limit=args.limit,
        start_index=args.start_index,
        include_images=True,
        image_size=args.image_size,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_lerobot_lowdim,
        "drop_last": True,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **kwargs), sampler


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    world_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "world_size": world_size,
            "args": _serializable_args(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / world_size
    return value


def main() -> None:
    args = parse_args()
    device, rank, local_rank, world_size = init_distributed(args)
    args.output_dir.mkdir(parents=True, exist_ok=True) if rank == 0 else None

    try:
        loader, sampler = make_loader(args, rank, world_size)
        first_batch = next(iter(loader))
        log_rank0(rank, "=== first batch ===")
        log_rank0(rank, describe_batch(first_batch))
        log_rank0(rank, f"distributed_backend={dist.get_backend() if dist.is_initialized() else 'single'}")
        log_rank0(rank, f"world_size={world_size} per_rank_batch_size={args.batch_size} global_batch_size={args.batch_size * world_size}")

        model = SmolVLACompatiblePolicy(state_dim=first_batch.state.shape[-1], action_dim=first_batch.action.shape[-1]).to(device)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        start_step = 0
        if args.resume_from is not None:
            start_step = load_checkpoint(args.resume_from, model, optimizer, device)
            log_rank0(rank, f"resumed_from={args.resume_from} step={start_step}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        global_step = start_step
        start_time = time.perf_counter()
        iterator = iter(loader)
        epoch = 0
        while global_step < args.train_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)

            batch = move_batch_to_device(batch, device)
            loss, prediction = model.loss(batch) if not isinstance(model, DistributedDataParallel) else model.module.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            global_step += 1
            metrics = torch.tensor(
                [loss.detach().item(), float(grad_norm), prediction.detach().mean().item(), batch.action.mean().item()],
                dtype=torch.float32,
                device=device,
            )
            metrics = reduce_mean(metrics, world_size)
            if rank == 0:
                print(
                    f"step={global_step} loss={metrics[0].item():.6f} grad_norm={metrics[1].item():.4f} "
                    f"pred_mean={metrics[2].item():.4f} target_mean={metrics[3].item():.4f}",
                    flush=True,
                )

            if rank == 0 and args.save_every and global_step % args.save_every == 0:
                ckpt = args.output_dir / f"step-{global_step}.pt"
                save_checkpoint(ckpt, model, optimizer, global_step, args, world_size)
                print(f"saved={ckpt}", flush=True)

        elapsed = time.perf_counter() - start_time
        local_steps = global_step - start_step
        local_samples = torch.tensor([local_steps * args.batch_size, elapsed], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_samples, op=dist.ReduceOp.SUM)
        global_samples = float(local_samples[0].item())
        max_elapsed = elapsed
        if dist.is_available() and dist.is_initialized():
            elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            max_elapsed = float(elapsed_tensor.item())

        if rank == 0:
            final_ckpt = args.output_dir / "latest.pt"
            save_checkpoint(final_ckpt, model, optimizer, global_step, args, world_size)
            print("=== summary ===", flush=True)
            print(f"steps={local_steps}", flush=True)
            print(f"final_step={global_step}", flush=True)
            print(f"world_size={world_size}", flush=True)
            print(f"global_batch_size={args.batch_size * world_size}", flush=True)
            print(f"elapsed_sec={max_elapsed:.3f}", flush=True)
            print(f"steps_per_sec={local_steps / max_elapsed:.3f}", flush=True)
            print(f"samples_per_sec={global_samples / max_elapsed:.3f}", flush=True)
            print(f"checkpoint={final_ckpt}", flush=True)

        if device.type == "cuda":
            memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 1024**2], dtype=torch.float64, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            log_rank0(rank, f"cuda_max_allocated_mib_per_rank_max={memory.item():.1f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
