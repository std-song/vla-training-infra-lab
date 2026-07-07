from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import statistics
import time

import torch
from torch.utils.data import DataLoader

from smolvla_nanotron.data.collator import VLABatch, collate_lerobot_lowdim
from smolvla_nanotron.data.lerobot_parquet_dataset import LeRobotParquetDataset


@dataclass(frozen=True)
class ProfileResult:
    include_images: bool
    batch_size: int
    workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool
    first_batch_sec: float
    steady_batches: int
    steady_frames: int
    steady_sec: float
    frames_per_sec: float
    mean_batch_sec: float
    p95_batch_sec: float
    cuda_max_allocated_mib: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LeRobot DataLoader throughput.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot/aloha_mobile_cabinet")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--profile-batches", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device-transfer", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_to_device(batch: VLABatch, device: torch.device) -> None:
    tensors = [
        batch.state,
        batch.effort,
        batch.action,
        batch.action_mask,
        batch.episode_index,
        batch.frame_index,
        batch.timestamp,
        batch.done,
        batch.task_index,
    ]
    for tensor in tensors:
        tensor.to(device, non_blocking=True)
    if batch.images is not None:
        for tensor in batch.images.values():
            tensor.to(device, non_blocking=True)


def make_loader(args: argparse.Namespace, workers: int) -> DataLoader:
    dataset = LeRobotParquetDataset(
        args.cache_root,
        repo_id=args.repo_id,
        limit=args.limit,
        start_index=args.start_index,
        include_images=args.include_images,
        image_size=args.image_size if args.include_images else None,
    )
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_lerobot_lowdim,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **kwargs)


def profile_one(args: argparse.Namespace, workers: int) -> ProfileResult:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader = make_loader(args, workers)
    iterator = iter(loader)

    first_start = time.perf_counter()
    first_batch = next(iterator)
    if args.device_transfer:
        move_to_device(first_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    first_batch_sec = time.perf_counter() - first_start

    for _ in range(args.warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if args.device_transfer:
            move_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    batch_times: list[float] = []
    frames = 0
    for _ in range(args.profile_batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if args.device_transfer:
            move_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)
        frames += batch.state.shape[0]

    steady_sec = sum(batch_times)
    mean_batch_sec = statistics.mean(batch_times) if batch_times else 0.0
    p95_batch_sec = sorted(batch_times)[min(len(batch_times) - 1, max(0, math.ceil(len(batch_times) * 0.95) - 1))] if batch_times else 0.0
    cuda_max = None
    if device.type == "cuda":
        cuda_max = torch.cuda.max_memory_allocated(device) / 1024**2

    return ProfileResult(
        include_images=args.include_images,
        batch_size=args.batch_size,
        workers=workers,
        prefetch_factor=args.prefetch_factor if workers > 0 else None,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if workers > 0 else False,
        first_batch_sec=first_batch_sec,
        steady_batches=len(batch_times),
        steady_frames=frames,
        steady_sec=steady_sec,
        frames_per_sec=(frames / steady_sec) if steady_sec else 0.0,
        mean_batch_sec=mean_batch_sec,
        p95_batch_sec=p95_batch_sec,
        cuda_max_allocated_mib=cuda_max,
    )


def print_table(results: list[ProfileResult]) -> None:
    print("| images | batch | workers | prefetch | pin | persistent | first_batch_s | steady_batches | fps | mean_batch_ms | p95_batch_ms | cuda_mib |")
    print("| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        cuda = "" if r.cuda_max_allocated_mib is None else f"{r.cuda_max_allocated_mib:.1f}"
        prefetch = "" if r.prefetch_factor is None else str(r.prefetch_factor)
        print(
            f"| {r.include_images} | {r.batch_size} | {r.workers} | {prefetch} | {r.pin_memory} | "
            f"{r.persistent_workers} | {r.first_batch_sec:.3f} | {r.steady_batches} | {r.frames_per_sec:.2f} | "
            f"{r.mean_batch_sec * 1000:.1f} | {r.p95_batch_sec * 1000:.1f} | {cuda} |"
        )


def main() -> None:
    args = parse_args()
    print("=== profiler config ===")
    print(vars(args))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    results = []
    for workers in args.workers:
        print(f"\n--- workers={workers} ---")
        result = profile_one(args, workers)
        results.append(result)
        print(result)

    print("\n=== markdown summary ===")
    print_table(results)


if __name__ == "__main__":
    main()
