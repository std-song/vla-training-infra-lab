from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _action_postprocess_kernel(pred, mean, std, low, high, mask, prev, out, total: tl.constexpr, action_dim: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < total
    action_idx = offsets % action_dim
    x = tl.load(pred + offsets, mask=valid, other=0.0)
    m = tl.load(mean + action_idx, mask=valid, other=0.0)
    s = tl.load(std + action_idx, mask=valid, other=1.0)
    lo = tl.load(low + action_idx, mask=valid, other=-3.4e38)
    hi = tl.load(high + action_idx, mask=valid, other=3.4e38)
    keep = tl.load(mask + offsets, mask=valid, other=0).to(tl.int1)
    old = tl.load(prev + offsets, mask=valid, other=0.0)
    y = x * s + m
    y = tl.minimum(tl.maximum(y, lo), hi)
    y = tl.where(keep, y, old)
    tl.store(out + offsets, y, mask=valid)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a simplified VLA action head and Triton fused action post-processing kernel.")
    parser.add_argument("--out", default="project3_vla_infer/results/vla_action_head_triton.csv")
    parser.add_argument("--batch-sizes", default="1,4,16,64,256")
    parser.add_argument("--horizons", default="10,32,64")
    parser.add_argument("--action-dims", default="14,64")
    parser.add_argument("--hidden-dim", type=int, default=896)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def cuda_event_ms(fn, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeat


def torch_postprocess(pred, mean, std, low, high, mask, prev):
    y = pred * std.view(1, 1, -1) + mean.view(1, 1, -1)
    y = torch.clamp(y, min=low.view(1, 1, -1), max=high.view(1, 1, -1))
    return torch.where(mask, y, prev)


def triton_postprocess(pred, mean, std, low, high, mask, prev, out, action_dim: int):
    total = pred.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _action_postprocess_kernel[grid](pred, mean, std, low, high, mask, prev, out, total, action_dim, block=block)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    rows = []

    for batch in parse_ints(args.batch_sizes):
        for horizon in parse_ints(args.horizons):
            for action_dim in parse_ints(args.action_dims):
                head = nn.Sequential(
                    nn.Linear(args.hidden_dim, args.hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(args.hidden_dim * 2, horizon * action_dim),
                ).to(device=device, dtype=dtype).eval()
                hidden = torch.randn(batch, args.hidden_dim, device=device, dtype=dtype)

                with torch.inference_mode():
                    action_head_ms = cuda_event_ms(lambda: head(hidden), repeat=args.repeat, warmup=args.warmup)
                    pred = head(hidden).view(batch, horizon, action_dim).contiguous()

                mean = torch.randn(action_dim, device=device, dtype=dtype)
                std = torch.rand(action_dim, device=device, dtype=dtype) + 0.5
                low = torch.full((action_dim,), -2.0, device=device, dtype=dtype)
                high = torch.full((action_dim,), 2.0, device=device, dtype=dtype)
                mask = torch.rand(batch, horizon, action_dim, device=device) > 0.1
                prev = torch.randn(batch, horizon, action_dim, device=device, dtype=dtype)
                out = torch.empty_like(pred)

                with torch.inference_mode():
                    torch_y = torch_postprocess(pred, mean, std, low, high, mask, prev)
                    triton_y = triton_postprocess(pred, mean, std, low, high, mask, prev, out, action_dim)
                    torch.cuda.synchronize()
                    max_err = (torch_y.float() - triton_y.float()).abs().max().item()
                    torch_ms = cuda_event_ms(lambda: torch_postprocess(pred, mean, std, low, high, mask, prev), repeat=args.repeat, warmup=args.warmup)
                    triton_ms = cuda_event_ms(lambda: triton_postprocess(pred, mean, std, low, high, mask, prev, out, action_dim), repeat=args.repeat, warmup=args.warmup)

                elements = batch * horizon * action_dim
                row = {
                    "dtype": args.dtype,
                    "batch_size": batch,
                    "horizon": horizon,
                    "action_dim": action_dim,
                    "elements": elements,
                    "hidden_dim": args.hidden_dim,
                    "action_head_ms": action_head_ms,
                    "torch_postprocess_us": torch_ms * 1000.0,
                    "triton_postprocess_us": triton_ms * 1000.0,
                    "triton_speedup": torch_ms / triton_ms if triton_ms > 0 else float("nan"),
                    "max_error": max_err,
                }
                rows.append(row)
                print(
                    f"B={batch} H={horizon} A={action_dim} elems={elements} "
                    f"head={action_head_ms:.4f}ms torch={row['torch_postprocess_us']:.2f}us "
                    f"triton={row['triton_postprocess_us']:.2f}us speedup={row['triton_speedup']:.2f}x err={max_err:.2e}",
                    flush=True,
                )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dtype",
        "batch_size",
        "horizon",
        "action_dim",
        "elements",
        "hidden_dim",
        "action_head_ms",
        "torch_postprocess_us",
        "triton_postprocess_us",
        "triton_speedup",
        "max_error",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
