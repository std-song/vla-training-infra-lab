from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "project3_vla_infer" / "results"


@dataclass(frozen=True)
class Profile:
    name: str
    image_count: int
    image_size: int
    prompt_tokens: int
    visual_tokens: int
    prefill_ms: float
    tpot_ms: float


@dataclass
class Request:
    request_id: int
    profile: Profile
    arrival_ms: float
    decode_tokens: int
    remaining_decode: int
    start_ms: float | None = None
    finish_ms: float | None = None
    allocated_blocks: list[int] = field(default_factory=list)
    allocated_tokens: int = 0


class PagedKVBlockManager:
    def __init__(self, block_size: int, total_blocks: int) -> None:
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.owners: dict[int, int] = {}
        self.used_tokens: dict[int, int] = {}
        self.peak_allocated = 0
        self.util_samples: list[float] = []
        self.allocated_samples: list[int] = []

    @property
    def allocated_blocks(self) -> int:
        return len(self.owners)

    def can_allocate(self, blocks: int) -> bool:
        return len(self.free_blocks) >= blocks

    def allocate_blocks(self, request: Request, tokens: int) -> bool:
        blocks = math.ceil(tokens / self.block_size)
        if not self.can_allocate(blocks):
            return False
        remaining = tokens
        for _ in range(blocks):
            block = self.free_blocks.pop()
            used = min(self.block_size, remaining)
            remaining -= used
            self.owners[block] = request.request_id
            self.used_tokens[block] = used
            request.allocated_blocks.append(block)
        request.allocated_tokens += tokens
        self.peak_allocated = max(self.peak_allocated, self.allocated_blocks)
        return True

    def append_token(self, request: Request) -> bool:
        if not request.allocated_blocks:
            return self.allocate_blocks(request, 1)
        last = request.allocated_blocks[-1]
        if self.used_tokens[last] < self.block_size:
            self.used_tokens[last] += 1
            request.allocated_tokens += 1
            return True
        return self.allocate_blocks(request, 1)

    def free(self, request: Request) -> None:
        for block in request.allocated_blocks:
            self.owners.pop(block, None)
            self.used_tokens.pop(block, None)
            self.free_blocks.append(block)
        request.allocated_blocks.clear()
        request.allocated_tokens = 0

    def sample(self) -> None:
        if not self.owners:
            self.util_samples.append(1.0)
            self.allocated_samples.append(0)
            return
        used = sum(self.used_tokens.values())
        capacity = len(self.owners) * self.block_size
        self.util_samples.append(used / capacity)
        self.allocated_samples.append(self.allocated_blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paged KV and continuous batching simulator for VLA serving workloads.")
    parser.add_argument("--visual-csv", default="project3_vla_infer/results/qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv")
    parser.add_argument("--out", default="project3_vla_infer/results/qwen25vl_paged_kv_continuous_batching.csv")
    parser.add_argument("--request-count", type=int, default=128)
    parser.add_argument("--mean-arrival-ms", type=float, default=90.0)
    parser.add_argument("--decode-options", default="16,32,64")
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-budget-mib", type=float, default=512.0)
    parser.add_argument("--bytes-per-token-mib", type=float, default=0.03515625)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_profiles(path: Path) -> list[Profile]:
    profiles: list[Profile] = []
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["decode_len"] != "64":
                continue
            key = (int(row["image_count"]), int(row["image_size"]))
            if key in seen:
                continue
            seen.add(key)
            profiles.append(
                Profile(
                    name=f'{row["image_count"]}x{row["image_size"]}',
                    image_count=int(row["image_count"]),
                    image_size=int(row["image_size"]),
                    prompt_tokens=int(row["input_tokens"]),
                    visual_tokens=int(row["visual_marker_tokens"]),
                    prefill_ms=float(row["prefill_ms"]),
                    tpot_ms=float(row["tpot_ms"]),
                )
            )
    if not profiles:
        raise ValueError(f"No decode_len=64 profiles found in {path}")
    return profiles


def make_workload(profiles: list[Profile], request_count: int, mean_arrival_ms: float, decode_options: list[int], seed: int) -> list[Request]:
    rng = random.Random(seed)
    # Bias toward multi-camera VLA requests while keeping mixed shapes.
    weights = [1.0 if p.image_count == 1 else 2.2 for p in profiles]
    now = 0.0
    requests = []
    for idx in range(request_count):
        now += rng.expovariate(1.0 / mean_arrival_ms)
        profile = rng.choices(profiles, weights=weights, k=1)[0]
        decode_tokens = rng.choice(decode_options)
        requests.append(Request(idx, profile, now, decode_tokens, decode_tokens))
    return requests


def clone_workload(requests: list[Request]) -> list[Request]:
    return [Request(r.request_id, r.profile, r.arrival_ms, r.decode_tokens, r.decode_tokens) for r in requests]


def request_service_ms(req: Request) -> float:
    return req.profile.prefill_ms + req.profile.tpot_ms * req.decode_tokens


def summarize(name: str, requests: list[Request], end_ms: float, manager: PagedKVBlockManager | None, bytes_per_token_mib: float, block_size: int) -> dict[str, float | str | int]:
    latencies = [r.finish_ms - r.arrival_ms for r in requests if r.finish_ms is not None]
    if len(latencies) != len(requests):
        raise RuntimeError(f"{name} did not finish every request")
    peak_blocks = manager.peak_allocated if manager else 0
    avg_util = statistics.mean(manager.util_samples) if manager and manager.util_samples else 1.0
    avg_blocks = statistics.mean(manager.allocated_samples) if manager and manager.allocated_samples else 0.0
    peak_kv_mib = peak_blocks * block_size * bytes_per_token_mib
    return {
        "scenario": name,
        "request_count": len(requests),
        "makespan_ms": end_ms,
        "throughput_req_s": len(requests) / (end_ms / 1000.0),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)],
        "peak_blocks": peak_blocks,
        "avg_allocated_blocks": avg_blocks,
        "avg_block_utilization": avg_util,
        "peak_kv_mib": peak_kv_mib,
    }


def run_serial(requests: list[Request], args: argparse.Namespace) -> dict[str, float | str | int]:
    total_blocks = math.floor(args.kv_budget_mib / (args.block_size * args.bytes_per_token_mib))
    manager = PagedKVBlockManager(args.block_size, total_blocks)
    now = 0.0
    for req in requests:
        now = max(now, req.arrival_ms)
        req.start_ms = now
        tokens = req.profile.prompt_tokens + req.decode_tokens
        if not manager.allocate_blocks(req, tokens):
            raise RuntimeError("serial request does not fit KV budget")
        now += request_service_ms(req)
        req.finish_ms = now
        manager.sample()
        manager.free(req)
    return summarize("serial_no_batch", requests, now, manager, args.bytes_per_token_mib, args.block_size)


def run_continuous(requests: list[Request], args: argparse.Namespace, paged: bool, guarded: bool = False) -> dict[str, float | str | int]:
    total_blocks = math.floor(args.kv_budget_mib / (args.block_size * args.bytes_per_token_mib))
    manager = PagedKVBlockManager(args.block_size, total_blocks)
    waiting = list(requests)
    active: list[Request] = []
    now = 0.0
    completed = 0

    while completed < len(requests):
        if not active and waiting and waiting[0].arrival_ms > now:
            now = waiting[0].arrival_ms

        admitted: list[Request] = []
        while waiting and waiting[0].arrival_ms <= now and len(active) + len(admitted) < args.max_active:
            req = waiting[0]
            reserve_tokens = req.profile.prompt_tokens if paged else req.profile.prompt_tokens + req.decode_tokens
            reserve_blocks = math.ceil(reserve_tokens / args.block_size)
            if guarded and paged:
                # Keep one future decode block available per active/admitted request.
                # This avoids admitting too many visual prefixes and then stalling at decode append.
                decode_watermark_blocks = len(active) + len(admitted) + 1
                if len(manager.free_blocks) < reserve_blocks + decode_watermark_blocks:
                    break
            if not manager.allocate_blocks(req, reserve_tokens):
                break
            waiting.pop(0)
            req.start_ms = now
            admitted.append(req)

        if admitted:
            # Prefill same-step arrivals as one coarse batch. This is a simulator, so use the max prefill.
            now += max(req.profile.prefill_ms for req in admitted)
            active.extend(admitted)
            manager.sample()
            continue

        if not active:
            continue

        step_ms = max(req.profile.tpot_ms for req in active) * (1.0 + 0.035 * max(0, len(active) - 1))
        now += step_ms
        still_active: list[Request] = []
        for req in active:
            if paged and not manager.append_token(req):
                still_active.append(req)
                continue
            req.remaining_decode -= 1
            if req.remaining_decode <= 0:
                req.finish_ms = now
                manager.free(req)
                completed += 1
            else:
                still_active.append(req)
        active = still_active
        manager.sample()

    if paged and guarded:
        name = "continuous_paged_kv_guarded"
    else:
        name = "continuous_paged_kv" if paged else "continuous_static_kv"
    return summarize(name, requests, now, manager, args.bytes_per_token_mib, args.block_size)


def main() -> None:
    args = parse_args()
    profiles = load_profiles(ROOT / args.visual_csv)
    decode_options = [int(item.strip()) for item in args.decode_options.split(",") if item.strip()]
    workload = make_workload(profiles, args.request_count, args.mean_arrival_ms, decode_options, args.seed)

    rows = [
        run_serial(clone_workload(workload), args),
        run_continuous(clone_workload(workload), args, paged=False),
        run_continuous(clone_workload(workload), args, paged=True),
        run_continuous(clone_workload(workload), args, paged=True, guarded=True),
    ]
    baseline = float(rows[0]["throughput_req_s"])
    for row in rows:
        row["speedup_vs_serial"] = float(row["throughput_req_s"]) / baseline
        row["block_size"] = args.block_size
        row["kv_budget_mib"] = args.kv_budget_mib
        row["max_active"] = args.max_active
        print(
            f"{row['scenario']}: throughput={row['throughput_req_s']:.2f} req/s "
            f"mean={row['mean_latency_ms']:.1f}ms p95={row['p95_latency_ms']:.1f}ms "
            f"peak_kv={row['peak_kv_mib']:.1f}MiB util={row['avg_block_utilization']:.3f} "
            f"speedup={row['speedup_vs_serial']:.2f}x",
            flush=True,
        )

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario",
        "request_count",
        "makespan_ms",
        "throughput_req_s",
        "speedup_vs_serial",
        "mean_latency_ms",
        "p95_latency_ms",
        "peak_blocks",
        "avg_allocated_blocks",
        "avg_block_utilization",
        "peak_kv_mib",
        "block_size",
        "kv_budget_mib",
        "max_active",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
