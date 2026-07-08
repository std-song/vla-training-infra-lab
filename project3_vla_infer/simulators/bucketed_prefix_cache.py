from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    prefix_id: int
    prefix_hit: bool = False
    start_ms: float | None = None
    finish_ms: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucketed continuous batching and prefix-cache simulator for Qwen2.5-VL VLA workloads.")
    parser.add_argument("--visual-csv", default="project3_vla_infer/results/qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv")
    parser.add_argument("--out", default="project3_vla_infer/results/qwen25vl_bucketed_prefix_cache_sim.csv")
    parser.add_argument("--request-count", type=int, default=256)
    parser.add_argument("--mean-arrival-ms", type=float, default=70.0)
    parser.add_argument("--decode-options", default="16,32,64")
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-requests", type=int, default=16)
    parser.add_argument("--prefix-pool", type=int, default=32)
    parser.add_argument("--prefix-hit-prefill-ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=11)
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
    return profiles


def make_workload(args: argparse.Namespace, profiles: list[Profile]) -> list[Request]:
    rng = random.Random(args.seed)
    weights = [1.0 if p.image_count == 1 else 2.4 for p in profiles]
    decode_options = [int(item.strip()) for item in args.decode_options.split(",") if item.strip()]
    now = 0.0
    requests: list[Request] = []
    for request_id in range(args.request_count):
        now += rng.expovariate(1.0 / args.mean_arrival_ms)
        profile = rng.choices(profiles, weights=weights, k=1)[0]
        decode_tokens = rng.choice(decode_options)
        # Repeated prefix ids model repeated task/image prefixes in robot control rollouts.
        prefix_id = rng.randrange(args.prefix_pool)
        requests.append(Request(request_id, profile, now, decode_tokens, prefix_id))
    return requests


def clone_workload(requests: list[Request]) -> list[Request]:
    return [Request(r.request_id, r.profile, r.arrival_ms, r.decode_tokens, r.prefix_id) for r in requests]


def select_batch(waiting: list[Request], policy: str, max_tokens: int, max_requests: int) -> list[Request]:
    if not waiting:
        return []
    candidates = list(waiting)
    if policy == "fcfs":
        ordered = candidates
    elif policy == "shape_bucket":
        first = candidates[0].profile.name
        ordered = [r for r in candidates if r.profile.name == first] + [r for r in candidates if r.profile.name != first]
    elif policy == "token_budget_bucket":
        # Prefer shorter visual prefixes to reduce padding waste while respecting arrival eligibility.
        ordered = sorted(candidates, key=lambda r: (r.profile.prompt_tokens, r.arrival_ms, r.request_id))
    else:
        raise ValueError(policy)

    batch: list[Request] = []
    max_prompt = 0
    for req in ordered:
        maybe_max = max(max_prompt, req.profile.prompt_tokens)
        maybe_tokens = maybe_max * (len(batch) + 1)
        if batch and (len(batch) >= max_requests or maybe_tokens > max_tokens):
            continue
        if not batch and maybe_tokens > max_tokens:
            batch.append(req)
            break
        batch.append(req)
        max_prompt = maybe_max
        if len(batch) >= max_requests:
            break
    return batch


def run_policy(requests: list[Request], args: argparse.Namespace, policy: str, prefix_cache: bool) -> dict[str, float | str | int]:
    waiting = list(requests)
    active_prefixes: set[int] = set()
    now = 0.0
    completed: list[Request] = []
    padding_waste_tokens = 0
    useful_prompt_tokens = 0
    prefix_hits = 0
    batch_sizes: list[int] = []

    while waiting:
        if waiting[0].arrival_ms > now:
            now = waiting[0].arrival_ms
        eligible = [r for r in waiting if r.arrival_ms <= now]
        batch = select_batch(eligible, policy, args.max_batch_tokens, args.max_batch_requests)
        if not batch:
            now = waiting[0].arrival_ms
            continue

        for req in batch:
            waiting.remove(req)
            req.start_ms = now
            if prefix_cache and req.prefix_id in active_prefixes:
                req.prefix_hit = True
                prefix_hits += 1
            active_prefixes.add(req.prefix_id)

        max_prompt = max(req.profile.prompt_tokens for req in batch)
        useful_prompt_tokens += sum(req.profile.prompt_tokens for req in batch)
        padding_waste_tokens += max_prompt * len(batch) - sum(req.profile.prompt_tokens for req in batch)
        batch_sizes.append(len(batch))

        prefill_ms = max(
            req.profile.prefill_ms * (args.prefix_hit_prefill_ratio if req.prefix_hit else 1.0)
            for req in batch
        )
        # Decode is modeled as a batch step loop. Longer batch and mixed TPOT add overhead.
        max_decode = max(req.decode_tokens for req in batch)
        decode_ms = 0.0
        for step in range(max_decode):
            alive = [req for req in batch if req.decode_tokens > step]
            max_tpot = max(req.profile.tpot_ms for req in alive)
            decode_ms += max_tpot * (1.0 + 0.035 * max(0, len(alive) - 1))
        now += prefill_ms + decode_ms

        for req in batch:
            req.finish_ms = now
            completed.append(req)

    latencies = [req.finish_ms - req.arrival_ms for req in completed if req.finish_ms is not None]
    p95 = sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
    return {
        "policy": policy,
        "prefix_cache": int(prefix_cache),
        "request_count": len(completed),
        "makespan_ms": now,
        "throughput_req_s": len(completed) / (now / 1000.0),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": p95,
        "avg_batch_size": statistics.mean(batch_sizes),
        "max_batch_size": max(batch_sizes),
        "padding_waste_tokens": padding_waste_tokens,
        "padding_waste_ratio": padding_waste_tokens / max(useful_prompt_tokens + padding_waste_tokens, 1),
        "prefix_hit_rate": prefix_hits / len(completed),
    }


def main() -> None:
    args = parse_args()
    profiles = load_profiles(ROOT / args.visual_csv)
    workload = make_workload(args, profiles)
    rows = []
    for policy in ["fcfs", "shape_bucket", "token_budget_bucket"]:
        for prefix_cache in [False, True]:
            row = run_policy(clone_workload(workload), args, policy, prefix_cache)
            rows.append(row)
            print(
                f"{policy} cache={prefix_cache}: throughput={row['throughput_req_s']:.2f} req/s "
                f"p95={row['p95_latency_ms']:.1f}ms waste={row['padding_waste_ratio']:.3f} "
                f"hit={row['prefix_hit_rate']:.2f}",
                flush=True,
            )

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "policy",
        "prefix_cache",
        "request_count",
        "makespan_ms",
        "throughput_req_s",
        "mean_latency_ms",
        "p95_latency_ms",
        "avg_batch_size",
        "max_batch_size",
        "padding_waste_tokens",
        "padding_waste_ratio",
        "prefix_hit_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
