"""Analyze the paired LIBERO delay sweep for the d<=4 Pi0.5 policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable


Row = dict[str, str]
Metric = Callable[[Row], float]


def read_rows(paths: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row: Row, key: str) -> float:
    return float(row[key])


def success(row: Row) -> float:
    return float(row["success"].lower() == "true")


def episode_key(row: Row) -> tuple[str, str, str]:
    return row["task_id"], row["episode_index"], row["seed"]


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(
    candidate: list[Row],
    reference: list[Row],
    metric: Metric,
    samples: int,
    seed: int,
) -> dict:
    candidate_by_episode = {episode_key(row): row for row in candidate}
    reference_by_episode = {episode_key(row): row for row in reference}
    common = sorted(candidate_by_episode.keys() & reference_by_episode.keys())
    differences = [
        metric(candidate_by_episode[key]) - metric(reference_by_episode[key]) for key in common
    ]
    if not differences:
        raise ValueError("Paired comparison has no common episodes")
    rng = random.Random(seed)
    bootstrap = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    ]
    return {
        "pairs": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
    }


def summarize(rows: list[Row]) -> dict:
    return {
        "episodes": len(rows),
        "success_rate": finite_mean([success(row) for row in rows]),
        "mean_steps": finite_mean([number(row, "steps") for row in rows]),
        "mean_inference_ms": finite_mean([number(row, "mean_inference_ms") for row in rows]),
        "p95_inference_ms": finite_mean([number(row, "p95_inference_ms") for row in rows]),
        "mean_state_prediction_mse": finite_mean(
            [number(row, "mean_state_prediction_mse") for row in rows]
        ),
        "mean_handoff_action_l2": finite_mean(
            [number(row, "mean_handoff_action_l2") for row in rows]
        ),
        "queue_underflows": int(sum(number(row, "queue_underflows") for row in rows)),
    }


def compare(
    candidate: list[Row], reference: list[Row], samples: int, seed: int
) -> dict:
    return {
        "success_rate": paired_bootstrap(candidate, reference, success, samples, seed),
        "steps": paired_bootstrap(
            candidate, reference, lambda row: number(row, "steps"), samples, seed + 1
        ),
        "handoff_action_l2": paired_bootstrap(
            candidate,
            reference,
            lambda row: number(row, "mean_handoff_action_l2"),
            samples,
            seed + 2,
        ),
    }


def build_groups(rows: list[Row]) -> dict[str, list[Row]]:
    source: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        if row["task_id"] == "3":
            source[(row["condition"], int(row["delay_ticks"]))].append(row)

    groups: dict[str, list[Row]] = {}
    for delay in range(5):
        groups[f"stale_policy|d{delay}"] = source[("stale", delay)]
        # At d=0 no future state is needed, so the synchronous learned-policy
        # anchor is shared by both learned-policy series.
        learned_condition = "learned_stale" if delay == 0 else "learned"
        groups[f"learned_future|d{delay}"] = source[(learned_condition, delay)]
        groups[f"learned_stale|d{delay}"] = source[("learned_stale", delay)]

    for name, values in groups.items():
        if len(values) != 10:
            raise ValueError(f"Expected 10 paired episodes for {name}, found {len(values)}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    groups = build_groups(read_rows(args.inputs))
    aggregate = {name: summarize(values) for name, values in sorted(groups.items())}
    paired: dict[str, dict] = {}

    for delay in range(1, 5):
        paired[f"learned_future_vs_same_weight_stale|d{delay}"] = compare(
            groups[f"learned_future|d{delay}"],
            groups[f"learned_stale|d{delay}"],
            args.bootstrap_samples,
            20260715 + delay * 10,
        )

    for delay in range(5):
        paired[f"learned_future_vs_stale_policy|d{delay}"] = compare(
            groups[f"learned_future|d{delay}"],
            groups[f"stale_policy|d{delay}"],
            args.bootstrap_samples,
            20260815 + delay * 10,
        )

    for series in ("stale_policy", "learned_future", "learned_stale"):
        reference = groups[f"{series}|d0"]
        for delay in range(1, 5):
            paired[f"{series}|d{delay}_vs_d0"] = compare(
                groups[f"{series}|d{delay}"],
                reference,
                args.bootstrap_samples,
                20260915 + delay * 10,
            )

    report = {
        "experiment": {
            "suite": "libero_spatial",
            "task_id": 3,
            "task": "pick up the black bowl on the cookie box and place it on the plate",
            "episodes_per_condition": 10,
            "paired_seeds": list(range(1000, 1010)),
            "control_frequency_hz": 10,
            "replan_interval_ticks": 10,
            "delay_ticks": list(range(5)),
            "delay_ms": [0, 100, 200, 300, 400],
            "bootstrap_samples": args.bootstrap_samples,
            "training_delay_range": "d=0..4",
        },
        "aggregate": aggregate,
        "paired": paired,
        "interpretation_guardrails": {
            "same_weight_causal_comparison": "learned_future vs learned_stale",
            "cross_weight_reference_only": "learned_future vs stale_policy",
            "d0_alias": "learned_future and learned_stale share the same synchronous learned-policy rows",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "d4_delay_analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
