"""Unify the paired task-3, d=2 Pi0.5/VLASH closed-loop matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from analyze_libero_standard_delay import paired_bootstrap


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict], condition: str, delay: int) -> list[dict]:
    return [
        row
        for row in rows
        if row["condition"] == condition
        and int(row["delay_ticks"]) == delay
        and int(row["task_id"]) == 3
    ]


def success(row: dict) -> float:
    return float(row["success"].lower() == "true")


def value(key: str):
    return lambda row: float(row[key])


def finite_mean(rows: list[dict], metric) -> float:
    values = [metric(row) for row in rows]
    values = [item for item in values if math.isfinite(item)]
    return sum(values) / len(values)


def aggregate(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "success_rate": finite_mean(rows, success),
        "mean_steps": finite_mean(rows, value("steps")),
        "mean_inference_ms": finite_mean(rows, value("mean_inference_ms")),
        "mean_handoff_action_l2": finite_mean(rows, value("mean_handoff_action_l2")),
        "queue_underflows": int(sum(float(row["queue_underflows"]) for row in rows)),
    }


def compare(candidate: list[dict], reference: list[dict], samples: int) -> dict:
    return {
        "success_rate": paired_bootstrap(candidate, reference, success, samples, 20260713),
        "steps": paired_bootstrap(candidate, reference, value("steps"), samples, 20260714),
        "handoff_action_l2": paired_bootstrap(
            candidate, reference, value("mean_handoff_action_l2"), samples, 20260715
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-sync", required=True, type=Path)
    parser.add_argument("--standard-delay", required=True, type=Path)
    parser.add_argument("--stale", required=True, type=Path)
    parser.add_argument("--learned-ablation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    standard_sync = read(args.standard_sync)
    standard_delay = read(args.standard_delay)
    stale = read(args.stale)
    learned = read(args.learned_ablation)
    groups = {
        "standard_sync": select(standard_sync, "standard_sync", 0),
        "standard_naive_d2": select(standard_delay, "standard_naive", 2),
        "standard_skip_d2": select(standard_delay, "standard_skip", 2),
        "stale_augmented_d2": select(stale, "stale", 2),
        "learned_future_d2": select(learned, "learned", 2),
        "learned_stale_input_d2": select(learned, "learned_stale", 2),
    }
    for name, rows in groups.items():
        if len(rows) != 10:
            raise ValueError(f"Expected 10 paired episodes for {name}, found {len(rows)}")

    report = {
        "experiment": {
            "suite": "libero_spatial",
            "task_id": 3,
            "delay_ticks": 2,
            "delay_ms": 200,
            "episodes_per_condition": 10,
            "paired_episode_indices": list(range(10)),
            "bootstrap_samples": args.bootstrap_samples,
        },
        "aggregate": {name: aggregate(rows) for name, rows in groups.items()},
        "same_weight_comparisons": {
            "standard_naive_vs_standard_sync": compare(
                groups["standard_naive_d2"], groups["standard_sync"], args.bootstrap_samples
            ),
            "standard_skip_vs_standard_naive": compare(
                groups["standard_skip_d2"], groups["standard_naive_d2"], args.bootstrap_samples
            ),
            "learned_future_vs_learned_stale_input": compare(
                groups["learned_future_d2"],
                groups["learned_stale_input_d2"],
                args.bootstrap_samples,
            ),
        },
        "cross_weight_comparisons": {
            "stale_augmented_vs_standard_naive": compare(
                groups["stale_augmented_d2"], groups["standard_naive_d2"], args.bootstrap_samples
            ),
            "learned_future_vs_stale_augmented": compare(
                groups["learned_future_d2"], groups["stale_augmented_d2"], args.bootstrap_samples
            ),
        },
        "interpretation_guardrail": (
            "Same-weight comparisons isolate deployment/input handling. Cross-weight comparisons "
            "also include independent LoRA optimization and different offset supervision counts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
