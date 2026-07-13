"""Aggregate and bootstrap paired LIBERO closed-loop evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    values_a: list[float], values_b: list[float], samples: int, seed: int
) -> dict[str, float]:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Paired bootstrap requires equally sized non-empty inputs")
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        indices = [rng.randrange(len(values_a)) for _ in values_a]
        differences.append(mean([values_a[i] - values_b[i] for i in indices]))
    differences.sort()
    return {
        "mean_difference": mean([a - b for a, b in zip(values_a, values_b)]),
        "bootstrap_ci95_low": percentile(differences, 0.025),
        "bootstrap_ci95_high": percentile(differences, 0.975),
    }


def parse_float(value: str) -> float:
    return float(value) if value.lower() != "nan" else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--pair-a", default="learned")
    parser.add_argument("--pair-b", default="learned_stale")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    with args.episodes_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["suite"], int(row["task_id"]), int(row["delay_ticks"]))].append(row)

    groups = {}
    for key, values in sorted(grouped.items()):
        condition, suite, task_id, delay = key
        state_errors = [
            parse_float(value["mean_state_prediction_mse"])
            for value in values
            if not math.isnan(parse_float(value["mean_state_prediction_mse"]))
        ]
        groups[f"{condition}|{suite}|task{task_id}|delay{delay}"] = {
            "episodes": len(values),
            "successes": sum(value["success"] == "True" for value in values),
            "success_rate": mean([float(value["success"] == "True") for value in values]),
            "mean_steps": mean([float(value["steps"]) for value in values]),
            "mean_inference_ms": mean([float(value["mean_inference_ms"]) for value in values]),
            "mean_handoff_action_l2": mean([float(value["mean_handoff_action_l2"]) for value in values]),
            "mean_state_prediction_mse": mean(state_errors),
            "queue_underflows": sum(int(value["queue_underflows"]) for value in values),
        }

    rows_a = {
        (row["suite"], int(row["task_id"]), int(row["episode_index"]), int(row["delay_ticks"])): row
        for row in rows if row["condition"] == args.pair_a
    }
    rows_b = {
        (row["suite"], int(row["task_id"]), int(row["episode_index"]), int(row["delay_ticks"])): row
        for row in rows if row["condition"] == args.pair_b
    }
    common = sorted(rows_a.keys() & rows_b.keys())
    paired = {"pair_a": args.pair_a, "pair_b": args.pair_b, "episodes": len(common)}
    for metric in ("success", "steps", "mean_handoff_action_l2"):
        if metric == "success":
            values_a = [float(rows_a[key][metric] == "True") for key in common]
            values_b = [float(rows_b[key][metric] == "True") for key in common]
        else:
            values_a = [float(rows_a[key][metric]) for key in common]
            values_b = [float(rows_b[key][metric]) for key in common]
        paired[metric] = paired_bootstrap(
            values_a, values_b, args.bootstrap_samples, args.seed
        )

    result = {"groups": groups, "paired_comparison": paired}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
