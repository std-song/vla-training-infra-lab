"""Aggregate paired standard-policy LIBERO delay ablations without pandas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row: dict, key: str) -> float:
    return float(row[key])


def truth(row: dict, key: str = "success") -> float:
    return float(row[key].lower() == "true")


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def condition_key(row: dict) -> str:
    return f"{row['condition']}|d{row['delay_ticks']}"


def episode_key(row: dict) -> tuple[str, str, str]:
    return row["task_id"], row["episode_index"], row["seed"]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(
    candidate: list[dict], reference: list[dict], metric, samples: int, seed: int
) -> dict:
    ref_by_episode = {episode_key(row): row for row in reference}
    pairs = [(row, ref_by_episode[episode_key(row)]) for row in candidate if episode_key(row) in ref_by_episode]
    differences = [metric(left) - metric(right) for left, right in pairs]
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        boot.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    return {
        "pairs": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.inputs) if row["task_id"] == "3"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)

    aggregate = {}
    for key, values in sorted(grouped.items()):
        aggregate[key] = {
            "episodes": len(values),
            "success_rate": mean_finite([truth(row) for row in values]),
            "mean_steps": mean_finite([number(row, "steps") for row in values]),
            "mean_inference_ms": mean_finite([number(row, "mean_inference_ms") for row in values]),
            "mean_handoff_action_l2": mean_finite(
                [number(row, "mean_handoff_action_l2") for row in values]
            ),
            "queue_underflows": int(sum(number(row, "queue_underflows") for row in values)),
        }

    reference = grouped["standard_sync|d0"]
    paired = {}
    for key, values in sorted(grouped.items()):
        if key == "standard_sync|d0":
            continue
        sync_by_episode = {episode_key(row): row for row in reference}
        transitions = defaultdict(int)
        for row in values:
            ref = sync_by_episode[episode_key(row)]
            transitions[f"sync_{int(truth(ref))}_candidate_{int(truth(row))}"] += 1
        paired[f"{key}_vs_sync"] = {
            "success_rate": paired_bootstrap(
                values, reference, truth, args.bootstrap_samples, seed=20260713
            ),
            "steps": paired_bootstrap(
                values,
                reference,
                lambda row: number(row, "steps"),
                args.bootstrap_samples,
                seed=20260714,
            ),
            "handoff_action_l2": paired_bootstrap(
                values,
                reference,
                lambda row: number(row, "mean_handoff_action_l2"),
                args.bootstrap_samples,
                seed=20260715,
            ),
            "success_transitions": dict(sorted(transitions.items())),
        }

    for delay in (1, 2, 4):
        candidate = grouped[f"standard_skip|d{delay}"]
        naive = grouped[f"standard_naive|d{delay}"]
        paired[f"standard_skip|d{delay}_vs_standard_naive|d{delay}"] = {
            "success_rate": paired_bootstrap(
                candidate, naive, truth, args.bootstrap_samples, seed=20260713 + delay
            ),
            "steps": paired_bootstrap(
                candidate,
                naive,
                lambda row: number(row, "steps"),
                args.bootstrap_samples,
                seed=20260723 + delay,
            ),
            "handoff_action_l2": paired_bootstrap(
                candidate,
                naive,
                lambda row: number(row, "mean_handoff_action_l2"),
                args.bootstrap_samples,
                seed=20260733 + delay,
            ),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": {
            "suite": "libero_spatial",
            "task_id": 3,
            "episodes_per_condition": 10,
            "paired_episode_indices": list(range(10)),
            "control_frequency_hz": 10,
            "replan_interval_ticks": 10,
            "delays_ticks": [1, 2, 4],
            "delays_ms": [100, 200, 400],
            "bootstrap_samples": args.bootstrap_samples,
        },
        "aggregate": aggregate,
        "paired": paired,
    }
    output = args.out_dir / "standard_delay_analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
