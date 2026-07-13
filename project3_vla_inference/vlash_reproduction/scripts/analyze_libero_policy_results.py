"""Paired bootstrap analysis for LIBERO future-state policy evaluation."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def load(path: str) -> dict[str, dict[tuple[int, int], dict[str, str]]]:
    grouped = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["result"], {})[(int(row["dataset_index"]), int(row["delay"]))] = row
    return grouped


def compare(reference: dict, candidate: dict, label: str, rng: np.random.Generator) -> list[dict]:
    rows = []
    delays = sorted({key[1] for key in reference} & {key[1] for key in candidate})
    for delay in delays:
        keys = sorted(key for key in reference if key[1] == delay and key in candidate)
        for metric in ("first_action_mse", "first4_mse", "chunk_mse"):
            ref = np.asarray([float(reference[key][metric]) for key in keys])
            cand = np.asarray([float(candidate[key][metric]) for key in keys])
            finite = np.isfinite(ref) & np.isfinite(cand)
            ref, cand = ref[finite], cand[finite]
            difference = ref - cand
            indices = rng.integers(0, len(difference), size=(10000, len(difference)))
            bootstrap = difference[indices].mean(axis=1)
            rows.append({
                "comparison": label,
                "delay": delay,
                "delay_ms": delay * 100,
                "metric": metric,
                "pairs": len(difference),
                "reference_mean": float(ref.mean()),
                "candidate_mean": float(cand.mean()),
                "candidate_reduction_percent": float(100 * (ref.mean() - cand.mean()) / ref.mean()),
                "mean_paired_difference": float(difference.mean()),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    grouped = load(args.per_sample)
    rng = np.random.default_rng(args.seed)
    rows = []
    rows.extend(compare(grouped["stale_baseline"], grouped["learned_dynamics"], "stale_vs_learned", rng))
    rows.extend(compare(grouped["learned_policy_stale_input"], grouped["learned_dynamics"], "learned_policy_stale_vs_learned_input", rng))
    rows.extend(compare(grouped["learned_dynamics"], grouped["oracle_future_state"], "learned_vs_oracle", rng))
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
