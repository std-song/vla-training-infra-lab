"""Paired comparisons for endpoint and learned future-state proxy experiments."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def load_rows(path: str, mode: str) -> dict[tuple[int, int], dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["dataset_index"]), int(row["delay"])): row
        for row in rows
        if row["mode"] == mode
    }


def compare(
    reference: dict[tuple[int, int], dict[str, float]],
    candidate: dict[tuple[int, int], dict[str, float]],
    label: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    output = []
    for delay in sorted({key[1] for key in reference} & {key[1] for key in candidate}):
        keys = sorted(key for key in reference if key[1] == delay and key in candidate)
        for metric in ("first_action_mse", "first4_mse", "chunk_mse"):
            ref = np.asarray([float(reference[key][metric]) for key in keys])
            cand = np.asarray([float(candidate[key][metric]) for key in keys])
            finite = np.isfinite(ref) & np.isfinite(cand)
            ref, cand = ref[finite], cand[finite]
            differences = ref - cand
            bootstrap_indices = rng.integers(0, len(differences), size=(10000, len(differences)))
            bootstrap = differences[bootstrap_indices].mean(axis=1)
            output.append(
                {
                    "comparison": label,
                    "delay": delay,
                    "metric": metric,
                    "pairs": len(differences),
                    "reference_mean": float(ref.mean()),
                    "candidate_mean": float(cand.mean()),
                    "candidate_reduction_percent": float(100 * (ref.mean() - cand.mean()) / ref.mean()),
                    "mean_paired_difference": float(differences.mean()),
                    "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    original_endpoint = load_rows(args.original, "endpoint_proxy")
    learned_endpoint = load_rows(args.learned, "endpoint_proxy")
    learned_proxy = load_rows(args.learned, "learned_proxy")
    oracle = load_rows(args.learned, "oracle_state")
    rows = []
    rows.extend(compare(learned_endpoint, learned_proxy, "new_policy_endpoint_vs_learned", args.seed))
    rows.extend(compare(original_endpoint, learned_proxy, "original_vlash_vs_new_learned", args.seed + 1))
    rows.extend(compare(learned_proxy, oracle, "new_learned_vs_oracle", args.seed + 2))
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
