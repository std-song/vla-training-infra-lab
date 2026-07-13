"""Inspect LIBERO dataset and Pi0.5 checkpoint interfaces before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--policy-path", required=True, type=Path)
    args = parser.parse_args()

    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    policy = json.loads((args.policy_path / "config.json").read_text(encoding="utf-8"))
    episode_table = pq.read_table(args.dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    episode_rows = episode_table.to_pylist()
    task_counts = Counter()
    for row in episode_rows:
        tasks = row.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = [tasks]
        task_counts.update(tasks)

    selected_features = {
        key: value
        for key, value in info["features"].items()
        if key in {"observation.state", "action"} or key.startswith("observation.images.")
    }
    selected_policy = {
        key: policy.get(key)
        for key in (
            "type", "chunk_size", "n_action_steps", "max_state_dim", "max_action_dim",
            "input_features", "output_features",
        )
    }
    print(json.dumps({
        "dataset": {key: info.get(key) for key in ("total_episodes", "total_frames", "total_tasks", "fps")},
        "features": selected_features,
        "episode_schema": str(episode_table.schema),
        "episode_examples": episode_rows[:3],
        "task_episode_counts": dict(sorted(task_counts.items())),
        "policy": selected_policy,
    }, indent=2))


if __name__ == "__main__":
    main()
