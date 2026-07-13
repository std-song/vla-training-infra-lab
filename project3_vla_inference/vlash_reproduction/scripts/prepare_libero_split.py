"""Create a task-stratified episode split for lerobot/libero."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_columns(data_root: Path, columns: list[str]) -> pa.Table:
    files = sorted((data_root / "data").glob("chunk-*/*.parquet"))
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in files])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    table = read_columns(args.dataset_root, ["episode_index", "task_index"])
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    episode_to_task = {}
    for episode, task in zip(episode_indices, task_indices, strict=True):
        previous = episode_to_task.setdefault(int(episode), int(task))
        if previous != int(task):
            raise RuntimeError(f"Episode {episode} contains multiple task indices")

    task_to_episodes: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        task_to_episodes[task].append(episode)
    rng = np.random.default_rng(args.seed)
    train_ids, validation_ids = [], []
    per_task = {}
    for task, episodes in sorted(task_to_episodes.items()):
        episodes = np.asarray(sorted(episodes), dtype=np.int64)
        rng.shuffle(episodes)
        validation_count = max(1, int(round(len(episodes) * args.validation_fraction)))
        validation = sorted(int(value) for value in episodes[:validation_count])
        train = sorted(int(value) for value in episodes[validation_count:])
        train_ids.extend(train)
        validation_ids.extend(validation)
        per_task[str(task)] = {"train": len(train), "validation": len(validation)}

    result = {
        "dataset": "lerobot/libero",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "per_task_counts": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "train_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
        "tasks": len(per_task),
        "per_task_counts": per_task,
    }, indent=2))


if __name__ == "__main__":
    main()
