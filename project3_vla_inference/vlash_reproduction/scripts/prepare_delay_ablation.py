"""Create a fixed episode split and render matched upstream-style configs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def episode_ids(metadata: LeRobotDatasetMetadata) -> list[int]:
    episodes = metadata.episodes
    if "episode_index" in episodes.column_names:
        return sorted(int(value) for value in episodes["episode_index"])
    return list(range(len(episodes)))


def render(template: Path, destination: Path, project_root: str, train_ids: list[int]) -> None:
    content = template.read_text(encoding="utf-8")
    content = content.replace("__PROJECT_ROOT__", project_root)
    content = content.replace("__TRAIN_EPISODES__", json.dumps(train_ids))
    destination.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--template-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet", root=args.dataset_root)
    ids = episode_ids(metadata)
    if len(ids) < 2:
        raise RuntimeError("Need at least two episodes for an episode-level split.")

    shuffled = ids.copy()
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * args.validation_fraction))
    validation_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "dataset": "lerobot/aloha_mobile_cabinet",
        "seed": args.seed,
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
    }
    (out_dir / "episode_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    template_dir = Path(args.template_dir)
    render(
        template_dir / "pi05_delay_ablation_sync.template.yaml",
        out_dir / "pi05_delay_ablation_sync.yaml",
        args.project_root,
        train_ids,
    )
    render(
        template_dir / "pi05_delay_ablation_vlash.template.yaml",
        out_dir / "pi05_delay_ablation_vlash.yaml",
        args.project_root,
        train_ids,
    )
    print(json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
