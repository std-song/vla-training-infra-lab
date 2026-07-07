from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LeRobotSchema:
    repo_id: str
    fps: int
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    features: dict[str, Any]
    task_by_index: dict[int, str]


@dataclass(frozen=True)
class LeRobotCachePaths:
    snapshot_dir: Path
    info_path: Path
    data_path: Path
    tasks_path: Path | None
    episodes_path: Path | None


def _is_repo_snapshot(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


def find_lerobot_snapshot(cache_root: str | Path, repo_id: str) -> LeRobotCachePaths:
    """Find a downloaded LeRobot dataset snapshot without contacting the Hub."""
    root = Path(cache_root).expanduser().resolve()
    repo_marker = "datasets--" + repo_id.replace("/", "--")

    candidates: list[Path] = []
    for marker_dir in root.rglob(repo_marker):
        snapshots = marker_dir / "snapshots"
        if snapshots.exists():
            candidates.extend([p for p in snapshots.iterdir() if p.is_dir() and _is_repo_snapshot(p)])
    if _is_repo_snapshot(root):
        candidates.append(root)

    if not candidates:
        raise FileNotFoundError(f"No local LeRobot snapshot for {repo_id!r} under {root}")

    snapshot = sorted(candidates, key=lambda p: len(str(p)))[0]
    info_path = snapshot / "meta" / "info.json"
    data_files = sorted((snapshot / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet found under {snapshot / 'data'}")

    task_path = snapshot / "meta" / "tasks.parquet"
    if not task_path.exists():
        task_path = None

    episode_files = sorted((snapshot / "meta" / "episodes").rglob("*.parquet"))
    episodes_path = episode_files[0] if episode_files else None

    return LeRobotCachePaths(
        snapshot_dir=snapshot,
        info_path=info_path,
        data_path=data_files[0],
        tasks_path=task_path,
        episodes_path=episodes_path,
    )


def _load_tasks(tasks_path: Path | None) -> dict[int, str]:
    if tasks_path is None:
        return {}
    df = pd.read_parquet(tasks_path)
    if "task" in df.columns and "task_index" in df.columns:
        return {int(row.task_index): str(row.task) for row in df.itertuples(index=False)}

    # LeRobot v3 stores task text as the index and task_index as the only column.
    if "task_index" in df.columns:
        return {int(row.task_index): str(idx) for idx, row in df.iterrows()}
    return {}


def load_schema(paths: LeRobotCachePaths, repo_id: str) -> LeRobotSchema:
    info = json.loads(paths.info_path.read_text())
    return LeRobotSchema(
        repo_id=repo_id,
        fps=int(info.get("fps", 0)),
        robot_type=str(info.get("robot_type", "unknown")),
        total_episodes=int(info.get("total_episodes", 0)),
        total_frames=int(info.get("total_frames", 0)),
        total_tasks=int(info.get("total_tasks", 0)),
        features=dict(info.get("features", {})),
        task_by_index=_load_tasks(paths.tasks_path),
    )


class LeRobotParquetDataset(Dataset):
    """Low-dimensional LeRobot dataset view for adapter smoke tests.

    This intentionally reads only parquet columns such as state/action/task metadata.
    Video decoding is left for a later image pipeline stage.
    """

    def __init__(self, cache_root: str | Path, repo_id: str = "lerobot/aloha_mobile_cabinet", limit: int | None = None):
        self.paths = find_lerobot_snapshot(cache_root, repo_id)
        self.schema = load_schema(self.paths, repo_id)
        self.frame_df = pd.read_parquet(self.paths.data_path)
        if limit is not None:
            self.frame_df = self.frame_df.iloc[:limit].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame_df.iloc[index]
        task_index = int(row["task_index"])
        return {
            "state": torch.as_tensor(np.asarray(row["observation.state"]).copy(), dtype=torch.float32),
            "effort": torch.as_tensor(np.asarray(row["observation.effort"]).copy(), dtype=torch.float32),
            "action": torch.as_tensor(np.asarray(row["action"]).copy(), dtype=torch.float32),
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "done": bool(row["next.done"]),
            "index": int(row["index"]),
            "task_index": task_index,
            "task_text": self.schema.task_by_index.get(task_index, ""),
        }

