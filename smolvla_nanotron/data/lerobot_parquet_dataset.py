from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def camera_features(schema: LeRobotSchema) -> list[str]:
    return [name for name, feature in schema.features.items() if feature.get("dtype") == "video"]


class VideoFrameDecoder:
    """Lazy TorchCodec decoder wrapper for sampled LeRobot video frames."""

    def __init__(self, snapshot_dir: Path, cameras: list[str], image_size: int | None = None):
        self.snapshot_dir = snapshot_dir
        self.cameras = cameras
        self.image_size = image_size
        self._decoders: dict[str, Any] = {}

    def _video_path(self, camera: str) -> Path:
        # This sampled adapter intentionally targets file-000.mp4. The metadata
        # resolver for arbitrary episode/file shards is a later pipeline step.
        path = self.snapshot_dir / "videos" / camera / "chunk-000" / "file-000.mp4"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing sampled video shard for {camera}: {path}. "
                "Download videos/*/chunk-000/file-000.mp4 first."
            )
        return path

    def _decoder(self, camera: str) -> Any:
        if camera not in self._decoders:
            from torchcodec.decoders import VideoDecoder

            self._decoders[camera] = VideoDecoder(str(self._video_path(camera)))
        return self._decoders[camera]

    def decode(self, frame_index: int) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        for camera in self.cameras:
            frame = self._decoder(camera)[frame_index]
            if self.image_size is not None:
                frame = F.interpolate(
                    frame.unsqueeze(0).float(),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).to(torch.uint8)
            images[camera] = frame.contiguous()
        return images


class LeRobotParquetDataset(Dataset):
    """LeRobot parquet dataset view for adapter smoke tests.

    By default this reads only low-dimensional state/action/task columns.
    Set `include_images=True` to decode sampled RGB frames from local video shards.
    """

    def __init__(
        self,
        cache_root: str | Path,
        repo_id: str = "lerobot/aloha_mobile_cabinet",
        limit: int | None = None,
        include_images: bool = False,
        cameras: list[str] | None = None,
        image_size: int | None = None,
    ):
        self.paths = find_lerobot_snapshot(cache_root, repo_id)
        self.schema = load_schema(self.paths, repo_id)
        self.frame_df = pd.read_parquet(self.paths.data_path)
        if limit is not None:
            self.frame_df = self.frame_df.iloc[:limit].reset_index(drop=True)

        self.include_images = include_images
        self.cameras = cameras or camera_features(self.schema)
        self.video_decoder = (
            VideoFrameDecoder(self.paths.snapshot_dir, self.cameras, image_size=image_size) if include_images else None
        )

    def __len__(self) -> int:
        return len(self.frame_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame_df.iloc[index]
        task_index = int(row["task_index"])
        frame_index = int(row["frame_index"])
        sample = {
            "state": torch.as_tensor(np.asarray(row["observation.state"]).copy(), dtype=torch.float32),
            "effort": torch.as_tensor(np.asarray(row["observation.effort"]).copy(), dtype=torch.float32),
            "action": torch.as_tensor(np.asarray(row["action"]).copy(), dtype=torch.float32),
            "episode_index": int(row["episode_index"]),
            "frame_index": frame_index,
            "timestamp": float(row["timestamp"]),
            "done": bool(row["next.done"]),
            "index": int(row["index"]),
            "task_index": task_index,
            "task_text": self.schema.task_by_index.get(task_index, ""),
        }
        if self.video_decoder is not None:
            sample["images"] = self.video_decoder.decode(frame_index)
        return sample
