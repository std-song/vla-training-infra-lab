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


def _frame_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        return frame
    data = getattr(frame, "data", None)
    if isinstance(data, torch.Tensor):
        return data
    raise TypeError(f"Unsupported TorchCodec frame type: {type(frame)!r}")


@dataclass(frozen=True)
class VideoFrameRef:
    camera: str
    chunk_index: int
    file_index: int
    video_timestamp: float
    video_path: Path


class VideoShardResolver:
    """Resolve LeRobot episode/frame rows to camera video shards."""

    def __init__(self, paths: LeRobotCachePaths, cameras: list[str]):
        if paths.episodes_path is None:
            raise FileNotFoundError("Missing meta/episodes parquet; cannot resolve video shards.")
        self.snapshot_dir = paths.snapshot_dir
        self.cameras = cameras
        episode_df = pd.read_parquet(paths.episodes_path)
        self.episode_by_index = {int(row["episode_index"]): row for _, row in episode_df.iterrows()}

    def resolve(self, camera: str, episode_index: int, timestamp: float) -> VideoFrameRef:
        row = self.episode_by_index[episode_index]
        prefix = f"videos/{camera}"
        chunk_index = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        from_timestamp = float(row[f"{prefix}/from_timestamp"])
        video_timestamp = from_timestamp + timestamp
        video_path = self.snapshot_dir / "videos" / camera / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing video shard for {camera}: {video_path}. "
                "Download the required LeRobot video file before enabling images."
            )
        return VideoFrameRef(
            camera=camera,
            chunk_index=chunk_index,
            file_index=file_index,
            video_timestamp=video_timestamp,
            video_path=video_path,
        )


class VideoFrameDecoder:
    """Lazy TorchCodec decoder wrapper for LeRobot video shards."""

    def __init__(self, paths: LeRobotCachePaths, cameras: list[str], image_size: int | None = None):
        self.resolver = VideoShardResolver(paths, cameras)
        self.cameras = cameras
        self.image_size = image_size
        self._decoders: dict[Path, Any] = {}

    def _decoder(self, video_path: Path) -> Any:
        if video_path not in self._decoders:
            from torchcodec.decoders import VideoDecoder

            self._decoders[video_path] = VideoDecoder(str(video_path))
        return self._decoders[video_path]

    def decode(self, episode_index: int, timestamp: float) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        for camera in self.cameras:
            ref = self.resolver.resolve(camera, episode_index, timestamp)
            frame = _frame_tensor(self._decoder(ref.video_path).get_frame_played_at(ref.video_timestamp))
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
    Set `include_images=True` to resolve video shards from `meta/episodes` and decode RGB frames.
    """

    def __init__(
        self,
        cache_root: str | Path,
        repo_id: str = "lerobot/aloha_mobile_cabinet",
        limit: int | None = None,
        start_index: int = 0,
        include_images: bool = False,
        cameras: list[str] | None = None,
        image_size: int | None = None,
    ):
        self.paths = find_lerobot_snapshot(cache_root, repo_id)
        self.schema = load_schema(self.paths, repo_id)
        frame_df = pd.read_parquet(self.paths.data_path)
        stop_index = None if limit is None else start_index + limit
        self.frame_df = frame_df.iloc[start_index:stop_index].reset_index(drop=True)

        self.include_images = include_images
        self.cameras = cameras or camera_features(self.schema)
        self.video_decoder = VideoFrameDecoder(self.paths, self.cameras, image_size=image_size) if include_images else None

    def __len__(self) -> int:
        return len(self.frame_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame_df.iloc[index]
        task_index = int(row["task_index"])
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        timestamp = float(row["timestamp"])
        sample = {
            "state": torch.as_tensor(np.asarray(row["observation.state"]).copy(), dtype=torch.float32),
            "effort": torch.as_tensor(np.asarray(row["observation.effort"]).copy(), dtype=torch.float32),
            "action": torch.as_tensor(np.asarray(row["action"]).copy(), dtype=torch.float32),
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "done": bool(row["next.done"]),
            "index": int(row["index"]),
            "task_index": task_index,
            "task_text": self.schema.task_by_index.get(task_index, ""),
        }
        if self.video_decoder is not None:
            sample["images"] = self.video_decoder.decode(episode_index, timestamp)
        return sample
