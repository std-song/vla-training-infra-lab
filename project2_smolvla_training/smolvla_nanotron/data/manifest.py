"""LeRobot dataset manifest construction and training-readiness validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManifestEntry:
    index: int
    episode_index: int
    frame_index: int
    timestamp: float
    task_index: int
    task_text: str
    video_refs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    index: int | None = None
    episode_index: int | None = None


def discover_snapshot(dataset_root: str | Path) -> Path:
    """Accept either a LeRobot snapshot or a Hugging Face cache root."""
    root = Path(dataset_root).expanduser().resolve()
    if (root / "meta" / "info.json").exists() and (root / "data").exists():
        return root

    candidates = [
        path.parent.parent
        for path in root.rglob("info.json")
        if path.parent.name == "meta" and (path.parent.parent / "data").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No LeRobot snapshot found under {root}")
    return sorted(candidates, key=lambda path: len(str(path)))[0]


def _task_map(snapshot: Path) -> dict[int, str]:
    path = snapshot / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    table = pd.read_parquet(path)
    if {"task", "task_index"}.issubset(table.columns):
        return {int(row.task_index): str(row.task) for row in table.itertuples(index=False)}
    if "task_index" in table.columns:
        return {int(row.task_index): str(index) for index, row in table.iterrows()}
    return {}


def _load_frames(snapshot: Path) -> pd.DataFrame:
    files = sorted((snapshot / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data files under {snapshot / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _episode_rows(snapshot: Path) -> dict[int, pd.Series]:
    files = sorted((snapshot / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {snapshot / 'meta' / 'episodes'}")
    table = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if "episode_index" not in table.columns:
        raise ValueError("Episode metadata has no episode_index column")
    return {int(row["episode_index"]): row for _, row in table.iterrows()}


def _camera_names(info: dict[str, Any]) -> list[str]:
    return [name for name, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def _video_refs(snapshot: Path, episode: pd.Series | None, cameras: list[str], timestamp: float) -> dict[str, dict[str, Any]]:
    if episode is None:
        return {}
    refs: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        prefix = f"videos/{camera}"
        required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp"]
        if not all(column in episode.index for column in required):
            continue
        chunk = int(episode[f"{prefix}/chunk_index"])
        file = int(episode[f"{prefix}/file_index"])
        path = Path("videos") / camera / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
        refs[camera] = {
            "path": path.as_posix(),
            "video_timestamp": float(episode[f"{prefix}/from_timestamp"]) + timestamp,
        }
    return refs


def build_manifest(dataset_root: str | Path) -> tuple[Path, list[ManifestEntry], pd.DataFrame, dict[str, Any]]:
    """Build stable sample records without decoding video payloads."""
    snapshot = discover_snapshot(dataset_root)
    info = json.loads((snapshot / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = _task_map(snapshot)
    frames = _load_frames(snapshot)
    episodes = _episode_rows(snapshot)
    cameras = _camera_names(info)

    required = {"episode_index", "frame_index", "timestamp", "task_index"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"Frame parquet misses required columns: {sorted(missing)}")

    entries: list[ManifestEntry] = []
    for row in frames.itertuples(index=False):
        episode_index = int(getattr(row, "episode_index"))
        timestamp = float(getattr(row, "timestamp"))
        entries.append(
            ManifestEntry(
                index=int(getattr(row, "index", len(entries))),
                episode_index=episode_index,
                frame_index=int(getattr(row, "frame_index")),
                timestamp=timestamp,
                task_index=int(getattr(row, "task_index")),
                task_text=tasks.get(int(getattr(row, "task_index")), ""),
                video_refs=_video_refs(snapshot, episodes.get(episode_index), cameras, timestamp),
            )
        )
    return snapshot, entries, frames, info


def validate_dataset(
    snapshot: Path,
    entries: list[ManifestEntry],
    frames: pd.DataFrame,
    info: dict[str, Any],
    *,
    check_video_files: bool = True,
) -> list[ValidationIssue]:
    """Validate data contract violations before a training job starts."""
    issues: list[ValidationIssue] = []
    expected_features = info.get("features", {})
    for position, entry in enumerate(entries):
        row = frames.iloc[position]
        if not np.isfinite(entry.timestamp) or entry.timestamp < 0:
            issues.append(ValidationIssue("invalid_timestamp", "timestamp must be finite and non-negative", entry.index, entry.episode_index))
        if not entry.task_text:
            issues.append(ValidationIssue("missing_task_text", "task_index has no task text mapping", entry.index, entry.episode_index))
        for column in ("observation.state", "action"):
            if column not in frames.columns:
                issues.append(ValidationIssue("missing_feature", f"missing required feature: {column}", entry.index, entry.episode_index))
                continue
            values = np.asarray(row[column])
            expected_shape = tuple(expected_features.get(column, {}).get("shape", values.shape))
            if values.shape != expected_shape:
                issues.append(ValidationIssue("shape_mismatch", f"{column} shape {values.shape}, expected {expected_shape}", entry.index, entry.episode_index))
            if not np.isfinite(values).all():
                issues.append(ValidationIssue("non_finite_feature", f"{column} contains NaN or Inf", entry.index, entry.episode_index))
        for camera, ref in entry.video_refs.items():
            if check_video_files and not (snapshot / ref["path"]).is_file():
                issues.append(ValidationIssue("missing_video_shard", f"missing {camera} shard: {ref['path']}", entry.index, entry.episode_index))

    for episode_index, group in frames.groupby("episode_index", sort=False):
        timestamps = group["timestamp"].to_numpy(dtype=float)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            issues.append(ValidationIssue("non_monotonic_timestamp", "timestamps decrease within episode", episode_index=int(episode_index)))
    return issues


def analyze_trajectory_dynamics(frames: pd.DataFrame) -> dict[str, float | int]:
    """Summarize adjacent state/action changes within each episode.

    These are descriptive statistics, not robot-specific safety limits. They
    establish an evidence-backed threshold before a dataset-specific quality
    gate is enabled.
    """
    action_norms: list[float] = []
    state_norms: list[float] = []
    time_deltas: list[float] = []
    for _, group in frames.groupby("episode_index", sort=False):
        group = group.sort_values("frame_index")
        if len(group) < 2:
            continue
        actions = np.stack(group["action"].map(np.asarray).to_list())
        states = np.stack(group["observation.state"].map(np.asarray).to_list())
        action_norms.extend(np.linalg.norm(np.diff(actions, axis=0), axis=1).tolist())
        state_norms.extend(np.linalg.norm(np.diff(states, axis=0), axis=1).tolist())
        time_deltas.extend(np.diff(group["timestamp"].to_numpy(dtype=float)).tolist())

    def quantiles(values: list[float], prefix: str) -> dict[str, float | int]:
        array = np.asarray(values, dtype=float)
        if not len(array):
            return {f"{prefix}_count": 0}
        return {
            f"{prefix}_count": int(len(array)),
            f"{prefix}_median": float(np.quantile(array, 0.5)),
            f"{prefix}_p95": float(np.quantile(array, 0.95)),
            f"{prefix}_p99": float(np.quantile(array, 0.99)),
            f"{prefix}_max": float(array.max()),
        }

    return {
        **quantiles(action_norms, "action_delta_l2"),
        **quantiles(state_norms, "state_delta_l2"),
        **quantiles(time_deltas, "frame_interval_s"),
    }


def statistical_action_jump_issues(frames: pd.DataFrame, mad_multiplier: float = 12.0) -> list[ValidationIssue]:
    """Flag unusually large adjacent action jumps using a robust global MAD rule.

    This is intentionally opt-in: a statistical outlier needs human review and
    is not automatically a physically invalid robot trajectory.
    """
    if mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be positive")
    deltas: list[tuple[int, int, float]] = []
    for episode_index, group in frames.groupby("episode_index", sort=False):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].map(np.asarray).to_list())
        norms = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        indices = group["index"].to_numpy(dtype=int)[1:]
        deltas.extend((int(index), int(episode_index), float(norm)) for index, norm in zip(indices, norms))
    if not deltas:
        return []
    values = np.asarray([delta for _, _, delta in deltas])
    median = float(np.median(values))
    robust_sigma = float(1.4826 * np.median(np.abs(values - median)))
    if robust_sigma == 0:
        return []
    threshold = median + mad_multiplier * robust_sigma
    return [
        ValidationIssue(
            "statistical_action_jump",
            f"action delta L2={delta:.6g} exceeds robust threshold {threshold:.6g}",
            index=index,
            episode_index=episode_index,
        )
        for index, episode_index, delta in deltas
        if delta > threshold
    ]


def write_manifest(entries: list[ManifestEntry], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_quality_report(
    snapshot: Path,
    entries: list[ManifestEntry],
    issues: list[ValidationIssue],
    output_path: str | Path,
    dynamics: dict[str, float | int] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = pd.Series([issue.code for issue in issues], dtype="object").value_counts().to_dict()
    report = {
        "snapshot": str(snapshot),
        "samples": len(entries),
        "episodes": len({entry.episode_index for entry in entries}),
        "valid_samples": len(entries) - len({issue.index for issue in issues if issue.index is not None}),
        "issue_counts": counts,
        "dynamics": dynamics or {},
        "issues": [asdict(issue) for issue in issues],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
