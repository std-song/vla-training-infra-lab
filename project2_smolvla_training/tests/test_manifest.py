from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from smolvla_nanotron.data.manifest import (
    analyze_trajectory_dynamics,
    build_manifest,
    statistical_action_jump_issues,
    validate_dataset,
    write_manifest,
    write_quality_report,
)
from smolvla_nanotron.data.sampler import EpisodeAwareSampler, SampleIdentity


def make_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    (snapshot / "meta" / "episodes").mkdir(parents=True)
    (snapshot / "data" / "chunk-000").mkdir(parents=True)
    (snapshot / "videos" / "observation.images.cam_high" / "chunk-000").mkdir(parents=True)
    (snapshot / "videos" / "observation.images.cam_high" / "chunk-000" / "file-000.mp4").touch()
    (snapshot / "meta" / "info.json").write_text(json.dumps({"features": {
        "observation.images.cam_high": {"dtype": "video"},
        "observation.state": {"shape": [2]}, "action": {"shape": [2]},
    }}), encoding="utf-8")
    pd.DataFrame({"task": ["open cabinet"], "task_index": [0]}).to_parquet(snapshot / "meta" / "tasks.parquet")
    pd.DataFrame({
        "episode_index": [0],
        "videos/observation.images.cam_high/chunk_index": [0],
        "videos/observation.images.cam_high/file_index": [0],
        "videos/observation.images.cam_high/from_timestamp": [0.0],
    }).to_parquet(snapshot / "meta" / "episodes" / "chunk-000.parquet")
    pd.DataFrame({
        "index": [0, 1], "episode_index": [0, 0], "frame_index": [0, 1], "timestamp": [0.0, 0.02],
        "task_index": [0, 0], "observation.state": [np.array([0.0, 1.0]), np.array([1.0, 2.0])],
        "action": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
    }).to_parquet(snapshot / "data" / "chunk-000" / "file-000.parquet")
    return snapshot


def test_manifest_and_quality_report(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    found_snapshot, entries, frames, info = build_manifest(snapshot)
    issues = validate_dataset(found_snapshot, entries, frames, info)

    assert found_snapshot == snapshot
    assert len(entries) == 2
    assert entries[0].task_text == "open cabinet"
    assert not issues
    assert write_manifest(entries, tmp_path / "manifest.jsonl").read_text(encoding="utf-8").count("\n") == 2
    report = json.loads(write_quality_report(snapshot, entries, issues, tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["valid_samples"] == 2


def test_validator_catches_bad_action_and_missing_video(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    data_path = snapshot / "data" / "chunk-000" / "file-000.parquet"
    frame = pd.read_parquet(data_path)
    frame.at[1, "action"] = np.array([np.nan, 0.0])
    frame.to_parquet(data_path)
    (snapshot / "videos" / "observation.images.cam_high" / "chunk-000" / "file-000.mp4").unlink()

    found_snapshot, entries, frames, info = build_manifest(snapshot)
    codes = {issue.code for issue in validate_dataset(found_snapshot, entries, frames, info)}
    assert {"non_finite_feature", "missing_video_shard"}.issubset(codes)


def test_episode_aware_sampler_is_disjoint_deterministic_and_resumable() -> None:
    samples = [SampleIdentity(index=index, episode_index=episode) for episode, size in enumerate((3, 3, 4)) for index in range(sum((3, 3, 4)[:episode]), sum((3, 3, 4)[: episode + 1]))]
    rank0 = EpisodeAwareSampler(samples, rank=0, world_size=2, seed=17, epoch=2)
    rank1 = EpisodeAwareSampler(samples, rank=1, world_size=2, seed=17, epoch=2)
    first = rank0.indices()
    second = rank1.indices()

    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == set(range(10))
    assert first == EpisodeAwareSampler(samples, rank=0, world_size=2, seed=17, epoch=2).indices()
    resumed = EpisodeAwareSampler.from_state_dict(samples, rank0.state_dict(consumed_samples=2))
    assert resumed.indices() == first[2:]
    assert list(iter(rank0)) == first
    assert len(rank0) == len(first)


def test_dynamics_and_statistical_action_jump_detection(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    data_path = snapshot / "data" / "chunk-000" / "file-000.parquet"
    frame = pd.read_parquet(data_path)
    frame.at[1, "action"] = np.array([100.0, 100.0])
    frame.to_parquet(data_path)
    _, _, frames, _ = build_manifest(snapshot)

    dynamics = analyze_trajectory_dynamics(frames)
    assert dynamics["action_delta_l2_count"] == 1
    assert dynamics["action_delta_l2_max"] > 100

    # Create several natural-scale deltas so the robust scale is non-zero,
    # followed by one obvious outlier.
    repeated = pd.DataFrame({
        "episode_index": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "frame_index": [0, 1] * 5,
        "index": list(range(10)),
        "action": [
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([0.9, 0.0]),
            np.array([0.0, 0.0]), np.array([1.2, 0.0]),
            np.array([0.0, 0.0]), np.array([100.0, 100.0]),
        ],
    })
    assert any(issue.code == "statistical_action_jump" for issue in statistical_action_jump_issues(repeated, mad_multiplier=2.0))
