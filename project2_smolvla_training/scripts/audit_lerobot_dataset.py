"""Build a LeRobot training manifest and reject malformed samples before training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Permit ``python scripts/audit_lerobot_dataset.py`` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolvla_nanotron.data.manifest import (
    analyze_trajectory_dynamics,
    build_manifest,
    statistical_action_jump_issues,
    validate_dataset,
    write_manifest,
    write_quality_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="LeRobot snapshot directory or Hugging Face cache root")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset_audit"))
    parser.add_argument("--skip-video-existence-check", action="store_true")
    parser.add_argument("--action-jump-mad-multiplier", type=float, default=None,
                        help="Optional robust-MAD action-jump review threshold; does not assert physical invalidity.")
    args = parser.parse_args()

    snapshot, entries, frames, info = build_manifest(args.dataset_root)
    issues = validate_dataset(snapshot, entries, frames, info, check_video_files=not args.skip_video_existence_check)
    dynamics = analyze_trajectory_dynamics(frames)
    if args.action_jump_mad_multiplier is not None:
        issues.extend(statistical_action_jump_issues(frames, args.action_jump_mad_multiplier))
    manifest = write_manifest(entries, args.output_dir / "manifest.jsonl")
    report = write_quality_report(snapshot, entries, issues, args.output_dir / "quality_report.json", dynamics=dynamics)
    print(f"snapshot={snapshot}")
    print(f"manifest={manifest}")
    print(f"report={report}")
    print(f"samples={len(entries)} issues={len(issues)}")
    print("dynamics=" + ", ".join(f"{key}={value}" for key, value in dynamics.items()))


if __name__ == "__main__":
    main()
