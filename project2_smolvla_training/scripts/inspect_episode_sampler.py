"""Inspect rank ownership for a manifest without launching DDP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolvla_nanotron.data.sampler import EpisodeAwareSampler, SampleIdentity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()

    samples = [SampleIdentity(index=row["index"], episode_index=row["episode_index"])
               for row in (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines())]
    for rank in range(args.world_size):
        indices = EpisodeAwareSampler(samples, rank=rank, world_size=args.world_size, seed=args.seed, epoch=args.epoch).indices()
        print(f"rank={rank} samples={len(indices)} first_indices={indices[:10]}")


if __name__ == "__main__":
    main()
