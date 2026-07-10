"""Deterministic episode-aware sample planning for distributed VLA training."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class SampleIdentity:
    index: int
    episode_index: int


class EpisodeAwareSampler:
    """Assign whole episodes to ranks before shuffling their frames.

    Keeping one episode on one rank prevents two workers from repeatedly opening
    the same camera shards. It is a better default for video-backed VLA data
    than frame-wise random partitioning when episode lengths are comparable.
    """

    def __init__(
        self,
        samples: Iterable[SampleIdentity],
        *,
        rank: int,
        world_size: int,
        seed: int = 0,
        epoch: int = 0,
        start_offset: int = 0,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        self.samples = list(samples)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch
        self.start_offset = start_offset

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.start_offset = 0

    def indices(self) -> list[int]:
        by_episode: dict[int, list[int]] = {}
        for sample in self.samples:
            by_episode.setdefault(sample.episode_index, []).append(sample.index)

        rng = random.Random(self.seed + self.epoch)
        episodes = list(by_episode)
        rng.shuffle(episodes)

        # Greedy least-loaded allocation balances equal-length ALOHA episodes
        # while preserving episode ownership on exactly one rank.
        rank_episodes: list[list[int]] = [[] for _ in range(self.world_size)]
        rank_loads = [0] * self.world_size
        for episode in episodes:
            owner = min(range(self.world_size), key=lambda candidate: (rank_loads[candidate], candidate))
            rank_episodes[owner].append(episode)
            rank_loads[owner] += len(by_episode[episode])

        assigned: list[int] = []
        for episode in rank_episodes[self.rank]:
            frame_indices = list(by_episode[episode])
            rng.shuffle(frame_indices)
            assigned.extend(frame_indices)
        return assigned[self.start_offset :]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.indices())

    def __len__(self) -> int:
        return len(self.indices())

    def state_dict(self, consumed_samples: int) -> dict[str, int]:
        if consumed_samples < 0:
            raise ValueError("consumed_samples must be non-negative")
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "consumed_samples": consumed_samples,
        }

    @classmethod
    def from_state_dict(cls, samples: Iterable[SampleIdentity], state: dict[str, int]) -> "EpisodeAwareSampler":
        return cls(
            samples,
            rank=state["rank"],
            world_size=state["world_size"],
            seed=state["seed"],
            epoch=state["epoch"],
            start_offset=state["consumed_samples"],
        )
