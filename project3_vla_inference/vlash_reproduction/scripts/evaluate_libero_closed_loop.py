"""Closed-loop LIBERO evaluation for delayed Pi0.5/VLASH policies.

The evaluator models inference delay in environment ticks rather than sleeping.
While a request is pending, LIBERO executes actions already present in the queue.
At the handoff tick, the newly predicted action chunk replaces the old suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class as get_lerobot_policy_class
from vlash.policies.factory import get_policy_class

from libero_state_predictor import ActionSequenceStatePredictor, predict_future_state


@dataclass
class EpisodeResult:
    condition: str
    suite: str
    task_id: int
    task: str
    episode_index: int
    seed: int
    delay_ticks: int
    delay_ms: int
    success: bool
    reward: float
    steps: int
    policy_calls: int
    mean_inference_ms: float
    p95_inference_ms: float
    mean_state_prediction_mse: float
    mean_handoff_action_l2: float
    queue_underflows: int


def load_predictor(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ActionSequenceStatePredictor(**config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    return model, stats


def policy_batch(
    observation: dict,
    task: str,
    device: torch.device,
    policy,
    state: torch.Tensor | None = None,
):
    # LeRobot 0.4.1 rotates LIBERO frames with ``[::-1, ::-1]``, which
    # produces negative NumPy strides that torch.from_numpy cannot consume.
    contiguous_observation = dict(observation)
    if isinstance(observation.get("pixels"), dict):
        contiguous_observation["pixels"] = {
            key: np.ascontiguousarray(value) for key, value in observation["pixels"].items()
        }
    batch = preprocess_observation(contiguous_observation)
    batch["task"] = [task]
    if state is not None:
        batch["observation.state"] = state.reshape(1, -1).float().cpu()
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    # The upstream LeRobot PI0.5 expects tokenized language in the batch,
    # whereas VLASH's policy owns a tokenizer and consumes ``task`` directly.
    if not hasattr(policy, "language_tokenizer"):
        tokenizer = getattr(policy, "_closed_loop_tokenizer", None)
        if tokenizer is None:
            tokenizer_path = os.environ.get(
                "VLASH_PALIGEMMA_TOKENIZER_PATH", "google/paligemma-3b-pt-224"
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            policy._closed_loop_tokenizer = tokenizer
        prompt = f"Task: {task.rstrip()};\nAction: "
        tokenized = tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(device).bool()
    return batch


def infer_chunk(policy, batch: dict, seed: int) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    actions_np = actions[0].detach().float().cpu().numpy()
    return np.clip(actions_np, -1.0, 1.0), elapsed_ms


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def run_episode(
    policy,
    predictor,
    predictor_stats,
    condition: str,
    suite_name: str,
    task_id: int,
    episode_index: int,
    seed: int,
    delay_ticks: int,
    replan_interval: int,
    max_steps: int | None,
    device: torch.device,
) -> EpisodeResult:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        episode_index=episode_index,
    )
    observation, _ = env.reset(seed=seed)
    task = env.task_description
    policy.reset()
    action_queue: deque[np.ndarray] = deque()
    inference_ms: list[float] = []
    state_prediction_mse: list[float] = []
    handoff_action_l2: list[float] = []
    queue_underflows = 0
    policy_calls = 0
    total_reward = 0.0
    success = False
    pending: dict | None = None

    # Bootstrap the controller synchronously so delayed modes have actions to
    # execute while the first asynchronous refresh is pending.
    first_batch = policy_batch(observation, task, device, policy)
    initial_chunk, elapsed = infer_chunk(policy, first_batch, seed)
    action_queue.extend(initial_chunk)
    inference_ms.append(elapsed)
    policy_calls += 1
    episode_limit = min(max_steps or env._max_episode_steps, env._max_episode_steps)

    try:
        for step in range(episode_limit):
            # A completed request takes over at this logical control tick.
            if pending is not None and step >= pending["ready_step"]:
                actual_state = np.asarray(observation["agent_pos"], dtype=np.float32)
                if pending["predicted_state"] is not None:
                    delta = pending["predicted_state"] - actual_state
                    state_prediction_mse.append(float(np.mean(delta * delta)))
                old_next = action_queue[0] if action_queue else None
                new_chunk = pending["actions"]
                if condition == "standard_skip":
                    new_chunk = new_chunk[delay_ticks:]
                    if len(new_chunk) == 0:
                        raise RuntimeError(
                            f"Cannot skip {delay_ticks} actions from a chunk of "
                            f"length {len(pending['actions'])}"
                        )
                if old_next is not None:
                    handoff_action_l2.append(float(np.linalg.norm(new_chunk[0] - old_next)))
                action_queue.clear()
                action_queue.extend(new_chunk)
                pending = None

            # Refresh every replan_interval executed actions. The image/task are
            # from launch time; only learned mode compensates the delayed state.
            if step > 0 and step % replan_interval == 0 and pending is None:
                current_state = torch.as_tensor(observation["agent_pos"], device=device).float().reshape(1, -1)
                conditioned_state = current_state
                predicted_state_np = None
                if condition == "learned" and delay_ticks > 0:
                    if predictor is None or predictor_stats is None:
                        raise RuntimeError("The learned condition requires a future-state predictor")
                    if len(action_queue) < delay_ticks:
                        raise RuntimeError(
                            f"Need {delay_ticks} queued actions for future-state prediction, got {len(action_queue)}"
                        )
                    prefix = np.stack(list(action_queue)[:delay_ticks])
                    padded = np.zeros((predictor.max_horizon, prefix.shape[-1]), dtype=np.float32)
                    padded[:delay_ticks] = prefix
                    action_prefix = torch.from_numpy(padded).to(device).unsqueeze(0)
                    horizon = torch.tensor([delay_ticks], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        conditioned_state = predict_future_state(
                            predictor, current_state, action_prefix, horizon, predictor_stats
                        )
                    predicted_state_np = conditioned_state[0].detach().float().cpu().numpy()

                batch = policy_batch(observation, task, device, policy, conditioned_state[0])
                chunk, elapsed = infer_chunk(policy, batch, seed + policy_calls * 1009)
                inference_ms.append(elapsed)
                policy_calls += 1
                if delay_ticks == 0:
                    old_next = action_queue[0] if action_queue else None
                    if old_next is not None:
                        handoff_action_l2.append(float(np.linalg.norm(chunk[0] - old_next)))
                    action_queue.clear()
                    action_queue.extend(chunk)
                else:
                    pending = {
                        "ready_step": step + delay_ticks,
                        "actions": chunk,
                        "predicted_state": predicted_state_np,
                    }

            if not action_queue:
                queue_underflows += 1
                action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            else:
                action = action_queue.popleft().astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info.get("is_success", False))
            if terminated or truncated:
                steps = step + 1
                break
        else:
            steps = episode_limit
    finally:
        env.close()

    return EpisodeResult(
        condition=condition,
        suite=suite_name,
        task_id=task_id,
        task=task,
        episode_index=episode_index,
        seed=seed,
        delay_ticks=delay_ticks,
        delay_ms=delay_ticks * 100,
        success=success,
        reward=total_reward,
        steps=steps,
        policy_calls=policy_calls,
        mean_inference_ms=float(np.mean(inference_ms)),
        p95_inference_ms=percentile(inference_ms, 95),
        mean_state_prediction_mse=float(np.mean(state_prediction_mse)) if state_prediction_mse else float("nan"),
        mean_handoff_action_l2=float(np.mean(handoff_action_l2)) if handoff_action_l2 else float("nan"),
        queue_underflows=queue_underflows,
    )


def append_result(path: Path, result: EpisodeResult) -> None:
    row = asdict(result)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-policy", type=Path)
    parser.add_argument("--stale-policy", type=Path)
    parser.add_argument("--learned-policy", type=Path)
    parser.add_argument("--standard-loader", choices=["vlash", "lerobot"], default="vlash")
    parser.add_argument("--stale-loader", choices=["vlash", "lerobot"], default="vlash")
    parser.add_argument("--learned-loader", choices=["vlash", "lerobot"], default="vlash")
    parser.add_argument("--predictor-path", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial"])
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "standard_sync",
            "standard_naive",
            "standard_skip",
            "sync",
            "stale",
            "learned",
            "learned_stale",
        ],
        default=["sync", "stale", "learned", "learned_stale"],
    )
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if "learned" in args.conditions and max(args.delays) > 4:
        raise ValueError("The trained predictor supports at most four delay ticks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "episodes.csv"
    device = torch.device("cuda")
    predictor = None
    predictor_stats = None
    if "learned" in args.conditions:
        if args.predictor_path is None:
            parser.error("--predictor-path is required for the learned condition")
        predictor, predictor_stats = load_predictor(args.predictor_path, device)

    grouped_conditions = {
        "standard": [name for name in args.conditions if name.startswith("standard_")],
        "stale": [name for name in args.conditions if name in ("sync", "stale")],
        "learned": [name for name in args.conditions if name in ("learned", "learned_stale")],
    }
    for policy_kind, conditions in grouped_conditions.items():
        if not conditions:
            continue
        policy_path = getattr(args, f"{policy_kind}_policy")
        loader = getattr(args, f"{policy_kind}_loader")
        if policy_path is None:
            parser.error(f"--{policy_kind}-policy is required for {conditions}")
        policy_class = get_policy_class("pi05") if loader == "vlash" else get_lerobot_policy_class("pi05")
        if loader == "lerobot":
            policy_config = PreTrainedConfig.from_pretrained(policy_path)
            policy_config.compile_model = False
            policy = policy_class.from_pretrained(policy_path, config=policy_config).eval().to(device)
        else:
            policy = policy_class.from_pretrained(policy_path).eval().to(device)
        for condition in conditions:
            delays = [0] if condition in ("sync", "standard_sync") else args.delays
            effective_condition = "stale" if condition == "learned_stale" else condition
            for suite in args.suites:
                for task_id in args.task_ids:
                    for delay in delays:
                        for episode_index in range(args.episodes):
                            result = run_episode(
                                policy=policy,
                                predictor=predictor,
                                predictor_stats=predictor_stats,
                                condition=effective_condition,
                                suite_name=suite,
                                task_id=task_id,
                                episode_index=episode_index,
                                seed=args.seed + episode_index,
                                delay_ticks=delay,
                                replan_interval=args.replan_interval,
                                max_steps=args.max_steps,
                                device=device,
                            )
                            result.condition = condition
                            append_result(result_path, result)
                            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        del policy
        torch.cuda.empty_cache()

    rows = list(csv.DictReader(result_path.open(encoding="utf-8")))
    summary = {}
    for row in rows:
        key = f"{row['condition']}|{row['suite']}|task{row['task_id']}|delay{row['delay_ticks']}"
        summary.setdefault(key, []).append(row)
    aggregated = {
        key: {
            "episodes": len(values),
            "success_rate": float(np.mean([value["success"] == "True" for value in values])),
            "mean_reward": float(np.mean([float(value["reward"]) for value in values])),
            "mean_steps": float(np.mean([float(value["steps"]) for value in values])),
            "mean_inference_ms": float(np.mean([float(value["mean_inference_ms"]) for value in values])),
        }
        for key, values in summary.items()
    }
    (args.out_dir / "summary.json").write_text(json.dumps(aggregated, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
