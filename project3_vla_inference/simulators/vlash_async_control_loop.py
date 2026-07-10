from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Scenario:
    name: str
    async_inference: bool
    future_state: bool
    action_quant_ratio: int = 1


@dataclass
class PendingInference:
    complete_step: int
    observation_step: int
    chunk: list[float]
    predicted_state_at_completion: float


@dataclass
class QueueAction:
    value: float
    observation_step: int
    intended_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLASH-inspired async VLA control-loop simulator.")
    parser.add_argument("--out-summary", default="project3_vla_infer/results/vlash_async_control_loop_summary.csv")
    parser.add_argument("--out-trace", default="project3_vla_infer/results/vlash_async_control_loop_trace.csv")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--policy-latency-ms", type=float, default=87.65)
    parser.add_argument("--queue-pop-ms", type=float, default=3.467)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--refill-threshold", type=int, default=8)
    parser.add_argument("--target-change-s", type=float, default=2.0)
    parser.add_argument("--plant-gain", type=float, default=1.0)
    parser.add_argument("--policy-kp", type=float, default=2.8)
    return parser.parse_args()


def target_at(step: int, change_step: int) -> float:
    return 0.0 if step < change_step else 1.0


def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_action_chunk(
    state: float,
    start_step: int,
    chunk_size: int,
    change_step: int,
    dt_s: float,
    policy_kp: float,
    plant_gain: float,
    quant_ratio: int,
) -> list[float]:
    pred = state
    actions: list[float] = []
    macro_action = 0.0
    for offset in range(chunk_size):
        if offset % quant_ratio == 0:
            target = target_at(start_step + offset, change_step)
            macro_action = clip(policy_kp * (target - pred))
        actions.append(macro_action)
        pred += macro_action * plant_gain * dt_s
    return actions


def rollout_state(state: float, queued: list[QueueAction], steps: int, dt_s: float, plant_gain: float) -> float:
    pred = state
    for idx in range(steps):
        action = queued[idx].value if idx < len(queued) else 0.0
        pred += action * plant_gain * dt_s
    return pred


def first_reaction_latency_ms(trace: list[dict[str, float | str | int]], change_step: int, dt_ms: float) -> float:
    for row in trace:
        step = int(row["step"])
        if step >= change_step and float(row["action"]) > 0.05:
            return (step - change_step) * dt_ms
    return math.nan


def summarize(
    scenario: Scenario,
    trace: list[dict[str, float | str | int]],
    full_model_calls: int,
    generated_actions: int,
    queue_pop_ms: float,
    policy_latency_ms: float,
    change_step: int,
    dt_ms: float,
) -> dict[str, float | str | int]:
    post_change = [row for row in trace if int(row["step"]) >= change_step]
    errors = [abs(float(row["target"]) - float(row["state"])) for row in post_change]
    staleness = [float(row["state_staleness_ms"]) for row in trace if float(row["state_staleness_ms"]) >= 0.0]
    stalls = [row for row in trace if int(row["stalled"]) == 1]
    queue_pops = [row for row in trace if int(row["queue_pop"]) == 1]
    control_ticks = len(trace)
    control_overhead_ms = len(queue_pops) * queue_pop_ms / scenario.action_quant_ratio
    blocking_infer_ms = full_model_calls * policy_latency_ms if not scenario.async_inference else len(stalls) * dt_ms
    return {
        "scenario": scenario.name,
        "control_hz": 1000.0 / dt_ms,
        "policy_latency_ms": policy_latency_ms,
        "chunk_size": max(1, generated_actions // max(full_model_calls, 1)),
        "action_quant_ratio": scenario.action_quant_ratio,
        "full_model_calls": full_model_calls,
        "generated_actions": generated_actions,
        "executed_actions": len(queue_pops),
        "mean_abs_error": statistics.mean(errors),
        "p95_abs_error": sorted(errors)[max(0, math.ceil(len(errors) * 0.95) - 1)],
        "reaction_latency_ms": first_reaction_latency_ms(trace, change_step, dt_ms),
        "stall_ratio": len(stalls) / control_ticks,
        "mean_state_staleness_ms": statistics.mean(staleness) if staleness else 0.0,
        "p95_state_staleness_ms": sorted(staleness)[max(0, math.ceil(len(staleness) * 0.95) - 1)] if staleness else 0.0,
        "control_overhead_ms": control_overhead_ms,
        "blocking_or_empty_queue_ms": blocking_infer_ms,
    }


def run_sync(args: argparse.Namespace, scenario: Scenario) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    dt_s = 1.0 / args.control_hz
    dt_ms = 1000.0 * dt_s
    total_steps = int(args.duration_s * args.control_hz)
    change_step = int(args.target_change_s * args.control_hz)
    infer_steps = max(1, math.ceil(args.policy_latency_ms / dt_ms))
    step = 0
    state = 0.0
    full_model_calls = 0
    generated_actions = 0
    trace: list[dict[str, float | str | int]] = []

    while step < total_steps:
        observation_step = step
        chunk = build_action_chunk(
            state,
            step + infer_steps,
            args.chunk_size,
            change_step,
            dt_s,
            args.policy_kp,
            args.plant_gain,
            scenario.action_quant_ratio,
        )
        full_model_calls += 1
        generated_actions += len(chunk)

        for _ in range(infer_steps):
            if step >= total_steps:
                break
            target = target_at(step, change_step)
            trace.append(
                {
                    "scenario": scenario.name,
                    "step": step,
                    "time_s": step * dt_s,
                    "state": state,
                    "target": target,
                    "action": 0.0,
                    "queue_len": 0,
                    "stalled": 1,
                    "queue_pop": 0,
                    "state_staleness_ms": -1.0,
                }
            )
            step += 1

        for offset, action in enumerate(chunk):
            if step >= total_steps:
                break
            target = target_at(step, change_step)
            state += action * args.plant_gain * dt_s
            trace.append(
                {
                    "scenario": scenario.name,
                    "step": step,
                    "time_s": step * dt_s,
                    "state": state,
                    "target": target,
                    "action": action,
                    "queue_len": max(0, len(chunk) - offset - 1),
                    "stalled": 0,
                    "queue_pop": 1,
                    "state_staleness_ms": (step - observation_step) * dt_ms,
                }
            )
            step += 1

    return summarize(scenario, trace, full_model_calls, generated_actions, args.queue_pop_ms, args.policy_latency_ms, change_step, dt_ms), trace


def run_async(args: argparse.Namespace, scenario: Scenario) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    dt_s = 1.0 / args.control_hz
    dt_ms = 1000.0 * dt_s
    total_steps = int(args.duration_s * args.control_hz)
    change_step = int(args.target_change_s * args.control_hz)
    infer_steps = max(1, math.ceil(args.policy_latency_ms / dt_ms))
    state = 0.0
    queue: list[QueueAction] = []
    pending: PendingInference | None = None
    full_model_calls = 0
    generated_actions = 0
    trace: list[dict[str, float | str | int]] = []

    def maybe_start_inference(step: int) -> None:
        nonlocal pending, full_model_calls, generated_actions
        if pending is not None or len(queue) > args.refill_threshold:
            return
        completion_step = step + infer_steps
        if scenario.future_state:
            policy_state = rollout_state(state, queue, infer_steps, dt_s, args.plant_gain)
            chunk_start_step = completion_step
            observation_step = completion_step
        else:
            policy_state = state
            chunk_start_step = step
            observation_step = step
        chunk = build_action_chunk(
            policy_state,
            chunk_start_step,
            args.chunk_size,
            change_step,
            dt_s,
            args.policy_kp,
            args.plant_gain,
            scenario.action_quant_ratio,
        )
        pending = PendingInference(completion_step, observation_step, chunk, policy_state)
        full_model_calls += 1
        generated_actions += len(chunk)

    for step in range(total_steps):
        if pending is not None and pending.complete_step <= step:
            for offset, action in enumerate(pending.chunk):
                queue.append(QueueAction(action, pending.observation_step, step + offset))
            pending = None

        maybe_start_inference(step)

        if queue:
            item = queue.pop(0)
            action = item.value
            stalled = 0
            queue_pop = 1
            staleness_ms = max(0.0, (step - item.observation_step) * dt_ms)
        else:
            action = 0.0
            stalled = 1
            queue_pop = 0
            staleness_ms = -1.0

        state += action * args.plant_gain * dt_s
        trace.append(
            {
                "scenario": scenario.name,
                "step": step,
                "time_s": step * dt_s,
                "state": state,
                "target": target_at(step, change_step),
                "action": action,
                "queue_len": len(queue),
                "stalled": stalled,
                "queue_pop": queue_pop,
                "state_staleness_ms": staleness_ms,
            }
        )

    return summarize(scenario, trace, full_model_calls, generated_actions, args.queue_pop_ms, args.policy_latency_ms, change_step, dt_ms), trace


def main() -> None:
    args = parse_args()
    scenarios = [
        Scenario("sync_chunk_blocking", async_inference=False, future_state=False),
        Scenario("async_naive_queue", async_inference=True, future_state=False),
        Scenario("async_future_state_queue", async_inference=True, future_state=True),
        Scenario("async_future_state_quantized_q2", async_inference=True, future_state=True, action_quant_ratio=2),
    ]
    summaries: list[dict[str, float | str | int]] = []
    traces: list[dict[str, float | str | int]] = []
    for scenario in scenarios:
        if scenario.async_inference:
            summary, trace = run_async(args, scenario)
        else:
            summary, trace = run_sync(args, scenario)
        summaries.append(summary)
        traces.extend(trace)
        print(
            f"{scenario.name}: error={summary['mean_abs_error']:.3f} "
            f"reaction={summary['reaction_latency_ms']:.1f}ms stall={summary['stall_ratio']:.3f} "
            f"stale={summary['mean_state_staleness_ms']:.1f}ms",
            flush=True,
        )

    summary_path = ROOT / args.out_summary
    trace_path = ROOT / args.out_trace
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    summary_columns = list(summaries[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summaries)

    trace_columns = list(traces[0].keys())
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_columns)
        writer.writeheader()
        writer.writerows(traces)

    print(f"wrote {summary_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
