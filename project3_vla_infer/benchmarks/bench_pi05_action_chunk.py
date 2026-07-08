import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def build_language_tokens(cfg: dict, batch_size: int, device: str, tokenizer_name_or_path: str | None, task: str):
    if tokenizer_name_or_path is None:
        return (
            torch.zeros(batch_size, int(cfg["tokenizer_max_length"]), dtype=torch.long, device=device),
            torch.ones(batch_size, int(cfg["tokenizer_max_length"]), dtype=torch.bool, device=device),
        )

    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    state = np.zeros((batch_size, int(cfg["max_state_dim"])), dtype=np.float32)
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    prompts = []
    for i in range(batch_size):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized[i]))
        prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
    encoded = tokenizer(
        prompts,
        max_length=int(cfg["tokenizer_max_length"]),
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device=device, dtype=torch.bool)


def build_batch(cfg: dict, batch_size: int, device: str, tokenizer_name_or_path: str | None, task: str) -> dict:
    height, width = cfg["image_resolution"]
    tokens, mask = build_language_tokens(cfg, batch_size, device, tokenizer_name_or_path, task)
    batch = {
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": mask,
    }
    for name in cfg["input_features"]:
        if name.startswith("observation.images."):
            batch[name] = torch.zeros(batch_size, 3, height, width, dtype=torch.float32, device=device)
    return batch


def timed_call(policy: PI05Policy, batch: dict) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    return actions, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pi0.5 full action chunk inference with PI05Policy.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--out", default="project3_vla_infer/results/pi05_policy_action_benchmark.csv")
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--task", default="open the cabinet")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    device = "cuda"

    policy = PI05Policy.from_pretrained(str(model_dir), strict=args.strict)
    policy.eval().to(device)

    rows = []
    for batch_size in [int(x) for x in args.batch_sizes.split(",")]:
        batch = build_batch(cfg, batch_size, device, args.tokenizer_name_or_path, args.task)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        actions, cold_ms = timed_call(policy, batch)
        for _ in range(args.warmup):
            timed_call(policy, batch)

        warm_latencies = []
        for _ in range(args.repeat):
            _, latency_ms = timed_call(policy, batch)
            warm_latencies.append(latency_ms)

        row = {
            "batch_size": batch_size,
            "action_shape": str(tuple(actions.shape)),
            "cold_ms": round(cold_ms, 3),
            "warm_avg_ms": round(statistics.mean(warm_latencies), 3),
            "warm_p50_ms": round(statistics.median(warm_latencies), 3),
            "warm_min_ms": round(min(warm_latencies), 3),
            "warm_max_ms": round(max(warm_latencies), 3),
            "chunk_size": actions.shape[1],
            "action_dim": actions.shape[2],
            "amortized_ms_per_action": round(statistics.mean(warm_latencies) / actions.shape[1], 3),
            "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        }
        print(row)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", out)


if __name__ == "__main__":
    main()

