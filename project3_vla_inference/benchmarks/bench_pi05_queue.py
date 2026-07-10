import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def build_language_tokens(cfg: dict, device: str, tokenizer_name_or_path: str | None, task: str):
    if tokenizer_name_or_path is None:
        return (
            torch.zeros(1, int(cfg["tokenizer_max_length"]), dtype=torch.long, device=device),
            torch.ones(1, int(cfg["tokenizer_max_length"]), dtype=torch.bool, device=device),
        )

    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    state = np.zeros((1, int(cfg["max_state_dim"])), dtype=np.float32)
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized[0]))
    prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    encoded = tokenizer(
        [prompt],
        max_length=int(cfg["tokenizer_max_length"]),
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device=device, dtype=torch.bool)


def build_batch(cfg: dict, device: str, tokenizer_name_or_path: str | None, task: str) -> dict:
    height, width = cfg["image_resolution"]
    tokens, mask = build_language_tokens(cfg, device, tokenizer_name_or_path, task)
    batch = {
        "observation.language.tokens": tokens,
        "observation.language.attention_mask": mask,
    }
    for name in cfg["input_features"]:
        if name.startswith("observation.images."):
            batch[name] = torch.zeros(1, 3, height, width, dtype=torch.float32, device=device)
    return batch


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pi0.5 select_action queue amortization with PI05Policy.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup-chunks", type=int, default=4)
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--out", default="project3_vla_infer/results/pi05_policy_queue_benchmark.csv")
    parser.add_argument("--summary-out", default="project3_vla_infer/results/pi05_policy_queue_benchmark_summary.csv")
    parser.add_argument("--tokenizer-name-or-path", default=None)
    parser.add_argument("--task", default="open the cabinet")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    policy = PI05Policy.from_pretrained(str(model_dir), strict=args.strict)
    policy.eval().to("cuda")
    batch = build_batch(cfg, "cuda", args.tokenizer_name_or_path, args.task)

    for _ in range(args.warmup_chunks):
        timed(lambda: policy.predict_action_chunk(batch))

    policy.reset()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(args.steps):
        action, latency_ms = timed(lambda: policy.select_action(batch))
        rows.append({"step": step, "latency_ms": round(latency_ms, 3), "action_shape": str(tuple(action.shape))})

    full = [row["latency_ms"] for row in rows if row["latency_ms"] > 20]
    queue = [row["latency_ms"] for row in rows if row["latency_ms"] <= 20]
    summary = {
        "full_model_calls": len(full),
        "full_avg_ms": round(statistics.mean(full), 3),
        "full_p50_ms": round(statistics.median(full), 3),
        "queue_pops": len(queue),
        "queue_avg_ms": round(statistics.mean(queue), 3),
        "queue_p50_ms": round(statistics.median(queue), 3),
        "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    print(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "latency_ms", "action_shape"])
        writer.writeheader()
        writer.writerows(rows)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()

