from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen2 decode with and without KV cache.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="project3_vla_infer/results/qwen2_kv_cache_compare_sdpa_bf16.csv")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--prompt-lengths", default="128,512")
    parser.add_argument("--decode-lengths", default="16,32")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_ms(fn):
    sync()
    start = time.perf_counter()
    output = fn()
    sync()
    return (time.perf_counter() - start) * 1000.0, output


def make_inputs(tokenizer, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    text = "You are a robot policy. Predict the next robot action from the current task and scene. " * 256
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if input_ids.shape[1] < seq_len:
        pad_len = seq_len - input_ids.shape[1]
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = torch.cat([input_ids, torch.full((1, pad_len), pad_id, dtype=input_ids.dtype)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, pad_len), dtype=attention_mask.dtype)], dim=1)
    return input_ids[:, :seq_len].repeat(batch_size, 1).to(device), attention_mask[:, :seq_len].repeat(batch_size, 1).to(device)


@torch.inference_mode()
def decode_with_cache(model, token: torch.Tensor, attention_mask: torch.Tensor, past_key_values, steps: int) -> None:
    mask = attention_mask
    past = past_key_values
    next_token = token
    for _ in range(steps):
        outputs = model(input_ids=next_token, attention_mask=mask, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)], dim=1)


@torch.inference_mode()
def decode_without_cache(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> None:
    ids = input_ids
    mask = attention_mask
    for _ in range(steps):
        outputs = model(input_ids=ids, attention_mask=mask, use_cache=False)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        ids = torch.cat([ids, next_token], dim=1)
        mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)], dim=1)


@torch.inference_mode()
def run_case(model, tokenizer, batch_size: int, prompt_len: int, decode_len: int, repeat: int, device: torch.device) -> dict[str, float | int]:
    input_ids, attention_mask = make_inputs(tokenizer, batch_size, prompt_len, device)

    warmup = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    token = warmup.logits[:, -1:].argmax(dim=-1)
    decode_with_cache(model, token, attention_mask, warmup.past_key_values, min(4, decode_len))
    decode_without_cache(model, input_ids, attention_mask, min(2, decode_len))
    sync()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cached_ms = []
    nocache_ms = []
    for _ in range(repeat):
        prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        token = prefill.logits[:, -1:].argmax(dim=-1)
        cached_time, _ = timed_ms(lambda: decode_with_cache(model, token, attention_mask, prefill.past_key_values, decode_len))
        nocache_time, _ = timed_ms(lambda: decode_without_cache(model, input_ids, attention_mask, decode_len))
        cached_ms.append(cached_time)
        nocache_ms.append(nocache_time)

    cached_avg = statistics.mean(cached_ms)
    nocache_avg = statistics.mean(nocache_ms)
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "cached_decode_ms": cached_avg,
        "nocache_decode_ms": nocache_avg,
        "cached_tpot_ms": cached_avg / decode_len,
        "nocache_tpot_ms": nocache_avg / decode_len,
        "cached_tokens_per_s": batch_size * decode_len / (cached_avg / 1000.0),
        "nocache_tokens_per_s": batch_size * decode_len / (nocache_avg / 1000.0),
        "speedup": nocache_avg / cached_avg,
        "max_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()

    rows = []
    for batch_size in parse_int_list(args.batch_sizes):
        for prompt_len in parse_int_list(args.prompt_lengths):
            for decode_len in parse_int_list(args.decode_lengths):
                row = run_case(model, tokenizer, batch_size, prompt_len, decode_len, args.repeat, device)
                row["dtype"] = args.dtype
                row["attn_implementation"] = args.attn_implementation
                rows.append(row)
                print(
                    f"b={batch_size} p={prompt_len} d={decode_len} "
                    f"cache={row['cached_decode_ms']:.1f}ms no_cache={row['nocache_decode_ms']:.1f}ms "
                    f"speedup={row['speedup']:.2f}x cache_tpot={row['cached_tpot_ms']:.2f}ms "
                    f"no_cache_tpot={row['nocache_tpot_ms']:.2f}ms",
                    flush=True,
                )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dtype",
        "attn_implementation",
        "batch_size",
        "prompt_len",
        "decode_len",
        "cached_decode_ms",
        "nocache_decode_ms",
        "cached_tpot_ms",
        "nocache_tpot_ms",
        "cached_tokens_per_s",
        "nocache_tokens_per_s",
        "speedup",
        "max_memory_mib",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
