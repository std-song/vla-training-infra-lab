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
    parser = argparse.ArgumentParser(description="Benchmark Qwen2 prefill/decode latency with KV cache.")
    parser.add_argument("--model-dir", required=True, help="Local Hugging Face or ModelScope model directory.")
    parser.add_argument("--out", default="results/qwen2_prefill_decode_sdpa_bf16.csv")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--prompt-lengths", default="128,512,1024")
    parser.add_argument("--decode-lengths", default="32,128")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup-decode", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_ms(fn):
    synchronize()
    start = time.perf_counter()
    output = fn()
    synchronize()
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

    input_ids = input_ids[:, :seq_len].repeat(batch_size, 1).to(device)
    attention_mask = attention_mask[:, :seq_len].repeat(batch_size, 1).to(device)
    return input_ids, attention_mask


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
def run_case(model, tokenizer, batch_size: int, prompt_len: int, decode_len: int, repeat: int, warmup_decode: int, device: torch.device) -> dict[str, float | int]:
    input_ids, attention_mask = make_inputs(tokenizer, batch_size, prompt_len, device)

    warmup = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    warmup_token = warmup.logits[:, -1:].argmax(dim=-1)
    decode_with_cache(model, warmup_token, attention_mask, warmup.past_key_values, min(warmup_decode, decode_len))
    synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    prefill_times = []
    decode_times = []
    for _ in range(repeat):
        prefill_ms, outputs = timed_ms(lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True))
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        decode_ms, _ = timed_ms(lambda: decode_with_cache(model, next_token, attention_mask, outputs.past_key_values, decode_len))
        prefill_times.append(prefill_ms)
        decode_times.append(decode_ms)

    prefill_avg = statistics.mean(prefill_times)
    decode_avg = statistics.mean(decode_times)
    tpot = decode_avg / decode_len
    decode_tokens_per_s = batch_size * decode_len / (decode_avg / 1000.0)
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "prefill_ms": prefill_avg,
        "decode_ms": decode_avg,
        "ttft_est_ms": prefill_avg + tpot,
        "tpot_ms": tpot,
        "decode_tokens_per_s": decode_tokens_per_s,
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
                row = run_case(model, tokenizer, batch_size, prompt_len, decode_len, args.repeat, args.warmup_decode, device)
                row["dtype"] = args.dtype
                row["attn_implementation"] = args.attn_implementation
                rows.append(row)
                print(
                    f"b={batch_size} p={prompt_len} d={decode_len} "
                    f"prefill={row['prefill_ms']:.1f}ms decode={row['decode_ms']:.1f}ms "
                    f"ttft={row['ttft_est_ms']:.1f}ms tpot={row['tpot_ms']:.3f}ms "
                    f"tok/s={row['decode_tokens_per_s']:.1f} mem={row['max_memory_mib']:.1f}MiB",
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
        "prefill_ms",
        "decode_ms",
        "ttft_est_ms",
        "tpot_ms",
        "decode_tokens_per_s",
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
