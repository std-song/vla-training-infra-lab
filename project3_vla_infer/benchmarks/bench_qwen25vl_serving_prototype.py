from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight vLLM-style serving prototype for Qwen2.5-VL VLA requests.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="project3_vla_infer/results/qwen25vl_serving_prototype.csv")
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-count", type=int, default=3)
    parser.add_argument("--decode-len", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype in {torch.bfloat16, torch.float16}:
        return 2
    return 4


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_ms(fn):
    synchronize()
    start = time.perf_counter()
    output = fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0, output


def make_image(size: int, index: int, request_id: int) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(24 + index * 28, 40 + request_id % 40, 58))
    draw = ImageDraw.Draw(image)
    pad = max(size // 12, 8)
    draw.rectangle([pad, pad, size - pad, size - pad], outline=(220, 220, 190), width=max(size // 80, 2))
    draw.line([pad, size // 2, size - pad, size // 2], fill=(180, 100 + index * 24, 72), width=max(size // 90, 2))
    draw.text((pad, pad), f"cam{index}/req{request_id}", fill=(240, 240, 220))
    return image


def build_one_inputs(processor, image_size: int, image_count: int, request_id: int, device: torch.device):
    images = [make_image(image_size, idx, request_id) for idx in range(image_count)]
    content = [{"type": "image", "image": image} for image in images]
    content.append(
        {
            "type": "text",
            "text": "You are a robot policy. Given these camera views, predict the next action for opening the cabinet.",
        }
    )
    messages = [{"role": "user", "content": content}]
    preprocess_ms, inputs = timed_ms(
        lambda: processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )
    transfer_ms, inputs = timed_ms(lambda: inputs.to(device))
    return inputs, preprocess_ms, transfer_ms


def batch_inputs(samples: list[Any]) -> dict[str, torch.Tensor]:
    keys = samples[0].keys()
    batched: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if key in {"input_ids", "attention_mask"}:
            batched[key] = torch.cat(values, dim=0)
        elif key in {"pixel_values", "image_grid_thw"}:
            batched[key] = torch.cat(values, dim=0)
        else:
            # Keep the common path extensible while failing loudly for unsupported nested values.
            if torch.is_tensor(values[0]):
                batched[key] = torch.cat(values, dim=0)
            else:
                raise TypeError(f"Unsupported batched input key: {key}")
    return batched


def count_visual_tokens(input_ids: torch.Tensor, processor) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return -1
    ids = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"])
    ids = [item for item in ids if item is not None and item != tokenizer.unk_token_id]
    if not ids:
        return -1
    return int(sum((input_ids == item).sum().item() for item in ids))


def kv_cache_mib(model, batch_size: int, tokens: int, dtype: torch.dtype) -> float:
    cfg = getattr(model.config, "text_config", model.config)
    layers = int(getattr(cfg, "num_hidden_layers"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads")))
    hidden = int(getattr(cfg, "hidden_size"))
    heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = hidden // heads
    bytes_total = batch_size * tokens * layers * kv_heads * head_dim * 2 * dtype_bytes(dtype)
    return bytes_total / 1024**2


@torch.inference_mode()
def generate_ms(model, inputs: dict[str, torch.Tensor], decode_len: int) -> float:
    elapsed, _ = timed_ms(lambda: model.generate(**inputs, max_new_tokens=decode_len, do_sample=False))
    return elapsed


@dataclass
class ScenarioResult:
    scenario: str
    request_count: int
    image_count: int
    image_size: int
    input_tokens: int
    visual_marker_tokens: int
    decode_len: int
    total_ms: float
    per_request_ms: float
    requests_per_s: float
    preprocess_ms_total: float
    transfer_ms_total: float
    generate_ms_total: float
    max_memory_mib: float
    estimated_kv_cache_mib: float


@torch.inference_mode()
def run_once(model, processor, args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> list[ScenarioResult]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cold_total_start = time.perf_counter()
    cold_preprocess = 0.0
    cold_transfer = 0.0
    cold_generate = 0.0
    first_inputs = None
    for request_id in range(args.request_count):
        inputs, preprocess_ms, transfer_ms = build_one_inputs(processor, args.image_size, args.image_count, request_id, device)
        gen_ms = generate_ms(model, inputs, args.decode_len)
        cold_preprocess += preprocess_ms
        cold_transfer += transfer_ms
        cold_generate += gen_ms
        if first_inputs is None:
            first_inputs = inputs
    synchronize()
    cold_total = (time.perf_counter() - cold_total_start) * 1000.0
    cold_memory = torch.cuda.max_memory_allocated() / 1024**2

    assert first_inputs is not None
    input_tokens = int(first_inputs["input_ids"].shape[1])
    visual_tokens = count_visual_tokens(first_inputs["input_ids"], processor)
    kv_mib = kv_cache_mib(model, args.request_count, input_tokens + args.decode_len, dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache_inputs = []
    cache_preprocess = 0.0
    cache_transfer = 0.0
    for request_id in range(args.request_count):
        inputs, preprocess_ms, transfer_ms = build_one_inputs(processor, args.image_size, args.image_count, request_id, device)
        cache_inputs.append(inputs)
        cache_preprocess += preprocess_ms
        cache_transfer += transfer_ms

    cached_generate_start = time.perf_counter()
    cached_generate = 0.0
    for inputs in cache_inputs:
        cached_generate += generate_ms(model, inputs, args.decode_len)
    synchronize()
    cached_total = (time.perf_counter() - cached_generate_start) * 1000.0
    cached_memory = torch.cuda.max_memory_allocated() / 1024**2

    batched = batch_inputs(cache_inputs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    microbatch_generate = generate_ms(model, batched, args.decode_len)
    microbatch_memory = torch.cuda.max_memory_allocated() / 1024**2

    return [
        ScenarioResult(
            scenario="serial_cold",
            request_count=args.request_count,
            image_count=args.image_count,
            image_size=args.image_size,
            input_tokens=input_tokens,
            visual_marker_tokens=visual_tokens,
            decode_len=args.decode_len,
            total_ms=cold_total,
            per_request_ms=cold_total / args.request_count,
            requests_per_s=args.request_count / (cold_total / 1000.0),
            preprocess_ms_total=cold_preprocess,
            transfer_ms_total=cold_transfer,
            generate_ms_total=cold_generate,
            max_memory_mib=cold_memory,
            estimated_kv_cache_mib=kv_mib,
        ),
        ScenarioResult(
            scenario="visual_input_cache_serial",
            request_count=args.request_count,
            image_count=args.image_count,
            image_size=args.image_size,
            input_tokens=input_tokens,
            visual_marker_tokens=visual_tokens,
            decode_len=args.decode_len,
            total_ms=cached_total,
            per_request_ms=cached_total / args.request_count,
            requests_per_s=args.request_count / (cached_total / 1000.0),
            preprocess_ms_total=0.0,
            transfer_ms_total=0.0,
            generate_ms_total=cached_generate,
            max_memory_mib=cached_memory,
            estimated_kv_cache_mib=kv_mib,
        ),
        ScenarioResult(
            scenario="visual_input_cache_microbatch",
            request_count=args.request_count,
            image_count=args.image_count,
            image_size=args.image_size,
            input_tokens=input_tokens,
            visual_marker_tokens=visual_tokens,
            decode_len=args.decode_len,
            total_ms=microbatch_generate,
            per_request_ms=microbatch_generate / args.request_count,
            requests_per_s=args.request_count / (microbatch_generate / 1000.0),
            preprocess_ms_total=0.0,
            transfer_ms_total=0.0,
            generate_ms_total=microbatch_generate,
            max_memory_mib=microbatch_memory,
            estimated_kv_cache_mib=kv_mib,
        ),
    ]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()

    all_rows: list[ScenarioResult] = []
    for idx in range(args.repeat):
        rows = run_once(model, processor, args, dtype, device)
        all_rows.extend(rows)
        print(f"repeat={idx + 1}/{args.repeat}")
        baseline = rows[0].per_request_ms
        for row in rows:
            print(
                f"{row.scenario}: total={row.total_ms:.1f}ms per_req={row.per_request_ms:.1f}ms "
                f"rps={row.requests_per_s:.2f} mem={row.max_memory_mib:.1f}MiB "
                f"speedup_vs_cold={baseline / row.per_request_ms:.2f}x",
                flush=True,
            )

    grouped: dict[str, list[ScenarioResult]] = {}
    for row in all_rows:
        grouped.setdefault(row.scenario, []).append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario",
        "request_count",
        "image_count",
        "image_size",
        "input_tokens",
        "visual_marker_tokens",
        "decode_len",
        "total_ms",
        "per_request_ms",
        "requests_per_s",
        "speedup_vs_serial_cold",
        "preprocess_ms_total",
        "transfer_ms_total",
        "generate_ms_total",
        "max_memory_mib",
        "estimated_kv_cache_mib",
        "dtype",
        "attn_implementation",
        "min_pixels",
        "max_pixels",
    ]
    baseline = statistics.mean(row.per_request_ms for row in grouped["serial_cold"])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for scenario, rows in grouped.items():
            sample = rows[0]
            per_request = statistics.mean(row.per_request_ms for row in rows)
            row = {
                "scenario": scenario,
                "request_count": sample.request_count,
                "image_count": sample.image_count,
                "image_size": sample.image_size,
                "input_tokens": sample.input_tokens,
                "visual_marker_tokens": sample.visual_marker_tokens,
                "decode_len": sample.decode_len,
                "total_ms": statistics.mean(item.total_ms for item in rows),
                "per_request_ms": per_request,
                "requests_per_s": statistics.mean(item.requests_per_s for item in rows),
                "speedup_vs_serial_cold": baseline / per_request,
                "preprocess_ms_total": statistics.mean(item.preprocess_ms_total for item in rows),
                "transfer_ms_total": statistics.mean(item.transfer_ms_total for item in rows),
                "generate_ms_total": statistics.mean(item.generate_ms_total for item in rows),
                "max_memory_mib": statistics.mean(item.max_memory_mib for item in rows),
                "estimated_kv_cache_mib": sample.estimated_kv_cache_mib,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
            writer.writerow(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
