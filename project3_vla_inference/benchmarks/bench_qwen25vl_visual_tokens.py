from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-VL visual-token prefill/decode latency.")
    parser.add_argument("--model-dir", required=True, help="Local Hugging Face or ModelScope model directory.")
    parser.add_argument("--out", default="project3_vla_infer/results/qwen25vl_visual_tokens_bf16.csv")
    parser.add_argument("--image-sizes", default="224,448")
    parser.add_argument("--image-counts", default="1,3")
    parser.add_argument("--decode-lengths", default="16,64")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup-decode", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
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


def make_image(size: int, index: int) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(24 + index * 30, 38, 52))
    draw = ImageDraw.Draw(image)
    pad = max(size // 12, 8)
    draw.rectangle([pad, pad, size - pad, size - pad], outline=(220, 220, 190), width=max(size // 80, 2))
    draw.line([pad, size // 2, size - pad, size // 2], fill=(180, 96 + index * 24, 72), width=max(size // 90, 2))
    draw.text((pad, pad), f"cam{index}", fill=(240, 240, 220))
    return image


def build_inputs(processor, image_size: int, image_count: int, device: torch.device):
    images = [make_image(image_size, idx) for idx in range(image_count)]
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
    # The processor may return BatchEncoding tensors on CPU. Move tensors only after measuring preprocessing.
    transfer_ms, inputs = timed_ms(lambda: inputs.to(device))
    return inputs, preprocess_ms, transfer_ms


def count_visual_tokens(input_ids: torch.Tensor, processor) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return -1
    ids = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"])
    ids = [item for item in ids if item is not None and item != tokenizer.unk_token_id]
    if not ids:
        return -1
    return int(sum((input_ids == item).sum().item() for item in ids))


@torch.inference_mode()
def run_case(model, processor, image_size: int, image_count: int, decode_len: int, repeat: int, warmup_decode: int, device: torch.device) -> dict[str, float | int]:
    inputs, preprocess_ms, transfer_ms = build_inputs(processor, image_size, image_count, device)

    _ = model(**inputs, use_cache=True)
    _ = model.generate(**inputs, max_new_tokens=min(warmup_decode, decode_len), do_sample=False)
    synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    prefill_times = []
    decode_times = []
    preprocess_times = []
    transfer_times = []
    for _ in range(repeat):
        iter_inputs, iter_preprocess_ms, iter_transfer_ms = build_inputs(processor, image_size, image_count, device)
        prefill_ms, _ = timed_ms(lambda: model(**iter_inputs, use_cache=True))
        generate_ms, _ = timed_ms(lambda: model.generate(**iter_inputs, max_new_tokens=decode_len, do_sample=False))
        preprocess_times.append(iter_preprocess_ms)
        transfer_times.append(iter_transfer_ms)
        prefill_times.append(prefill_ms)
        decode_times.append(max(generate_ms - prefill_ms, 0.0))

    prefill_avg = statistics.mean(prefill_times)
    decode_avg = statistics.mean(decode_times)
    preprocess_avg = statistics.mean(preprocess_times)
    transfer_avg = statistics.mean(transfer_times)
    tpot = decode_avg / decode_len
    return {
        "image_size": image_size,
        "image_count": image_count,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "visual_marker_tokens": count_visual_tokens(inputs["input_ids"], processor),
        "decode_len": decode_len,
        "preprocess_ms": preprocess_avg,
        "transfer_ms": transfer_avg,
        "prefill_ms": prefill_avg,
        "decode_ms": decode_avg,
        "ttft_est_ms": preprocess_avg + transfer_avg + prefill_avg + tpot,
        "tpot_ms": tpot,
        "decode_tokens_per_s": decode_len / (decode_avg / 1000.0),
        "max_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


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

    rows = []
    for image_size in parse_ints(args.image_sizes):
        for image_count in parse_ints(args.image_counts):
            for decode_len in parse_ints(args.decode_lengths):
                row = run_case(model, processor, image_size, image_count, decode_len, args.repeat, args.warmup_decode, device)
                row["dtype"] = args.dtype
                row["attn_implementation"] = args.attn_implementation
                row["min_pixels"] = args.min_pixels
                row["max_pixels"] = args.max_pixels
                rows.append(row)
                print(
                    f"images={image_count} size={image_size} d={decode_len} tokens={row['input_tokens']} "
                    f"preprocess={row['preprocess_ms']:.1f}ms prefill={row['prefill_ms']:.1f}ms "
                    f"decode={row['decode_ms']:.1f}ms tpot={row['tpot_ms']:.2f}ms "
                    f"ttft={row['ttft_est_ms']:.1f}ms mem={row['max_memory_mib']:.1f}MiB",
                    flush=True,
                )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dtype",
        "attn_implementation",
        "min_pixels",
        "max_pixels",
        "image_size",
        "image_count",
        "input_tokens",
        "visual_marker_tokens",
        "decode_len",
        "preprocess_ms",
        "transfer_ms",
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
