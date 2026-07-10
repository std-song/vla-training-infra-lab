from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


@dataclass
class Request:
    prompt: str
    image: Image.Image


def make_image(size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    pad = size // 4
    draw.rectangle((pad, pad, size - pad, size - pad), outline=(30, 30, 30), width=max(2, size // 80))
    draw.text((pad + 8, size // 2 - 12), label, fill=(0, 0, 0))
    return image


def build_request(processor, image: Image.Image, text: str) -> Request:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return Request(prompt=prompt, image=image)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_batch(llm: LLM, sampling: SamplingParams, requests: list[Request]) -> tuple[float, int]:
    inputs = [{"prompt": req.prompt, "multi_modal_data": {"image": req.image}} for req in requests]
    cuda_sync()
    start = time.perf_counter()
    outputs = llm.generate(inputs, sampling, use_tqdm=False)
    cuda_sync()
    elapsed = time.perf_counter() - start
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return elapsed, output_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL multimodal serving through vLLM.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", default="project3_vla_infer/results/qwen3vl_vllm_concurrency.csv")
    parser.add_argument("--image-sizes", default="224,448")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    rows: list[dict[str, float | int]] = []
    image_sizes = [int(value) for value in args.image_sizes.split(",") if value]
    concurrencies = [int(value) for value in args.concurrency.split(",") if value]

    for image_size in image_sizes:
        for concurrency in concurrencies:
            requests = [
                build_request(
                    processor,
                    make_image(image_size, f"cabinet-{idx}"),
                    "Describe the image briefly and identify the object.",
                )
                for idx in range(concurrency)
            ]

            for _ in range(args.warmup):
                run_batch(llm, sampling, requests)

            times: list[float] = []
            tokens: list[int] = []
            for _ in range(args.repeat):
                elapsed, output_tokens = run_batch(llm, sampling, requests)
                times.append(elapsed)
                tokens.append(output_tokens)

            avg_latency_s = sum(times) / len(times)
            avg_output_tokens = sum(tokens) / len(tokens)
            row = {
                "image_size": image_size,
                "concurrency": concurrency,
                "max_new_tokens": args.max_new_tokens,
                "avg_latency_s": avg_latency_s,
                "req_per_s": concurrency / avg_latency_s,
                "output_tokens_per_s": avg_output_tokens / avg_latency_s,
                "avg_output_tokens": avg_output_tokens,
                # vLLM executes in worker processes, so parent-process torch memory stats are not reliable.
                # Use nvidia-smi sampling for peak memory and record it separately.
                "max_mem_mib": 0.0,
            }
            print(row)
            rows.append(row)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
