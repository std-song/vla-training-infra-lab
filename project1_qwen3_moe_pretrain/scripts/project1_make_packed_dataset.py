#!/usr/bin/env python3
"""Tokenize and pack a mini corpus into fixed-length NPZ shards.

This is a lightweight analogue of scripts such as make_packed_dataset.py in
Qwen-style pretraining repos. It prefers Hugging Face tokenizers when available
and falls back to a deterministic byte tokenizer for offline validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised in minimal local envs
    np = None


class ByteTokenizer:
    name_or_path = "byte-fallback"
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        # Reserve 0 for padding, 1 for BOS, 2 for EOS. Bytes are offset by 3.
        return [1] + [b + 3 for b in text.encode("utf-8")] + [2]


def load_tokenizer(name_or_path: str, fallback: str):
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True, local_files_only=True)
        return tokenizer, "hf-local"
    except Exception as exc:
        if fallback != "byte":
            raise
        print(f"[warn] falling back to byte tokenizer: {exc}")
        return ByteTokenizer(), "byte"


def encode_text(tokenizer, text: str) -> list[int]:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text, add_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/project1/processed/manifest.json")
    parser.add_argument("--input", default="data/project1/processed/mini_mixed_corpus.jsonl")
    parser.add_argument("--out-dir", default="data/project1/packed/mini_s128")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--fallback-tokenizer", default="byte", choices=["byte"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shard-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32", "int64"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_mode = load_tokenizer(args.tokenizer, args.fallback_tokenizer)

    token_stream: list[int] = []
    source_counts: dict[str, int] = {}
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids = encode_text(tokenizer, obj["text"])
            token_stream.extend(ids)
            source_counts[obj["source"]] = source_counts.get(obj["source"], 0) + len(ids)

    usable = (len(token_stream) // args.seq_len) * args.seq_len
    token_stream = token_stream[:usable]
    if args.dtype == "uint16":
        item_size = 2
    elif args.dtype == "uint32":
        item_size = 4
    else:
        item_size = 8

    shard_paths = []
    for shard_idx, start in enumerate(range(0, len(token_stream), args.shard_tokens)):
        shard = token_stream[start : start + args.shard_tokens]
        if not shard:
            continue
        if np is not None:
            arr = np.asarray(shard, dtype=np.dtype(args.dtype))
            shard_path = out_dir / f"shard_{shard_idx:05d}.npz"
            np.savez_compressed(shard_path, input_ids=arr, seq_len=np.asarray([args.seq_len], dtype=np.int32))
            shard_format = "npz"
        else:
            shard_path = out_dir / f"shard_{shard_idx:05d}.{args.dtype}.bin"
            max_value = (1 << (8 * item_size)) - 1
            if any(token < 0 or token > max_value for token in shard):
                raise ValueError(f"token id exceeds {args.dtype} range; choose a wider dtype")
            with shard_path.open("wb") as f:
                for token in shard:
                    f.write(int(token).to_bytes(item_size, byteorder="little", signed=args.dtype == "int64"))
            shard_format = "bin"
        shard_paths.append(str(shard_path))

    report = {
        "input": args.input,
        "manifest": args.manifest,
        "tokenizer": getattr(tokenizer, "name_or_path", args.tokenizer),
        "tokenizer_mode": tokenizer_mode,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "total_tokens_raw": sum(source_counts.values()),
        "total_tokens_packed": int(len(token_stream)),
        "num_sequences": int(len(token_stream) // args.seq_len),
        "num_shards": len(shard_paths),
        "shard_format": shard_format if shard_paths else "",
        "source_token_counts": source_counts,
        "shards": shard_paths,
    }
    (out_dir / "packed_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
