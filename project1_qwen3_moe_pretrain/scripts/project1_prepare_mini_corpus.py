#!/usr/bin/env python3
"""Prepare a small mixed corpus manifest for Project 1.

The script follows the shape of larger Qwen-style pretraining pipelines:
multiple raw corpora are normalized into jsonl shards with a manifest that
records document counts, byte counts, and mixture weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_simple_yaml(path: Path) -> dict:
    """Parse the narrow YAML subset used by configs/data/mini_qwen3_pretrain_mix.yaml."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        pass

    # Dependency-free fallback for this config's simple structure.
    data: dict = {"corpora": []}
    current: dict | None = None
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - "):
            current = {}
            data.setdefault("corpora", []).append(current)
            key, value = line[4:].split(":", 1)
            current[key.strip()] = coerce_scalar(value.strip())
            section = "corpora"
        elif raw.startswith("    ") and current is not None and section == "corpora":
            key, value = line.strip().split(":", 1)
            current[key.strip()] = coerce_scalar(value.strip())
        elif not raw.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            if value.strip():
                data[key] = coerce_scalar(value.strip())
                section = None
            else:
                data[key] = {}
                section = key
        elif raw.startswith("  ") and section and isinstance(data.get(section), dict):
            key, value = line.strip().split(":", 1)
            data[section][key.strip()] = coerce_scalar(value.strip())
    return data


def coerce_scalar(value: str):
    if value in {"true", "false"}:
        return value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def iter_jsonl_text(path: Path, text_key: str):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj[text_key]).strip()
            if text:
                yield line_no, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/mini_qwen3_pretrain_mix.yaml")
    parser.add_argument("--out-dir", default="data/project1/processed")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = parse_simple_yaml(config_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": str(config_path),
        "seed": config.get("seed", 42),
        "corpora": [],
        "total_docs": 0,
        "total_bytes": 0,
    }

    mixed_path = out_dir / "mini_mixed_corpus.jsonl"
    with mixed_path.open("w", encoding="utf-8") as out:
        for corpus in config["corpora"]:
            src = Path(corpus["path"])
            stats = {
                "name": corpus["name"],
                "path": str(src),
                "weight": float(corpus["weight"]),
                "docs": 0,
                "bytes": 0,
            }
            for line_no, text in iter_jsonl_text(src, corpus.get("text_key", "text")):
                record = {"source": corpus["name"], "line_no": line_no, "text": text}
                encoded = json.dumps(record, ensure_ascii=False)
                out.write(encoded + "\n")
                stats["docs"] += 1
                stats["bytes"] += len(text.encode("utf-8"))
            manifest["corpora"].append(stats)
            manifest["total_docs"] += stats["docs"]
            manifest["total_bytes"] += stats["bytes"]

    manifest["mixed_jsonl"] = str(mixed_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
