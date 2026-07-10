#!/usr/bin/env python3
"""Parse Nanotron profiling logs into a compact CSV summary."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ITER_RE = re.compile(
    r"iteration:\s+(\d+)\s+/\s+(\d+).*?"
    r"time_per_iteration_ms:\s+([0-9.]+K?|[0-9.]+).*?"
    r"tokens_per_sec:\s+([0-9.]+K?|[0-9.]+).*?"
    r"tokens_per_sec_per_gpu:\s+([0-9.]+K?|[0-9.]+).*?"
    r"lm_loss:\s+([0-9.]+)"
)
MEM_RE = re.compile(r"Peak reserved:\s+([0-9.]+)MiB")


def human_float(value: str) -> float:
    return float(value[:-1]) * 1000 if value.endswith("K") else float(value)


def parse_log(path: Path, warmup_steps: int) -> dict:
    text = path.read_text(errors="ignore")
    rows = []
    for match in ITER_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group(1)),
                "total": int(match.group(2)),
                "time_ms": human_float(match.group(3)),
                "tokens_s": human_float(match.group(4)),
                "tokens_s_per_gpu": human_float(match.group(5)),
                "loss": float(match.group(6)),
            }
        )
    warm = [row for row in rows if row["step"] > warmup_steps] or rows
    mem = [float(value) for value in MEM_RE.findall(text)]
    return {
        "steps_seen": len(rows),
        "last_step": rows[-1]["step"] if rows else "",
        "avg_tokens_s_warm": mean(row["tokens_s"] for row in warm),
        "avg_tokens_s_per_gpu_warm": mean(row["tokens_s_per_gpu"] for row in warm),
        "avg_time_ms_warm": mean(row["time_ms"] for row in warm),
        "last_loss": rows[-1]["loss"] if rows else "",
        "peak_reserved_mib": max(mem) if mem else "",
        "log_failed": "Traceback" in text or "ChildFailedError" in text or "SIGSEGV" in text,
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("jobs", nargs="+", help="job=logfile.log")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "job",
        "steps_seen",
        "last_step",
        "avg_tokens_s_warm",
        "avg_tokens_s_per_gpu_warm",
        "avg_time_ms_warm",
        "last_loss",
        "peak_reserved_mib",
        "log_failed",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for spec in args.jobs:
            job, rel = spec.split("=", 1)
            writer.writerow({"job": job, **parse_log(log_dir / rel, args.warmup_steps)})
    print(out)


if __name__ == "__main__":
    main()
