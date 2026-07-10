#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-examples/smoke/config_qwen2_moe_baseline_100step.yaml}
RUN_ID=${2:-qwen2_moe_baseline_100step}
LOG_DIR=${LOG_DIR:-profiling_logs}

mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/${RUN_ID}_train.log"
SMI_LOG="$LOG_DIR/${RUN_ID}_nvidia_smi.csv"
SUMMARY="$LOG_DIR/${RUN_ID}_summary.txt"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 1 > "$SMI_LOG" &
SMI_PID=$!
trap 'kill "$SMI_PID" >/dev/null 2>&1 || true' EXIT

CUDA_DEVICE_MAX_CONNECTIONS=1 \
ENABLE_TIMERS=1 \
DEBUG_CPU=1 \
STATS_SAMPLING_INTERVAL_IN_SEC=1 \
PYTHONPATH=src \
torchrun --nproc_per_node=1 run_train.py --config-file "$CONFIG" 2>&1 | tee "$TRAIN_LOG"

kill "$SMI_PID" >/dev/null 2>&1 || true
trap - EXIT

python - "$TRAIN_LOG" "$SMI_LOG" "$SUMMARY" <<'PY'
from pathlib import Path
import csv
import re
import statistics
import sys

train_log = Path(sys.argv[1])
smi_log = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

step_re = re.compile(r"iteration:\s*(\d+)\s*/\s*(\d+).*?time_per_iteration_ms:\s*([0-9.]+).*?tokens_per_sec:\s*([0-9.]+[KMG]?)")

def parse_num(value: str) -> float:
    value = value.strip()
    multiplier = 1.0
    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("G"):
        multiplier = 1_000_000_000.0
        value = value[:-1]
    return float(value) * multiplier

steps = []
for line in train_log.read_text(errors="replace").splitlines():
    match = step_re.search(line)
    if match:
        steps.append({
            "step": int(match.group(1)),
            "total": int(match.group(2)),
            "ms": float(match.group(3)),
            "tps": parse_num(match.group(4)),
            "raw": line,
        })

steady = [item for item in steps if item["step"] >= 10]
max_mem = max_util = max_power = None
if smi_log.exists():
    with smi_log.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        mem_values, util_values, power_values = [], [], []
        for row in reader:
            try:
                mem_values.append(float(row[" memory.used [MiB]"].split()[0]))
                util_values.append(float(row[" utilization.gpu [%]"].split()[0]))
                power_values.append(float(row[" power.draw [W]"].split()[0]))
            except Exception:
                pass
        if mem_values:
            max_mem = max(mem_values)
        if util_values:
            max_util = max(util_values)
        if power_values:
            max_power = max(power_values)

lines = ["# Qwen2-MoE 1xRTX3090 baseline summary", ""]
lines.append(f"- parsed_steps: {len(steps)}")
if steady:
    lines.append(f"- steady_steps: {steady[0]['step']}..{steady[-1]['step']}")
    lines.append(f"- avg_time_per_iteration_ms_step_ge_10: {statistics.mean(x['ms'] for x in steady):.2f}")
    lines.append(f"- avg_tokens_per_sec_step_ge_10: {statistics.mean(x['tps'] for x in steady):.0f}")
    lines.append(f"- max_tokens_per_sec_step_ge_10: {max(x['tps'] for x in steady):.0f}")
    lines.append(f"- min_tokens_per_sec_step_ge_10: {min(x['tps'] for x in steady):.0f}")
if steps:
    lines.append(f"- final_step: {steps[-1]['step']}/{steps[-1]['total']}")
    lines.append(f"- final_time_per_iteration_ms: {steps[-1]['ms']:.2f}")
    lines.append(f"- final_tokens_per_sec: {steps[-1]['tps']:.0f}")
if max_mem is not None:
    lines.append(f"- max_sampled_gpu_memory_mib: {max_mem:.0f}")
if max_util is not None:
    lines.append(f"- max_sampled_gpu_util_percent: {max_util:.0f}")
if max_power is not None:
    lines.append(f"- max_sampled_power_w: {max_power:.2f}")
if steps:
    lines.extend(["", "## Last train metrics", "```text", steps[-1]["raw"], "```"])
summary_path.write_text("\n".join(lines) + "\n")
print(summary_path.read_text())
PY
