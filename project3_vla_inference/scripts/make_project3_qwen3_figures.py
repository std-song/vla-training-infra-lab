from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "project3_vla_infer" / "results"
FIGURES = ROOT / "assets" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bar_svg(title: str, labels: list[str], values: list[float], ylabel: str, out: Path) -> None:
    width, height = 760, 420
    left, right, top, bottom = 76, 28, 60, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(values) * 1.18 if values else 1.0
    bar_w = plot_w / max(len(values), 1) * 0.58
    gap = plot_w / max(len(values), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]

    for tick in range(5):
        value = max_v * tick / 4
        y = top + plot_h - (value / max_v) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.0f}</text>')

    palette = ["#2f6f73", "#c55f36", "#455a8a", "#8b6f2f", "#5c7f3b", "#7a4f86"]
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = left + gap * idx + gap / 2
        h = (value / max_v) * plot_h
        x = cx - bar_w / 2
        y = top + plot_h - h
        color = palette[idx % len(palette)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{value:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')

    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def line_svg(title: str, series: dict[str, list[tuple[int, float]]], ylabel: str, out: Path) -> None:
    width, height = 780, 420
    left, right, top, bottom = 76, 128, 58, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({x for points in series.values() for x, _ in points})
    max_x = max(xs) if xs else 1
    min_x = min(xs) if xs else 0
    max_y = max(y for points in series.values() for _, y in points) * 1.15
    min_y = 0.0

    def px(x: int) -> float:
        return left + (x - min_x) / max(max_x - min_x, 1) * plot_w

    def py(y: float) -> float:
        return top + plot_h - (y - min_y) / max(max_y - min_y, 1) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="13">Prompt length</text>',
    ]

    for x in xs:
        parts.append(f'<text x="{px(x):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x}</text>')
    for tick in range(5):
        value = max_y * tick / 4
        y = py(value)
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.0f}</text>')

    palette = ["#2f6f73", "#c55f36", "#455a8a", "#8b6f2f"]
    for idx, (name, points) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        coords = " ".join(f'{px(x):.1f},{py(y):.1f}' for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="{color}"/>')
        ly = top + 16 + idx * 24
        parts.append(f'<rect x="{left + plot_w + 28}" y="{ly - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{left + plot_w + 50}" y="{ly + 2}" font-family="Arial" font-size="12">{name}</text>')

    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    baseline = read_csv(RESULTS / "qwen3_prefill_decode_sdpa_bf16.csv")
    throughput_rows = [
        row
        for row in baseline
        if int(row["decode_len"]) == 128 and int(row["batch_size"]) in {1, 2, 4}
    ]
    throughput_rows.sort(key=lambda r: (int(r["prompt_len"]), int(r["batch_size"])))
    labels = [f'B{r["batch_size"]}\\nP{r["prompt_len"]}' for r in throughput_rows]
    values = [float(r["decode_tokens_per_s"]) for r in throughput_rows]
    bar_svg(
        "Qwen3-0.6B SDPA Decode Throughput",
        labels,
        values,
        "decode tokens/s",
        FIGURES / "project3_qwen3_decode_throughput.svg",
    )

    mem_series: dict[str, list[tuple[int, float]]] = {}
    for batch in (1, 2, 4):
        points = [
            (int(row["prompt_len"]), float(row["max_memory_mib"]))
            for row in baseline
            if int(row["batch_size"]) == batch and int(row["decode_len"]) == 128
        ]
        mem_series[f"B{batch}"] = sorted(points)
    line_svg(
        "Qwen3-0.6B KV Cache Memory by Prompt",
        mem_series,
        "max memory MiB",
        FIGURES / "project3_qwen3_prefill_memory.svg",
    )

    kv = read_csv(RESULTS / "qwen3_kv_cache_compare_sdpa_bf16.csv")
    kv_rows = [
        row
        for row in kv
        if int(row["decode_len"]) == 64 and int(row["prompt_len"]) in {128, 512}
    ]
    kv_rows.sort(key=lambda r: (int(r["batch_size"]), int(r["prompt_len"])))
    labels = [f'B{r["batch_size"]}\\nP{r["prompt_len"]}' for r in kv_rows]
    values = [float(r["speedup"]) for r in kv_rows]
    bar_svg(
        "Qwen3 KV Cache Speedup vs Full Recompute",
        labels,
        values,
        "speedup",
        FIGURES / "project3_qwen3_kv_cache_speedup.svg",
    )

    selected_shape = {"batch_size": "4", "prompt_len": "1024", "decode_len": "128"}
    backend_files = {
        "SDPA": "qwen3_prefill_decode_sdpa_bf16.csv",
        "eager": "qwen3_prefill_decode_eager_bf16_selected.csv",
        "FA2": "qwen3_prefill_decode_flashattn2_bf16_selected.csv",
    }
    labels, values = [], []
    for label, file_name in backend_files.items():
        rows = read_csv(RESULTS / file_name)
        match = next(
            row
            for row in rows
            if all(row[key] == value for key, value in selected_shape.items())
        )
        labels.append(label)
        values.append(float(match["tpot_ms"]))
    bar_svg(
        "Qwen3 Attention Backend TPOT, B4 P1024 D128",
        labels,
        values,
        "TPOT ms/token",
        FIGURES / "project3_qwen3_attention_backends.svg",
    )

    triton = read_csv(RESULTS / "qwen3_vla_action_triton_hidden1024.csv")
    triton_rows = [
        row
        for row in triton
        if row["action_dim"] == "14" and row["horizon"] in {"10", "32", "64"}
    ]
    triton_rows.sort(key=lambda r: (int(r["batch_size"]), int(r["horizon"])))
    labels = [f'B{r["batch_size"]}\\nH{r["horizon"]}' for r in triton_rows]
    values = [float(r["triton_speedup"]) for r in triton_rows]
    bar_svg(
        "Qwen3 Hidden-1024 VLA Action Postprocess Speedup",
        labels,
        values,
        "Triton / PyTorch speedup",
        FIGURES / "project3_qwen3_vla_action_triton.svg",
    )


if __name__ == "__main__":
    main()

