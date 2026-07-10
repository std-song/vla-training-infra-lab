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
    width, height = 900, 440
    left, right, top, bottom = 88, 30, 62, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(values) * 1.18 if values else 1.0
    slot = plot_w / max(len(values), 1)
    bar_w = slot * 0.56
    colors = ["#2f6f73", "#c55f36", "#455a8a", "#8b6f2f", "#5c7f3b", "#7a4f86"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for tick in range(5):
        value = max_v * tick / 4
        y = top + plot_h - value / max_v * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = left + slot * idx + slot / 2
        h = value / max_v * plot_h
        x = cx - bar_w / 2
        y = top + plot_h - h
        color = colors[idx % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700">{value:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 23}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def grouped_bar_svg(title: str, groups: list[str], series: list[tuple[str, list[float], str]], ylabel: str, out: Path) -> None:
    width, height = 900, 460
    left, right, top, bottom = 88, 130, 62, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(max(values) for _, values, _ in series) * 1.18
    group_w = plot_w / len(groups)
    bar_w = group_w / (len(series) + 1.4)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for tick in range(5):
        value = max_v * tick / 4
        y = top + plot_h - value / max_v * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.1f}</text>')
    for group_idx, group in enumerate(groups):
        base_x = left + group_idx * group_w + group_w / 2
        parts.append(f'<text x="{base_x:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="11">{group}</text>')
        for series_idx, (_, values, color) in enumerate(series):
            value = values[group_idx]
            x = base_x - (len(series) * bar_w) / 2 + series_idx * bar_w
            h = value / max_v * plot_h
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.84:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
    legend_x = left + plot_w + 24
    for idx, (name, _, color) in enumerate(series):
        y = top + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{y}" font-family="Arial" font-size="12">{name}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    eager = read_csv(RESULTS / "qwen3vl_vllm_eager_concurrency.csv")
    default = read_csv(RESULTS / "qwen3vl_vllm_default_concurrency.csv")

    default_rows = sorted(default, key=lambda row: (int(row["image_size"]), int(row["concurrency"])))
    labels = [f'{row["image_size"]}p\nc{row["concurrency"]}' for row in default_rows]
    bar_svg(
        "Qwen3-VL-4B vLLM Default Throughput",
        labels,
        [float(row["req_per_s"]) for row in default_rows],
        "requests/s",
        FIGURES / "project3_qwen3vl_vllm_throughput.svg",
    )
    bar_svg(
        "Qwen3-VL-4B vLLM Default Batch Latency",
        labels,
        [float(row["avg_latency_s"]) for row in default_rows],
        "seconds",
        FIGURES / "project3_qwen3vl_vllm_latency.svg",
    )

    groups = ["224 c1", "224 c2", "224 c4", "224 c8", "448 c1", "448 c2", "448 c4", "448 c8"]
    eager_map = {(int(row["image_size"]), int(row["concurrency"])): float(row["req_per_s"]) for row in eager}
    default_map = {(int(row["image_size"]), int(row["concurrency"])): float(row["req_per_s"]) for row in default}
    keys = [(224, 1), (224, 2), (224, 4), (224, 8), (448, 1), (448, 2), (448, 4), (448, 8)]
    grouped_bar_svg(
        "Qwen3-VL-4B vLLM Eager vs Default",
        groups,
        [
            ("eager", [eager_map[key] for key in keys], "#8b6f2f"),
            ("default", [default_map[key] for key in keys], "#2f6f73"),
        ],
        "requests/s",
        FIGURES / "project3_qwen3vl_vllm_eager_vs_default.svg",
    )


if __name__ == "__main__":
    main()
