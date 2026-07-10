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
    width, height = 820, 430
    left, right, top, bottom = 82, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(values) * 1.18 if values else 1.0
    slot = plot_w / max(len(values), 1)
    bar_w = slot * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
    ]
    for tick in range(5):
        value = max_v * tick / 4
        y = top + plot_h - value / max_v * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.0f}</text>')
    palette = ["#2f6f73", "#c55f36", "#455a8a", "#8b6f2f", "#5c7f3b", "#7a4f86", "#4d6b5f", "#9a5b47"]
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = left + slot * i + slot / 2
        h = value / max_v * plot_h
        x = cx - bar_w / 2
        y = top + plot_h - h
        color = palette[i % len(palette)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{value:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RESULTS / "qwen25vl_visual_tokens_dynamic_pixels_sdpa_bf16.csv")
    rows = [row for row in rows if row["decode_len"] == "64"]
    rows.sort(key=lambda r: (int(r["image_count"]), int(r["image_size"])))
    labels = [f'{r["image_count"]}img\\n{r["image_size"]}p' for r in rows]

    bar_svg(
        "Qwen2.5-VL Visual Marker Tokens",
        labels,
        [float(r["visual_marker_tokens"]) for r in rows],
        "visual marker tokens",
        FIGURES / "project3_qwen25vl_visual_tokens.svg",
    )
    bar_svg(
        "Qwen2.5-VL Multimodal Prefill Latency",
        labels,
        [float(r["prefill_ms"]) for r in rows],
        "prefill ms",
        FIGURES / "project3_qwen25vl_prefill_latency.svg",
    )
    bar_svg(
        "Qwen2.5-VL Estimated TTFT",
        labels,
        [float(r["ttft_est_ms"]) for r in rows],
        "TTFT ms",
        FIGURES / "project3_qwen25vl_ttft.svg",
    )
    bar_svg(
        "Qwen2.5-VL GPU Memory",
        labels,
        [float(r["max_memory_mib"]) for r in rows],
        "max memory MiB",
        FIGURES / "project3_qwen25vl_memory.svg",
    )


if __name__ == "__main__":
    main()
