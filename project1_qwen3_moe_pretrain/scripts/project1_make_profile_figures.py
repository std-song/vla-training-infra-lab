#!/usr/bin/env python3
"""Create SVG figures from Project 1 profiling CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "single": "#2563eb",
    "dp2": "#16a34a",
    "tp2": "#7c3aed",
    "pp2": "#ea580c",
    "ep2": "#0891b2",
}


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_svg(rows, metric: str, title: str, out: Path) -> None:
    width, height = 760, 330
    left, top = 150, 78
    bar_h, gap = 24, 18
    max_value = max(float(row[metric]) for row in rows)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{title}</text>',
        '<text x="32" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Qwen3-MoE-style Nanotron, warm-step average</text>',
    ]
    for i, row in enumerate(rows):
        y = top + i * (bar_h + gap)
        job = row["job"]
        value = float(row[metric])
        bar_w = 460 * value / max_value
        color = COLORS.get(job, "#64748b")
        lines.extend(
            [
                f'<text x="34" y="{y + 17}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{job}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_w + 10:.1f}" y="{y + 17}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{value:.1f}</text>',
            ]
        )
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/project1_qwen3_moe_style_clean_summary.csv")
    parser.add_argument("--out-dir", default="assets/project1")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    out_dir = Path(args.out_dir)
    bar_svg(rows, "avg_tokens_s_warm", "Throughput by Parallel Strategy", out_dir / "qwen3_moe_throughput.svg")
    bar_svg(rows, "peak_reserved_mib", "Peak Reserved Memory by Strategy", out_dir / "qwen3_moe_memory.svg")


if __name__ == "__main__":
    main()
