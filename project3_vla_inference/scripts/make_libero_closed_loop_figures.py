"""Generate dependency-free SVGs for the LIBERO closed-loop report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "libero_standard_delay_ablation"
FIGURES = ROOT / "assets" / "figures"


def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="#202124">{text}</text>'
    )


def bar_chart() -> str:
    report = json.loads((RESULT / "d2_policy_matrix_analysis.json").read_text())
    labels = ["Sync", "Naive", "Skip", "Stale aug.", "Future state", "Learned stale"]
    keys = [
        "standard_sync",
        "standard_naive_d2",
        "standard_skip_d2",
        "stale_augmented_d2",
        "learned_future_d2",
        "learned_stale_input_d2",
    ]
    values = [100 * report["aggregate"][key]["success_rate"] for key in keys]
    colors = ["#4c78a8", "#e45756", "#72b7b2", "#f2cf5b", "#54a24b", "#b279a2"]
    width, height = 900, 500
    left, top, chart_h, bar_w, gap = 80, 65, 330, 90, 42
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(svg_text(width / 2, 32, "LIBERO task 3 at 200 ms: closed-loop success", 20))
    for tick in range(0, 61, 10):
        y = top + chart_h - tick / 60 * chart_h
        parts.append(f'<line x1="{left}" y1="{y}" x2="850" y2="{y}" stroke="#e0e0e0"/>')
        parts.append(svg_text(left - 12, y + 5, f"{tick}%", 12, "end"))
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + 30 + index * (bar_w + gap)
        h = value / 60 * chart_h
        y = top + chart_h - h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" rx="3"/>')
        parts.append(svg_text(x + bar_w / 2, y - 10, f"{value:.0f}%", 14))
        parts.append(svg_text(x + bar_w / 2, top + chart_h + 28, label, 12))
    parts.append(svg_text(width / 2, 480, "10 paired initial states per condition", 12))
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart() -> str:
    report = json.loads((RESULT / "standard_delay_analysis.json").read_text())
    delays = [100, 200, 400]
    naive = [100 * report["aggregate"][f"standard_naive|d{d // 100}"]["success_rate"] for d in delays]
    skip = [100 * report["aggregate"][f"standard_skip|d{d // 100}"]["success_rate"] for d in delays]
    width, height = 820, 500
    left, top, chart_w, chart_h = 90, 65, 650, 330
    x_positions = [left + index * chart_w / 2 for index in range(3)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(svg_text(width / 2, 32, "Standard Pi0.5: delay-alignment ablation", 20))
    for tick in range(0, 61, 10):
        y = top + chart_h - tick / 60 * chart_h
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e0e0e0"/>')
        parts.append(svg_text(left - 12, y + 5, f"{tick}%", 12, "end"))
    for x, delay in zip(x_positions, delays):
        parts.append(svg_text(x, top + chart_h + 28, f"{delay} ms", 13))
    sync_y = top + chart_h - 30 / 60 * chart_h
    parts.append(f'<line x1="{left}" y1="{sync_y}" x2="{left + chart_w}" y2="{sync_y}" stroke="#777" stroke-dasharray="7,6"/>')
    parts.append(svg_text(left + chart_w - 5, sync_y - 8, "sync baseline 30%", 12, "end"))
    for values, color, label in ((naive, "#e45756", "Naive delayed"), (skip, "#4c78a8", "Skip stale prefix")):
        points = []
        for x, value in zip(x_positions, values):
            y = top + chart_h - value / 60 * chart_h
            points.append(f"{x},{y}")
            parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')
            parts.append(svg_text(x, y - 13, f"{value:.0f}%", 13))
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4"/>')
        legend_y = 440 if label == "Naive delayed" else 466
        parts.append(f'<line x1="290" y1="{legend_y - 5}" x2="330" y2="{legend_y - 5}" stroke="{color}" stroke-width="4"/>')
        parts.append(svg_text(340, legend_y, label, 13, "start"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "libero_d2_policy_success.svg").write_text(bar_chart(), encoding="utf-8")
    (FIGURES / "libero_standard_delay_success.svg").write_text(line_chart(), encoding="utf-8")


if __name__ == "__main__":
    main()
