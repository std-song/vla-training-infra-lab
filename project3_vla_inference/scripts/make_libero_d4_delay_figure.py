"""Generate a dependency-free SVG for the d<=4 LIBERO delay sweep."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "libero_d4_delay_sweep"
OUTPUT = ROOT / "assets" / "figures" / "libero_d4_delay_sweep.svg"


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="#202124">'
        f'{escape(value)}</text>'
    )


def panel(
    title: str,
    values: dict[str, list[float]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
    y_suffix: str,
) -> list[str]:
    colors = {
        "Stale-D4 policy": "#e45756",
        "Learned-D4 + predicted state": "#4c78a8",
        "Same Learned-D4 + stale state": "#54a24b",
    }
    parts = [text(x0 + width / 2, y0 - 25, title, 18)]
    for tick in range(6):
        value = y_max * tick / 5
        y = y0 + height - height * tick / 5
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + width}" y2="{y}" stroke="#e5e7eb"/>')
        label = f"{value:.0f}{y_suffix}"
        parts.append(text(x0 - 12, y + 4, label, 11, "end"))
    x_positions = [x0 + index * width / 4 for index in range(5)]
    for delay, x in enumerate(x_positions):
        parts.append(text(x, y0 + height + 24, f"d={delay}", 12))
    for label, series in values.items():
        points = []
        for x, value in zip(x_positions, series):
            y = y0 + height - min(value, y_max) / y_max * height
            points.append(f"{x},{y}")
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[label]}"/>')
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[label]}" stroke-width="3"/>'
        )
    return parts


def main() -> None:
    report = json.loads((RESULT / "d4_delay_analysis.json").read_text(encoding="utf-8"))
    aggregate = report["aggregate"]
    labels = {
        "Stale-D4 policy": "stale_policy",
        "Learned-D4 + predicted state": "learned_future",
        "Same Learned-D4 + stale state": "learned_stale",
    }
    success = {
        label: [100 * aggregate[f"{series}|d{delay}"]["success_rate"] for delay in range(5)]
        for label, series in labels.items()
    }
    handoff = {
        label: [aggregate[f"{series}|d{delay}"]["mean_handoff_action_l2"] for delay in range(5)]
        for label, series in labels.items()
    }

    width, height = 1200, 560
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(text(width / 2, 30, "Pi0.5 LIBERO task 3: d<=4 delay sweep", 21))
    parts.extend(panel("Closed-loop success", success, 80, 100, 450, 320, 60, "%"))
    parts.extend(panel("Action handoff L2", handoff, 680, 100, 450, 320, 1.2, ""))

    legend_y = 492
    colors = ["#e45756", "#4c78a8", "#54a24b"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = 110 + index * 355
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 34}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(text(x + 43, legend_y + 4, label, 12, "start"))
    parts.append(text(width / 2, 535, "10 paired initial states; 10 Hz control; d=0..4 means 0..400 ms logical delay", 12))
    parts.append("</svg>")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
