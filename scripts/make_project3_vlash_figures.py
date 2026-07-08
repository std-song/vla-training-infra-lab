from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "project3_vla_infer" / "results"
FIGURES = ROOT / "assets" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_name(name: str) -> str:
    return {
        "sync_chunk_blocking": "sync",
        "async_naive_queue": "async",
        "async_future_state_queue": "future",
        "async_future_state_quantized_q2": "future+q2",
    }.get(name, name)


def bar_svg(title: str, labels: list[str], values: list[float], ylabel: str, out: Path, precision: int = 2) -> None:
    width, height = 900, 440
    left, right, top, bottom = 92, 32, 62, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(values) * 1.22 if values else 1.0
    if max_v <= 0:
        max_v = 1.0
    slot = plot_w / max(len(values), 1)
    bar_w = slot * 0.55
    colors = ["#455a8a", "#c55f36", "#2f6f73", "#8b6f2f"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="25" y="{top + plot_h / 2}" transform="rotate(-90 25 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
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
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700">{value:.{precision}f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def line_svg(title: str, trace: list[dict[str, str]], scenario: str, out: Path) -> None:
    rows = [row for row in trace if row["scenario"] == scenario]
    width, height = 980, 420
    left, right, top, bottom = 76, 34, 58, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_t = max(float(row["time_s"]) for row in rows)
    min_y = -0.05
    max_y = 1.2

    def xy(time_s: float, value: float) -> tuple[float, float]:
        x = left + time_s / max_t * plot_w
        y = top + plot_h - (value - min_y) / (max_y - min_y) * plot_h
        return x, y

    def path_for(key: str) -> str:
        points = [xy(float(row["time_s"]), float(row[key])) for row in rows]
        return " ".join(("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}" for idx, (x, y) in enumerate(points))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = tick / 5 * max_t
        x, _ = xy(value, min_y)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}s</text>')
    for tick in range(5):
        value = min_y + tick / 4 * (max_y - min_y)
        _, y = xy(0.0, value)
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.1f}</text>')
    parts.append(f'<path d="{path_for("target")}" fill="none" stroke="#333" stroke-width="2.2"/>')
    parts.append(f'<path d="{path_for("state")}" fill="none" stroke="#2f6f73" stroke-width="2.4"/>')
    parts.append('<rect x="780" y="56" width="13" height="13" fill="#333"/>')
    parts.append('<text x="800" y="67" font-family="Arial" font-size="12">target</text>')
    parts.append('<rect x="780" y="80" width="13" height="13" fill="#2f6f73"/>')
    parts.append('<text x="800" y="91" font-family="Arial" font-size="12">simulated state</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = read_csv(RESULTS / "vlash_async_control_loop_summary.csv")
    trace = read_csv(RESULTS / "vlash_async_control_loop_trace.csv")
    labels = [short_name(row["scenario"]) for row in summary]
    bar_svg(
        "VLASH-Inspired Control Loop Error",
        labels,
        [float(row["mean_abs_error"]) for row in summary],
        "mean absolute error",
        FIGURES / "project3_vlash_async_error.svg",
        precision=3,
    )
    bar_svg(
        "VLASH-Inspired Reaction Latency",
        labels,
        [float(row["reaction_latency_ms"]) for row in summary],
        "milliseconds",
        FIGURES / "project3_vlash_async_reaction.svg",
        precision=1,
    )
    bar_svg(
        "VLASH-Inspired State Staleness",
        labels,
        [float(row["mean_state_staleness_ms"]) for row in summary],
        "milliseconds",
        FIGURES / "project3_vlash_async_staleness.svg",
        precision=1,
    )
    line_svg(
        "Future-State Async Queue Tracking",
        trace,
        "async_future_state_queue",
        FIGURES / "project3_vlash_async_trace.svg",
    )


if __name__ == "__main__":
    main()
