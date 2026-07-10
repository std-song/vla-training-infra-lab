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
    left, right, top, bottom = 86, 30, 60, 90
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
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{value:.1f}</text>')
    palette = ["#2f6f73", "#c55f36", "#455a8a", "#8b6f2f", "#5c7f3b", "#7a4f86"]
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = left + slot * i + slot / 2
        h = value / max_v * plot_h
        x = cx - bar_w / 2
        y = top + plot_h - h
        color = palette[i % len(palette)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{value:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    default = read_csv(RESULTS / "qwen25vl_paged_kv_continuous_batching.csv")
    labels = [r["scenario"].replace("continuous_", "cont\\n").replace("serial_no_batch", "serial") for r in default]
    bar_svg(
        "Continuous Batching Throughput, 512 MiB KV Budget",
        labels,
        [float(r["throughput_req_s"]) for r in default],
        "requests/s",
        FIGURES / "project3_paged_kv_throughput.svg",
    )
    bar_svg(
        "Continuous Batching P95 Latency, 512 MiB KV Budget",
        labels,
        [float(r["p95_latency_ms"]) for r in default],
        "P95 latency ms",
        FIGURES / "project3_paged_kv_p95_latency.svg",
    )

    sweep_files = [
        ("256", "qwen25vl_paged_kv_budget256.csv"),
        ("320", "qwen25vl_paged_kv_budget320.csv"),
        ("384", "qwen25vl_paged_kv_budget384.csv"),
    ]
    rows = []
    for budget, file_name in sweep_files:
        for row in read_csv(RESULTS / file_name):
            if row["scenario"] == "serial_no_batch":
                continue
            row = dict(row)
            row["budget"] = budget
            rows.append(row)
    labels = [f'{r["budget"]}MiB\\n{r["scenario"].replace("continuous_", "").replace("_kv", "")}' for r in rows]
    bar_svg(
        "KV Budget Sweep: Throughput",
        labels,
        [float(r["throughput_req_s"]) for r in rows],
        "requests/s",
        FIGURES / "project3_paged_kv_budget_sweep_throughput.svg",
    )
    bar_svg(
        "KV Budget Sweep: Peak KV Memory",
        labels,
        [float(r["peak_kv_mib"]) for r in rows],
        "MiB",
        FIGURES / "project3_paged_kv_budget_sweep_memory.svg",
    )


if __name__ == "__main__":
    main()
