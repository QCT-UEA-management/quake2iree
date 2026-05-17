"""Generate latency plots from benchmark CSV results.

Usage (standalone):
    python3 benchmarks/plots.py results/latency_2026-05-12.csv [output.png]

If output path is omitted, saves a PNG next to the CSV.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Visual style keyed by backend name. Add entries here when new backends land.
_STYLE: dict[str, dict] = {
    "iree-cpu":    {"color": "#2196F3", "marker": "o", "linestyle": "-"},
    "iree-cuda":   {"color": "#4CAF50", "marker": "s", "linestyle": "-"},
    "iree-rocm":   {"color": "#FF9800", "marker": "^", "linestyle": "-"},
    "iree-vulkan": {"color": "#9C27B0", "marker": "D", "linestyle": "-"},
    "iree-vmvx":   {"color": "#00BCD4", "marker": "v", "linestyle": "--"},
    "cudaq-cpu":   {"color": "#F44336", "marker": "o", "linestyle": "--"},
    "cudaq-gpu":   {"color": "#880E4F", "marker": "s", "linestyle": "--"},
}
_DEFAULT_STYLE: dict = {"color": "gray", "marker": "x", "linestyle": ":"}


def _load_csv(csv_path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Load results CSV into {backend: {n_qubits: {stat: value}}}."""
    data: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n_qubits"])
            data[row["backend"]][n] = {
                "mean_us": float(row["mean_us"]),
                "std_us":  float(row["std_us"]),
                "min_us":  float(row["min_us"]),
            }
    return data


def plot_latency(csv_path: Path, output_path: Path | None = None) -> None:
    """Plot mean execution time ± 1 std vs qubit count for all backends."""
    data = _load_csv(csv_path)
    if not data:
        raise ValueError(f"No data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))

    for backend, by_n in sorted(data.items()):
        ns    = sorted(by_n)
        means = [by_n[n]["mean_us"] for n in ns]
        stds  = [by_n[n]["std_us"]  for n in ns]
        lo    = [m - s for m, s in zip(means, stds)]
        hi    = [m + s for m, s in zip(means, stds)]

        style = _STYLE.get(backend, _DEFAULT_STYLE)
        ax.plot(ns, means, label=backend, linewidth=1.8, **style)
        ax.fill_between(ns, lo, hi, alpha=0.12, color=style["color"])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.set_xlabel("Qubits", fontsize=11)
    ax.set_ylabel("Execution time (µs)", fontsize=11)
    ax.set_title("Kernel execution latency — GHZ circuit", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()

    dest = output_path or csv_path.with_suffix(".png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {dest}")


if __name__ == "__main__":
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if csv_file is None:
        print("Usage: python3 benchmarks/plots.py <results.csv> [output.png]")
        sys.exit(1)
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_latency(csv_file, out_file)
