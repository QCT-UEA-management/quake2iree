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
            n    = int(row["n_qubits"])
            mean = float(row["mean_us"])
            std  = float(row["std_us"])
            data[row["backend"]][n] = {
                "compile_ms": float(row.get("compile_ms", 0)),
                "mean_us":    mean,
                "median_us":  float(row.get("median_us", mean)),
                "std_us":     std,
                "p95_us":     float(row.get("p95_us", mean + std)),
                "min_us":     float(row["min_us"]),
            }
    return data


def plot_latency(csv_path: Path, output_path: Path | None = None) -> None:
    """Plot median execution time with [median, p95] band vs qubit count."""
    data = _load_csv(csv_path)
    if not data:
        raise ValueError(f"No data found in {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))

    for backend, by_n in sorted(data.items()):
        ns      = sorted(by_n)
        medians = [by_n[n]["median_us"] for n in ns]
        p95s    = [by_n[n]["p95_us"]    for n in ns]

        style = _STYLE.get(backend, _DEFAULT_STYLE)
        ax.plot(ns, medians, label=backend, linewidth=1.8, **style)
        ax.fill_between(ns, medians, p95s, alpha=0.12, color=style["color"])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.set_xlabel("Qubits", fontsize=11)
    ax.set_ylabel("Execution time (µs, median)", fontsize=11)
    ax.set_title("Kernel execution latency — GHZ circuit", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()

    dest = output_path or csv_path.with_suffix(".png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {dest}")


def plot_compile_time(csv_path: Path, output_path: Path | None = None) -> None:
    """Plot IREE compilation time vs qubit count (CUDA-Q JIT cost not measurable)."""
    data = _load_csv(csv_path)
    iree_data = {b: d for b, d in data.items()
                 if any(v["compile_ms"] > 0 for v in d.values())}
    if not iree_data:
        print("No compile_ms data found — skipping compile-time plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for backend, by_n in sorted(iree_data.items()):
        ns  = sorted(by_n)
        cms = [by_n[n]["compile_ms"] for n in ns]
        style = _STYLE.get(backend, _DEFAULT_STYLE)
        ax.plot(ns, cms, label=backend, linewidth=1.8,
                marker=style["marker"], color=style["color"],
                linestyle=style["linestyle"])

    ax.set_xlabel("Qubits", fontsize=11)
    ax.set_ylabel("Compilation time (ms)", fontsize=11)
    ax.set_title("AOT compilation time — GHZ circuit (CUDA-Q JIT not shown)", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()

    dest = output_path or csv_path.with_suffix("").with_name(
        csv_path.stem + "_compile.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved compile plot: {dest}")


def plot_total_time(csv_path: Path, output_path: Path | None = None,
                    max_evals: int = 1000) -> None:
    """Plot cumulative wall time (compile + N × exec) vs number of evaluations.

    Shows the crossover point where IREE's AOT compilation cost is amortised
    by faster per-call execution compared to CUDA-Q.
    X-axis: number of circuit evaluations (log scale).
    Y-axis: total time in milliseconds.
    """
    data = _load_csv(csv_path)
    if not data:
        raise ValueError(f"No data found in {csv_path}")

    ns = [1] + list(range(10, max_evals + 1, 10))

    fig, ax = plt.subplots(figsize=(9, 5))

    all_qubit_counts: set[int] = set()
    for backend, by_n in sorted(data.items()):
        all_qubit_counts.update(by_n.keys())
        # Average exec and compile cost across all qubit counts.
        all_medians = [v["median_us"] for v in by_n.values()]
        all_compile = [v["compile_ms"] for v in by_n.values()]
        median_us  = sum(all_medians) / len(all_medians)
        compile_ms = sum(all_compile) / len(all_compile)

        exec_ms_per_call = median_us / 1000.0
        totals = [compile_ms + n * exec_ms_per_call for n in ns]

        style = _STYLE.get(backend, _DEFAULT_STYLE)
        ax.plot(ns, totals, label=backend, linewidth=1.8, **style)

    qubit_str = ", ".join(str(q) for q in sorted(all_qubit_counts))
    ax.set_xscale("log")
    ax.set_xlabel("Number of circuit evaluations", fontsize=11)
    ax.set_ylabel("Total wall time (ms)", fontsize=11)
    ax.set_title(
        f"Total cost: compilation + N × execution  (qubits averaged: {qubit_str})",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()

    dest = output_path or csv_path.with_suffix("").with_name(
        csv_path.stem + "_total.png")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"Saved total-time plot: {dest}")


if __name__ == "__main__":
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if csv_file is None:
        print("Usage: python3 benchmarks/plots.py <results.csv> [output.png]")
        sys.exit(1)
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_latency(csv_file, out_file)
    plot_compile_time(csv_file)
    plot_total_time(csv_file)
