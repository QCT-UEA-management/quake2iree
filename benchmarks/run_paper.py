#!/usr/bin/env python3
"""Run all paper benchmark figures in parallel on multi-NUMA + A100 hardware.

Replaces run_paper.sh — pure Python, no numactl dependency.
NUMA CPU affinity is set via os.sched_setaffinity in subprocess preexec_fn,
which pins each child process (and any threads it spawns) to a NUMA node's
CPU set before the benchmark starts.

Strategy
--------
Phase 1 (simultaneous):
  CPU group  — 4 scripts pinned to separate NUMA nodes (background processes).
  GPU group  — F2a → F2b → F2c run serially on the A100 (foreground).
  Both groups overlap freely: the CPU is near-idle during GPU runs.

Phase 2:
  Wait for CPU background jobs, then report.

Expected wall time (paper run): ~4-6 h  (vs. ~12-16 h serial)

Usage
-----
  python3 benchmarks/run_paper.py           # full paper run (all figures)
  python3 benchmarks/run_paper.py --cpu     # CPU figures only (no GPU required)
  python3 benchmarks/run_paper.py --gpu     # GPU figures only
  python3 benchmarks/run_paper.py --test    # smoke test: 3 qubit sizes, 5 runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO    = Path(__file__).parent.parent
RESULTS = REPO / "benchmarks" / "results"

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

PAPER_QUBITS = "2,4,6,8,10,12,14,16,18,20,22,24"
PAPER_RUNS   = 200
PAPER_WARMUP = 20

HEA_QUBITS   = "2,4,6,8,10,12,14,16,18,20"  # CUDA-Q CPU too slow above 20

TEST_QUBITS  = "2,4,6"
TEST_RUNS    = 5
TEST_WARMUP  = 2

# ---------------------------------------------------------------------------
# NUMA helpers
# ---------------------------------------------------------------------------

def _parse_cpulist(cpulist: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpulist.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def get_numa_nodes() -> dict[int, set[int]]:
    """Return {node_id: cpu_set} from sysfs. Empty dict if NUMA unavailable."""
    node_dir = Path("/sys/devices/system/node")
    nodes: dict[int, set[int]] = {}
    for p in sorted(node_dir.glob("node[0-9]*")):
        node_id = int(p.name[4:])
        try:
            nodes[node_id] = _parse_cpulist((p / "cpulist").read_text())
        except OSError:
            pass
    return nodes


def _make_preexec(cpus: set[int]):
    def preexec():
        os.sched_setaffinity(0, cpus)
    return preexec


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Process launchers
# ---------------------------------------------------------------------------

def _build_cmd(
    script: str,
    backends: str,
    qubits: str,
    runs: int,
    warmup: int,
    output: str,
    plot: bool,
) -> list[str]:
    cmd = [
        sys.executable, f"benchmarks/{script}",
        "--backends",     backends,
        "--qubit-counts", qubits,
        "--runs",         str(runs),
        "--warmup",       str(warmup),
        "--output",       f"benchmarks/results/{output}",
    ]
    if plot:
        cmd.append("--plot")
    return cmd


def _launch_background(
    cmd: list[str],
    log_path: Path,
    cpus: set[int] | None,
) -> subprocess.Popen:
    """Start a process in the background, redirecting stdout+stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")  # closed when the process exits
    kwargs: dict = dict(stdout=log_file, stderr=log_file)
    if cpus:
        kwargs["preexec_fn"] = _make_preexec(cpus)
    return subprocess.Popen(cmd, **kwargs)


def _run_foreground(cmd: list[str]) -> None:
    """Run a process in the foreground; print a warning on non-zero exit."""
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"  WARNING: exited with code {result.returncode}: {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# Phase 1 — CPU group (parallel background jobs)
# ---------------------------------------------------------------------------

def launch_cpu_jobs(cfg: argparse.Namespace) -> list[tuple[subprocess.Popen, str]]:
    nodes   = get_numa_nodes()
    n_nodes = len(nodes)
    log(f"Phase 1 — launching CPU benchmarks ({n_nodes} NUMA node(s) detected)")
    if n_nodes < 4:
        log("  WARNING: fewer than 4 NUMA nodes; some jobs will share a node")

    prefix     = "test_" if cfg.test else "paper_"
    hea_qubits = TEST_QUBITS if cfg.test else HEA_QUBITS

    # (label, script, backends, qubit_override_or_None, output_csv)
    job_defs = [
        ("F1a GHZ  CPU", "01_ghz.py",       "iree-cpu,cudaq-cpu", None,       f"{prefix}ghz_cpu.csv"),
        ("F1b QFT  CPU", "03_qft.py",        "iree-cpu,cudaq-cpu", None,       f"{prefix}qft_cpu.csv"),
        ("F1c QAOA CPU", "04_qaoa.py",       "iree-cpu,cudaq-cpu", None,       f"{prefix}qaoa_cpu.csv"),
        ("F3  HEA  CPU", "02_parametric.py", "iree-cpu,cudaq-cpu", hea_qubits, f"{prefix}hea_cpu.csv"),
    ]

    procs: list[tuple[subprocess.Popen, str]] = []
    for i, (label, script, backends, qubit_override, output) in enumerate(job_defs):
        node_id  = i % max(n_nodes, 1)
        cpus     = nodes.get(node_id)
        qubits   = qubit_override if qubit_override else cfg.qubits
        log_path = RESULTS / output.replace(".csv", ".log")
        cmd      = _build_cmd(script, backends, qubits, cfg.runs, cfg.warmup, output, plot=True)
        proc     = _launch_background(cmd, log_path, cpus)
        pin_info = f"NUMA {node_id}" if cpus else "unpinned"
        log(f"  {label} → {pin_info}  pid={proc.pid}  log={log_path.name}")
        procs.append((proc, label))

    return procs


# ---------------------------------------------------------------------------
# Phase 1 — GPU group (serial foreground jobs)
# ---------------------------------------------------------------------------

def run_gpu_jobs(cfg: argparse.Namespace) -> None:
    log("Phase 1 — running GPU benchmarks serially (A100 exclusive)")
    prefix = "test_" if cfg.test else "paper_"

    for label, script, output in [
        ("F2a GHZ  GPU", "01_ghz.py",  f"{prefix}ghz_gpu.csv"),
        ("F2b QFT  GPU", "03_qft.py",  f"{prefix}qft_gpu.csv"),
        ("F2c QAOA GPU", "04_qaoa.py", f"{prefix}qaoa_gpu.csv"),
    ]:
        log(f"  {label} ...")
        _run_foreground(_build_cmd(
            script, "iree-cuda,cudaq-gpu",
            cfg.qubits, cfg.runs, cfg.warmup, output, plot=True,
        ))

    log("  GPU group done.")


# ---------------------------------------------------------------------------
# Phase 2 — collect CPU background jobs
# ---------------------------------------------------------------------------

def wait_cpu_jobs(procs: list[tuple[subprocess.Popen, str]]) -> None:
    log("Phase 3 — waiting for CPU background jobs ...")
    failed: list[str] = []
    for proc, label in procs:
        proc.wait()
        if proc.returncode != 0:
            failed.append(label)
            log(f"  FAILED  {label}  (exit {proc.returncode}) — check .log file")
        else:
            log(f"  done    {label}")
    if failed:
        log(f"  {len(failed)} job(s) failed: {failed}")
    else:
        log("  All CPU jobs completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu",  action="store_true", help="CPU figures only (no GPU required)")
    mode.add_argument("--gpu",  action="store_true", help="GPU figures only")
    parser.add_argument(
        "--test", action="store_true",
        help=f"smoke test with reduced parameters "
             f"(qubits={TEST_QUBITS}, runs={TEST_RUNS}, warmup={TEST_WARMUP})",
    )
    cfg = parser.parse_args()

    if cfg.test:
        cfg.qubits = TEST_QUBITS
        cfg.runs   = TEST_RUNS
        cfg.warmup = TEST_WARMUP
        log(f"TEST MODE — qubits={cfg.qubits}  runs={cfg.runs}  warmup={cfg.warmup}")
    else:
        cfg.qubits = PAPER_QUBITS
        cfg.runs   = PAPER_RUNS
        cfg.warmup = PAPER_WARMUP

    run_mode = "cpu" if cfg.cpu else ("gpu" if cfg.gpu else "all")
    log(f"Mode: {run_mode}  |  results → benchmarks/results/")
    RESULTS.mkdir(parents=True, exist_ok=True)

    cpu_procs: list[tuple[subprocess.Popen, str]] = []

    if run_mode in ("cpu", "all"):
        cpu_procs = launch_cpu_jobs(cfg)

    if run_mode in ("gpu", "all"):
        run_gpu_jobs(cfg)

    if cpu_procs:
        wait_cpu_jobs(cpu_procs)

    log("All done. Results in benchmarks/results/")


if __name__ == "__main__":
    main()
