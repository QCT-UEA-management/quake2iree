# Compatibility shim — logic has moved to q2i/benchmark.py
from q2i.benchmark import *  # noqa: F401, F403
from q2i.benchmark import (
    TimingResult, compile_for_iree, time_iree, time_cudaq,
    run_sweep, save_csv, benchmark_cli,
)
