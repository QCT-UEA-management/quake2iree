# quake2iree benchmarks

This directory measures **kernel execution latency** across IREE backends and CUDA-Q
native simulation. Results are written as CSV files and can be turned into publication-
quality plots with `plots.py`.

## Structure

| File | Role |
| --- | --- |
| `backends.py` | Backend registry — `IREEBackend` and `CUDAQBackend` descriptors |
| `runner.py` | Timing harness — in-process compile + statistical timing loop |
| `01_ghz.py` | GHZ latency sweep across qubit counts and backends |
| `02_parametric.py` | Hardware-efficient ansatz with runtime angles — amortisation sweep |
| `03_qft.py` | Quantum Fourier Transform — O(n²) CR1 gates, compile-time constant angles |
| `04_qaoa.py` | QAOA-MaxCut on a ring graph — parametric circuit, amortisation benchmark |
| `plots.py` | Matplotlib plot generator from result CSVs |
| `results/` | Committed result CSVs (one file per run / hardware config) |

## How to run

All commands below must be run from the **repo root** (`quake2iree/`).

### 1. Prerequisites

```bash
# Build q2i-opt (the MLIR lowering tool)
bash scripts/build.sh

# Python dependencies (if not already installed)
pip install iree-compiler iree-runtime cudaq matplotlib
```

### 2. Start here — CPU only (no GPU required)

This is the safest starting point. It runs both IREE and CUDA-Q on CPU so you
can compare them without needing a GPU or CUDA driver.

```bash
# Fixed-circuit benchmark (GHZ)
python3 benchmarks/01_ghz.py --no-cuda

# Parametric benchmark (hardware-efficient ansatz with runtime angles)
python3 benchmarks/02_parametric.py --no-cuda
```

Both scripts sweep qubit counts `2,4,6,8,10,12` by default, run 200 timed
calls per measurement (after 20 warmup calls), and write a CSV to
`benchmarks/results/<circuit>_YYYY-MM-DD.csv`.

### 3. Choosing which backends to run

Use `--backends` to select any combination explicitly. This overrides `--no-cuda`.

```bash
# Available backends: iree-cpu, iree-cuda, iree-vmvx, cudaq-cpu, cudaq-gpu
#   iree-cpu   — IREE compiled to native CPU via LLVM (always available)
#   iree-cuda  — IREE compiled to NVIDIA CUDA (requires CUDA driver)
#   iree-vmvx  — IREE portable reference interpreter (slowest, always available)
#   cudaq-cpu  — CUDA-Q qpp-cpu simulator (reference)
#   cudaq-gpu  — CUDA-Q NVIDIA GPU simulator (requires CUDA driver)

# IREE CPU vs CUDA-Q CPU only
python3 benchmarks/01_ghz.py --backends iree-cpu,cudaq-cpu

# IREE CPU vs IREE CUDA (pure IREE comparison, no CUDA-Q)
python3 benchmarks/01_ghz.py --backends iree-cpu,iree-cuda

# All backends — requires CUDA driver for iree-cuda and cudaq-gpu
python3 benchmarks/01_ghz.py
```

### 4. Customising the sweep

```bash
# Larger qubit range and more timing samples (for a paper)
python3 benchmarks/01_ghz.py \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20 \
    --runs 500 \
    --warmup 50 \
    --backends iree-cpu,iree-cuda,cudaq-cpu,cudaq-gpu

# Write results to a specific file
python3 benchmarks/01_ghz.py --output benchmarks/results/my_run.csv
```

### 5. Generating plots

Add `--plot` to any command to produce PNGs alongside the CSV:

```bash
python3 benchmarks/01_ghz.py --no-cuda --plot
python3 benchmarks/02_parametric.py --no-cuda --plot
```

`01_ghz.py --plot` generates two figures:

- `<name>.png` — kernel execution latency (median line, shaded to p95)
- `<name>_compile.png` — IREE AOT compilation time vs qubit count

`02_parametric.py --plot` generates three figures (the two above plus):

- `<name>_total.png` — total wall time (compile + N × exec) vs number of
  evaluations, showing the crossover point where IREE's AOT cost pays off

To plot an existing CSV without re-running the benchmark:

```bash
# Generates all applicable plots from an existing CSV
python3 benchmarks/plots.py benchmarks/results/parametric-hea_2026-05-18.csv

# Individual plots
python3 -c "
from pathlib import Path
from benchmarks.plots import plot_latency, plot_compile_time, plot_total_time
csv = Path('benchmarks/results/parametric-hea_2026-05-18.csv')
plot_latency(csv)       # <name>.png
plot_compile_time(csv)  # <name>_compile.png  (IREE only)
plot_total_time(csv)    # <name>_total.png    (crossover chart)
"
```

## Output format

Each run produces a CSV with one row per (backend, n\_qubits) pair:

```csv
backend,n_qubits,compile_ms,mean_us,median_us,std_us,p95_us,min_us,n_samples
iree-cpu,2,403.1,41.564,41.234,1.142,42.557,40.498,200
cudaq-cpu,2,0.0,23040.494,22287.970,3021.282,27244.716,19667.516,200
```

`compile_ms` is the one-time AOT compilation cost for IREE (q2i-opt + IREE/LLVM
pipeline). It is `0.0` for CUDA-Q because its JIT cost is opaque and absorbed
into the warmup phase.

## Measurement methodology

- All timing uses `time.perf_counter()` around a single kernel call.
- IREE kernels execute in-process via `iree.runtime` — no subprocess overhead.
- Each measurement is preceded by a configurable warmup phase (default 20 calls).
- Reported statistics: median, p95, mean, standard deviation, and minimum
  over N calls (default 200).
- **Execution timing excludes compilation**: each kernel is compiled once before
  the timing loop, and the compilation time is recorded separately in `compile_ms`.
- **IREE `compile_ms`** covers the full AOT pipeline: quake IR emission,
  `q2i-opt` lowering, and `iree-compile` (LLVM backend). At 12 qubits this
  is approximately 1 second.
- **CUDA-Q `compile_ms` is always 0**: JIT compilation happens inside the first
  `cudaq.get_state()` call and cannot be separated from execution without
  modifying CUDA-Q internals. It is absorbed into the warmup phase.
- The **median** is the primary execution metric; it is robust to the occasional
  GC pause that inflates the mean (visible as high p95 for CUDA-Q).
- For workloads that evaluate the same circuit repeatedly (e.g. parametric
  sweeps with runtime-argument kernels), IREE amortises its compilation cost
  after approximately `compile_ms / (cudaq_median_ms − iree_median_ms)` calls
  — roughly 40 calls at 12 qubits.

## Adding a new benchmark circuit

Create a new file `benchmarks/NN_name.py`. It only needs a kernel factory
and one call to `benchmark_cli`.

**Non-parametric circuit** (fixed gates, no runtime angles):

```python
#!/usr/bin/env python3
"""One-line description of the circuit."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from benchmarks.runner import benchmark_cli

def make_my_kernel(n_qubits: int):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n_qubits)
    # ... define gates ...
    return kernel

if __name__ == "__main__":
    benchmark_cli("MyCircuit", make_my_kernel)
```

**Parametric circuit** (runtime angles, e.g. for VQE/QAOA inner loops):

```python
def make_my_kernel(n_qubits: int):
    n_params = n_qubits  # or however many angles you need
    result = cudaq.make_kernel(*([float] * n_params))
    kernel, *thetas = result
    q = kernel.qalloc(n_qubits)
    for i, theta in enumerate(thetas):
        kernel.ry(theta, q[i])
    # Fixed representative values — exec time is angle-independent.
    fixed_angles = [0.785398] * n_params
    return kernel, fixed_angles  # return tuple; runner.py detects it

if __name__ == "__main__":
    benchmark_cli("MyParametric", make_my_kernel, extra_plots=["total_time"])
```

All CLI flags (`--backends`, `--qubit-counts`, `--runs`, `--plot`, …), CSV
saving, and plotting are handled by `runner.py` automatically.

## Adding a new hardware backend

All backend definitions live in `backends.py`. To add a new target (e.g. Metal,
ROCm with a specific chip):

1. Define a factory function or constant in `backends.py`:

   ```python
   def iree_metal() -> IREEBackend:
       return IREEBackend(name="iree-metal", target_backend="metal", driver="metal")
   ```

2. Add a `case` to `resolve_backend()` in `backends.py`.
3. Add the name string to `KNOWN_BACKENDS` in `backends.py`.

That is all — `runner.py` and the benchmark scripts need no changes. The new
backend becomes immediately available via `--backends iree-metal`.

Currently available backends: `iree-cpu`, `iree-cuda`, `iree-rocm`,
`iree-vulkan`, `iree-metal`, `iree-vmvx`, `cudaq-cpu`, `cudaq-gpu`.

## Reproducibility (paper use)

Record the following in the paper or supplementary material:

- GPU model and driver version (`nvidia-smi`)
- CUDA-Q version (`python3 -c "import cudaq; print(cudaq.__version__)"`)
- IREE version (`iree-compile --version`)
- Python and OS version
- Exact command used to produce each result CSV

Commit the result CSVs to `benchmarks/results/` so reviewers can regenerate
plots without re-running the benchmarks.
