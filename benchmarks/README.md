# quake2iree benchmarks

This directory measures **kernel execution latency** across IREE backends and CUDA-Q
native simulation. Results are written as CSV files and can be turned into publication-
quality plots with `plots.py`.

## Structure

| File | Role |
|---|---|
| `backends.py` | Backend registry — `IREEBackend` and `CUDAQBackend` descriptors |
| `runner.py` | Timing harness — in-process compile + statistical timing loop |
| `01_latency.py` | GHZ latency sweep across qubit counts and backends |
| `plots.py` | Matplotlib plot generator from result CSVs |
| `results/` | Committed result CSVs (one file per run / hardware config) |

## How to run

Build the project first (`./build.sh`), then:

```bash
# CPU only (always works):
python3 benchmarks/01_latency.py --no-cuda

# CPU + NVIDIA GPU (requires CUDA driver):
python3 benchmarks/01_latency.py

# Custom qubit range and run count:
python3 benchmarks/01_latency.py --qubit-counts 2,4,8,12,16 --runs 500

# Generate plot immediately after benchmarking:
python3 benchmarks/01_latency.py --plot

# Plot an existing CSV:
python3 benchmarks/plots.py benchmarks/results/latency_2026-05-12.csv
```

## Output format

Each run produces a CSV with one row per (backend, n\_qubits) pair:

```
backend,n_qubits,mean_us,std_us,min_us,n_samples
iree-cpu,2,61.940,2.310,58.100,200
iree-cuda,2,918.190,15.200,890.000,200
cudaq-cpu,2,23233.010,320.500,22800.000,200
cudaq-gpu,2,10176.610,210.300,9950.000,200
```

## Measurement methodology

- All timing uses `time.perf_counter()` around a single kernel call.
- IREE kernels execute in-process via `iree.runtime` — no subprocess overhead.
- Each measurement is preceded by a configurable warmup phase (default 20 calls).
- Reported statistics: mean, standard deviation, and minimum over N calls (default 200).
- **Compilation time is excluded**: each kernel is compiled once as setup before timing.

## Adding a new backend

1. Add a descriptor in `backends.py` following the existing `IREEBackend` or
   `CUDAQBackend` pattern.
2. Pass the new backend to `run_benchmarks()` in the relevant benchmark script.
3. Optionally add a visual style entry in `plots.py` under `_STYLE`.

Currently stubbed but not wired into any script: `iree_rocm()`, `iree_vulkan()`,
`IREE_VMVX` — they are ready to use once the corresponding driver is available.

## Reproducibility (paper use)

Record the following in the paper or supplementary material:

- GPU model and driver version (`nvidia-smi`)
- CUDA-Q version (`python3 -c "import cudaq; print(cudaq.__version__)"`)
- IREE version (`iree-compile --version`)
- Python and OS version
- Exact command used to produce each result CSV

Commit the result CSVs to `benchmarks/results/` so reviewers can regenerate
plots without re-running the benchmarks.
