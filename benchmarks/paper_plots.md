# Paper figure generation plan

This document lists the exact commands to produce every figure for the paper,
in order.  Run all commands from the **repo root** (`quake2iree/`).

---

## Figures overview

| ID  | Title                           | Script               | Backends                       | Where to run    |
| --- | ------------------------------- | -------------------- | ------------------------------ | --------------- |
| F1a | GHZ — CPU execution latency     | `01_ghz.py`      | iree-cpu, cudaq-cpu            | CPU instance    |
| F1b | QFT — CPU execution latency     | `03_qft.py`          | iree-cpu, cudaq-cpu            | CPU instance    |
| F1c | QAOA — CPU execution latency    | `04_qaoa.py`         | iree-cpu, cudaq-cpu            | CPU instance    |
| F2a | GHZ — GPU execution latency     | `01_ghz.py`      | iree-cuda, cudaq-gpu           | NVIDIA instance |
| F2b | QFT — GPU execution latency     | `03_qft.py`          | iree-cuda, cudaq-gpu           | NVIDIA instance |
| F2c | QAOA — GPU execution latency    | `04_qaoa.py`         | iree-cuda, cudaq-gpu           | NVIDIA instance |
| F3  | HEA — compile cost amortisation | `02_parametric.py`   | iree-cpu, cudaq-cpu            | CPU instance    |
| F4  | QFT — IREE portability          | `03_qft.py`          | iree-cpu, iree-cuda, iree-rocm | NVIDIA + AMD    |

---

## Qubit range

| Figure         | Qubit counts                          | Reason for cap                                                                    |
| -------------- | ------------------------------------- | --------------------------------------------------------------------------------- |
| F1 CPU         | `2,4,6,8,10,12,14,16,18,20,22,24,26`  | CPU RAM is not a constraint                                                       |
| F2 GPU         | `2,4,6,8,10,12,14,16,18,20,22,24`     | 4 GB VRAM (Quadro T1000); 26 qubits → 512 MB statevector + runtime overhead       |
| F3 HEA         | `2,4,6,8,10,12,14,16,18,20`           | beyond 20 qubits CUDA-Q CPU timing becomes very slow                              |
| F4 portability | `2,4,6,8,10,12,14,16,18,20,22,24`     | matches F2 GPU range for a fair comparison                                        |

> **CUDA-Q CPU slow note**: QFT has O(n²) gates × 2ⁿ state elements.  Above
> 20 qubits a single CUDA-Q call can take seconds.  With 500 runs that becomes
> hours.  If CUDA-Q CPU runs time out, cap F1b/F1c at 20 qubits and note it.

---

## F1 — CPU execution latency (iree-cpu vs cudaq-cpu)

Run on any CPU instance (dev container is fine).  Each command writes a CSV
and two PNGs (latency + compile time).

> **Thermal note (laptop)**: the i7-10850H throttles under sustained load.
> Use `--runs 200 --warmup 20` (shown below).  Run one script at a time and
> allow a few minutes of cooling between them.  Monitor CPU temperature with `watch -n1 sensors` (requires `lm-sensors`).
> If the frequency drops or temperature exceeds ~90 °C mid-run, results are
> unreliable; wait for the machine to cool before rerunning.

```bash
# F1a — GHZ
python3 benchmarks/01_ghz.py \
    --backends iree-cpu,cudaq-cpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 200 --warmup 20 \
    --plot \
    --output benchmarks/results/paper_ghz_cpu.csv

# F1b — QFT
python3 benchmarks/03_qft.py \
    --backends iree-cpu,cudaq-cpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 200 --warmup 20 \
    --plot \
    --output benchmarks/results/paper_qft_cpu.csv

# F1c — QAOA-MaxCut
python3 benchmarks/04_qaoa.py \
    --backends iree-cpu,cudaq-cpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 200 --warmup 20 \
    --plot \
    --output benchmarks/results/paper_qaoa_cpu.csv
```

Output files:

- `paper_ghz_cpu.csv` / `.png` / `_compile.png`
- `paper_qft_cpu.csv` / `.png` / `_compile.png`
- `paper_qaoa_cpu.csv` / `.png` / `_compile.png`

---

## F2 — GPU execution latency (iree-cuda vs cudaq-gpu)

Run on the **NVIDIA instance** (see hardware section below).

```bash
# F2a — GHZ
python3 benchmarks/01_ghz.py \
    --backends iree-cuda,cudaq-gpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 500 --warmup 50 \
    --plot \
    --output benchmarks/results/paper_ghz_gpu.csv

# F2b — QFT
python3 benchmarks/03_qft.py \
    --backends iree-cuda,cudaq-gpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 500 --warmup 50 \
    --plot \
    --output benchmarks/results/paper_qft_gpu.csv

# F2c — QAOA-MaxCut
python3 benchmarks/04_qaoa.py \
    --backends iree-cuda,cudaq-gpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 500 --warmup 50 \
    --plot \
    --output benchmarks/results/paper_qaoa_gpu.csv
```

---

## F3 — HEA compile cost amortisation

Run on CPU.  The `--plot` flag generates the `_total.png` crossover chart.

```bash
python3 benchmarks/02_parametric.py \
    --backends iree-cpu,cudaq-cpu \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20 \
    --runs 500 --warmup 50 \
    --plot \
    --output benchmarks/results/paper_hea_cpu.csv
```

Output files:

- `paper_hea_cpu.csv` / `.png` (latency) / `_compile.png` / `_total.png`

The key figure for the paper is `_total.png`: it shows the crossover point
where IREE's AOT cost is paid back by faster per-call execution.

---

## F4 — QFT portability across IREE backends

Run the **same CSV-generating command twice** — once on the NVIDIA instance,
once on the AMD instance — then merge the CSVs.

```bash
# On NVIDIA instance (adds iree-cpu and iree-cuda rows):
python3 benchmarks/03_qft.py \
    --backends iree-cpu,iree-cuda \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 500 --warmup 50 \
    --output benchmarks/results/paper_qft_portability_nvidia.csv

# On AMD instance (adds iree-rocm rows):
python3 benchmarks/03_qft.py \
    --backends iree-rocm \
    --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
    --runs 500 --warmup 50 \
    --output benchmarks/results/paper_qft_portability_amd.csv

# Merge and plot (run anywhere after copying both CSVs to the same machine):
python3 - << 'EOF'
import csv, pathlib
from benchmarks.plots import plot_latency, plot_compile_time

out = pathlib.Path("benchmarks/results/paper_qft_portability.csv")
rows = []
header = None
for src in ["paper_qft_portability_nvidia.csv", "paper_qft_portability_amd.csv"]:
    p = pathlib.Path("benchmarks/results") / src
    with open(p) as f:
        reader = csv.DictReader(f)
        if header is None:
            header = reader.fieldnames
        rows.extend(list(reader))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"Merged: {out}")
plot_latency(out)
plot_compile_time(out)
EOF
```

---

## Hardware recommendations

### NVIDIA instance (F2 + F4 NVIDIA leg)

**AWS g4dn.xlarge** — cheapest instance with a CUDA-capable GPU:

| Spec       | Value                                                        |
| ---------- | ------------------------------------------------------------ |
| GPU        | NVIDIA T4 (16 GB VRAM)                                       |
| vCPUs      | 4                                                            |
| RAM        | 16 GB                                                        |
| On-demand  | ~$0.53 / hr                                                  |
| AMI        | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)  |

```bash
# Verify GPU is visible:
nvidia-smi

# Verify IREE CUDA backend:
iree-compile --version
python3 -c "import iree.runtime as r; print(r.query_available_drivers())"
```

If T4 memory is tight at 26 qubits, **g5.xlarge** (A10G, 24 GB VRAM, ~$1.01/hr)
is the next step up.

### AMD instance (F4 AMD leg)

AWS does not offer a reliable ROCm-ready instance as of mid-2026.  Recommended
alternatives:

| Provider         | Instance / Config                         | Approx. cost |
| ---------------- | ----------------------------------------- | ------------ |
| **Vast.ai**      | AMD RX 7900 XTX or MI210 (filter by ROCm) | $0.30–$1/hr  |
| **Lambda Labs**  | AMD MI250 nodes (when available)          | ~$1.50/hr    |
| **On-premise**   | Any machine with a recent AMD dGPU + ROCm | —            |

ROCm setup checklist on the AMD instance:

```bash
# 1. Install ROCm (if not pre-installed):
#    https://rocm.docs.amd.com/en/latest/deploy/linux/index.html

# 2. Verify GPU is visible:
rocm-smi

# 3. Install IREE with ROCm support:
pip install iree-compiler iree-runtime  # must be a ROCm-enabled build

# 4. Verify the rocm driver is available:
python3 -c "import iree.runtime as r; print(r.query_available_drivers())"
# Should include 'rocm'

# 5. Build q2i-opt as usual:
bash scripts/build.sh
```

> **Note**: IREE's ROCm backend targets AMD GPUs via the HIP/ROCm stack.  The
> same compiled VMFB cannot run on both NVIDIA and AMD — each target backend
> produces a different binary.  This is exactly the point of the F4 figure:
> the *source circuit* (quake IR) is identical; only the `--iree-hal-target-backends`
> flag changes.

---

## Running on an HPC cluster with Apptainer

See [docs/hpc_apptainer.md](../docs/hpc_apptainer.md) for the full setup
guide (building the SIF, SLURM scripts, GPU passthrough, troubleshooting).

The commands below assume:
- `$REPO` = path to the cloned repo on the cluster (`~/quake2iree`)
- `$SIF`  = `$REPO/quake2iree.sif` (built once from `quake2iree.def`)
- q2i-opt already built via Step 2 of the guide

### F1 — CPU (all three algorithms)

```bash
for SCRIPT in 01_ghz.py 03_qft.py 04_qaoa.py; do
    apptainer exec \
        --bind $REPO:/workspaces/quake2iree \
        $SIF \
        python3 benchmarks/$SCRIPT \
            --backends iree-cpu,cudaq-cpu \
            --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
            --runs 200 --warmup 20 --plot
done
```

### F2 — NVIDIA GPU (all three algorithms)

```bash
for SCRIPT in 01_ghz.py 03_qft.py 04_qaoa.py; do
    apptainer exec --nv \
        --bind $REPO:/workspaces/quake2iree \
        $SIF \
        python3 benchmarks/$SCRIPT \
            --backends iree-cuda,cudaq-gpu \
            --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24 \
            --runs 200 --warmup 20 --plot
done
```

### F3 — HEA amortisation

```bash
apptainer exec \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/02_parametric.py \
        --backends iree-cpu,cudaq-cpu \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20 \
        --runs 200 --warmup 20 --plot \
        --output benchmarks/results/paper_hea_cpu.csv
```

### F4 — QFT portability

```bash
# NVIDIA leg (run on NVIDIA node):
apptainer exec --nv \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/03_qft.py \
        --backends iree-cpu,iree-cuda \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24 \
        --runs 200 --warmup 20 \
        --output benchmarks/results/paper_qft_portability_nvidia.csv

# AMD leg (run on AMD node):
apptainer exec --rocm \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/03_qft.py \
        --backends iree-rocm \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24 \
        --runs 200 --warmup 20 \
        --output benchmarks/results/paper_qft_portability_amd.csv
```

After both jobs complete, copy the CSVs to one machine and run the merge
snippet in the **F4** section above to produce the combined plot.

---

## Re-plotting from existing CSVs

If you need to regenerate plots without re-running benchmarks:

```bash
python3 benchmarks/plots.py benchmarks/results/paper_ghz_cpu.csv
python3 benchmarks/plots.py benchmarks/results/paper_qft_cpu.csv
python3 benchmarks/plots.py benchmarks/results/paper_qaoa_cpu.csv
# etc. — plots.py always generates latency + compile + total (when data present)
```

---

## Checklist

- [ ] F1a GHZ CPU — `paper_ghz_cpu.csv` + `.png` + `_compile.png`
- [ ] F1b QFT CPU — `paper_qft_cpu.csv` + `.png` + `_compile.png`
- [ ] F1c QAOA CPU — `paper_qaoa_cpu.csv` + `.png` + `_compile.png`
- [ ] F2a GHZ GPU — `paper_ghz_gpu.csv` + `.png` + `_compile.png`
- [ ] F2b QFT GPU — `paper_qft_gpu.csv` + `.png` + `_compile.png`
- [ ] F2c QAOA GPU — `paper_qaoa_gpu.csv` + `.png` + `_compile.png`
- [ ] F3  HEA amortisation — `paper_hea_cpu.csv` + `_total.png`
- [ ] F4  QFT portability — `paper_qft_portability.csv` + `.png`
