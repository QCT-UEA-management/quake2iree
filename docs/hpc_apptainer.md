# Running quake2iree on an HPC cluster with Apptainer

The workflow has two phases:

1. **On your Mac** — build the Docker image from the Dockerfile and push it
   to Docker Hub.
2. **On the HPC cluster** — pull the image with Apptainer, build `q2i-opt`,
   and submit SLURM jobs.

---

## Phase 1 — Build and publish the Docker image (Mac)

### Prerequisites

- Docker Desktop installed and running.
- Logged in to Docker Hub with your account (`marhvera`):

  ```bash
  docker login
  ```

### Detect your Mac architecture

```bash
uname -m
# x86_64  → Intel Mac Pro
# arm64   → Apple Silicon Mac Pro (M-series)
```

This matters because HPC clusters run x86\_64 Linux.  Apple Silicon Macs build
ARM images by default, which will not run on the cluster.

### Build and push

**Intel Mac (x86\_64):**

```bash
cd /path/to/quake2iree

docker build \
    -t marhvera/quake2iree:latest \
    .devcontainer/

docker push marhvera/quake2iree:latest
```

**Apple Silicon Mac (arm64 — M1 / M2 / M3 / M4):**

Docker Buildx must cross-compile to `linux/amd64`:

```bash
cd /path/to/quake2iree

# Create a multi-platform builder if you don't have one yet:
docker buildx create --name multiarch --use

docker buildx build \
    --platform linux/amd64 \
    -t marhvera/quake2iree:latest \
    --push \
    .devcontainer/
```

The `--push` flag uploads directly without a local intermediate image, which
avoids a load step that does not work for cross-compiled images.

> **Build time**: expect 10–20 minutes the first time.  Subsequent builds are
> fast because the `marhvera/llvm-dev:latest` base layer is cached.

### Verify the image is live

```bash
docker pull marhvera/quake2iree:latest
docker run --rm marhvera/quake2iree:latest python3 -c "import iree.runtime; print('ok')"
```

---

## Phase 2 — HPC cluster setup

### 2.1 Load Apptainer

```bash
# Spack environment:
spack load apptainer

# Or module system (cluster-dependent):
module load apptainer    # or: module load singularity
```

### 2.2 Pull the SIF image

```bash
# Pull once — creates quake2iree.sif in the current directory (~1–2 GB):
apptainer pull quake2iree.sif docker://marhvera/quake2iree:latest
```

Store the SIF in a stable location such as `$HOME` or a project scratch
directory.  It is reused for every benchmark run.

```bash
# Keep it next to the repo for convenience:
mv quake2iree.sif ~/quake2iree/
```

### 2.3 Clone the repo

```bash
git clone <repo-url> ~/quake2iree
```

### 2.4 Build q2i-opt

`q2i-opt` is compiled from source using the LLVM toolchain inside the
container.  Run this once after cloning (or after any C++ changes):

```bash
apptainer exec \
    --bind ~/quake2iree:/workspaces/quake2iree \
    ~/quake2iree/quake2iree.sif \
    bash -c "cd /workspaces/quake2iree && bash scripts/build.sh"
```

The compiled binary lands in `~/quake2iree/build/bin/q2i-opt` on the cluster
filesystem.

### 2.5 Smoke test

```bash
apptainer exec \
    --bind ~/quake2iree:/workspaces/quake2iree \
    ~/quake2iree/quake2iree.sif \
    python3 -m pytest test/test_circuits.py -q
```

All 25 tests should pass.

---

## Running benchmarks interactively

Quick check on the login node (CPU only, small qubit range):

```bash
apptainer exec \
    --bind ~/quake2iree:/workspaces/quake2iree \
    ~/quake2iree/quake2iree.sif \
    python3 benchmarks/03_qft.py \
        --backends iree-cpu \
        --qubit-counts 2,4,6 --runs 10 --warmup 5
```

---

## SLURM job scripts

Copy these to `~/quake2iree/scripts/` and submit with `sbatch`.

### CPU benchmark

```bash
#!/bin/bash
#SBATCH --job-name=q2i-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpu_%j.out

REPO=$HOME/quake2iree
SIF=$REPO/quake2iree.sif

apptainer exec \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/$BENCHMARK_SCRIPT \
        --backends iree-cpu,cudaq-cpu \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24,26 \
        --runs 200 --warmup 20 --plot \
        --output benchmarks/results/$OUTPUT_CSV
```

Submit for each algorithm:

```bash
mkdir -p ~/quake2iree/logs

sbatch --export=BENCHMARK_SCRIPT=01_latency.py,OUTPUT_CSV=paper_ghz_cpu.csv  scripts/slurm_cpu.sh
sbatch --export=BENCHMARK_SCRIPT=03_qft.py,OUTPUT_CSV=paper_qft_cpu.csv      scripts/slurm_cpu.sh
sbatch --export=BENCHMARK_SCRIPT=04_qaoa.py,OUTPUT_CSV=paper_qaoa_cpu.csv    scripts/slurm_cpu.sh
```

### NVIDIA GPU benchmark

```bash
#!/bin/bash
#SBATCH --job-name=q2i-gpu-nvidia
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_nvidia_%j.out

REPO=$HOME/quake2iree
SIF=$REPO/quake2iree.sif

# --nv passes through the host NVIDIA driver and CUDA libraries
apptainer exec --nv \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/$BENCHMARK_SCRIPT \
        --backends iree-cuda,cudaq-gpu \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24 \
        --runs 200 --warmup 20 --plot \
        --output benchmarks/results/$OUTPUT_CSV
```

```bash
sbatch --export=BENCHMARK_SCRIPT=01_latency.py,OUTPUT_CSV=paper_ghz_gpu.csv  scripts/slurm_gpu_nvidia.sh
sbatch --export=BENCHMARK_SCRIPT=03_qft.py,OUTPUT_CSV=paper_qft_gpu.csv      scripts/slurm_gpu_nvidia.sh
sbatch --export=BENCHMARK_SCRIPT=04_qaoa.py,OUTPUT_CSV=paper_qaoa_gpu.csv    scripts/slurm_gpu_nvidia.sh
```

### AMD GPU benchmark (portability figure)

```bash
#!/bin/bash
#SBATCH --job-name=q2i-gpu-amd
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_amd_%j.out

REPO=$HOME/quake2iree
SIF=$REPO/quake2iree.sif

# --rocm passes through the host ROCm/HIP stack
apptainer exec --rocm \
    --bind $REPO:/workspaces/quake2iree \
    $SIF \
    python3 benchmarks/03_qft.py \
        --backends iree-rocm \
        --qubit-counts 2,4,6,8,10,12,14,16,18,20,22,24 \
        --runs 200 --warmup 20 \
        --output benchmarks/results/paper_qft_portability_amd.csv
```

Monitor all jobs:

```bash
squeue -u $USER
tail -f ~/quake2iree/logs/cpu_<jobid>.out
```

---

## Copy results back to your Mac

```bash
scp -r <user>@<cluster>:~/quake2iree/benchmarks/results/ \
    /path/to/quake2iree/benchmarks/results/
```

Then generate or regenerate plots locally:

```bash
python3 benchmarks/plots.py benchmarks/results/paper_qft_cpu.csv
```

---

## Verify GPU passthrough

```bash
# NVIDIA
apptainer exec --nv ~/quake2iree/quake2iree.sif nvidia-smi

# AMD
apptainer exec --rocm ~/quake2iree/quake2iree.sif rocm-smi

# IREE driver list
apptainer exec --nv ~/quake2iree/quake2iree.sif \
    python3 -c "import iree.runtime as r; print(r.query_available_drivers())"
# Expected: cuda, hip, local-task, local-sync, vulkan
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `FATAL: kernel does not support unprivileged user namespaces` | Cluster policy | Ask sysadmin to enable user namespaces or `--fakeroot` |
| `cuda: driver not initialized` | Missing flag | Add `--nv` to the `apptainer exec` command |
| `hip: driver not initialized` | Missing flag | Add `--rocm` to the `apptainer exec` command |
| `ModuleNotFoundError: No module named 'q2i'` | Missing bind mount | Ensure `--bind ~/quake2iree:/workspaces/quake2iree` is present |
| `q2i-opt: command not found` | Build step skipped | Run Phase 2 step 2.4 first |
| Wrong architecture (`Exec format error`) | ARM image on x86 cluster | Rebuild with `--platform linux/amd64` (Apple Silicon section above) |
| CUDA version mismatch | Container runtime newer than host driver | Rebuild image from a base image that matches the cluster CUDA version |
