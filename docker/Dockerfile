FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential cmake git vim ninja-build \
  python3 python3-pip \
  libjemalloc-dev tzdata \
  libedit-dev libxml2-dev zlib1g-dev libncurses5-dev libzstd-dev \
  && rm -rf /var/lib/apt/lists/*

# Clone LLVM
WORKDIR /opt
# RUN git clone --depth 1 https://github.com/llvm/llvm-project.git
RUN git clone --branch llvmorg-16.0.6 --depth 1 https://github.com/llvm/llvm-project.git

# Configure
WORKDIR /opt/llvm-project/build
RUN cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_PARALLEL_COMPILE_JOBS=2 \
  -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm-install

# Build and install
RUN ninja -j2 install

# Final image
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential cmake git vim python3 python3-pip ninja-build \
  libedit-dev libxml2-dev zlib1g-dev libncurses5-dev libzstd-dev \
  clang-16 \
  && rm -rf /var/lib/apt/lists/*

# Set working directory for dev
WORKDIR /workspace

COPY --from=builder /opt/llvm-install /usr/local
ENV PATH=/usr/local/bin:$PATH
