# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache-2.0 which accompanies this distribution.            #
# ============================================================================ #

# Build on top of the pre-built LLVM/MLIR Hub image.
# Adds:
#   - openssh-server  (SSH access from external tools / JetBrains / CLI)
#   - iree-compiler + iree-runtime Python packages (end-to-end IREE testing)
FROM marhvera/llvm-dev:latest

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# SSH server
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    # Create the privilege-separation directory required by sshd
    && mkdir -p /var/run/sshd \
    # Allow root login with password (dev container — not production)
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/'  /etc/ssh/sshd_config \
    # Set a default password for root; override at build time with --build-arg
    && echo "root:dev" | chpasswd \
    # Pre-generate SSH host keys so sshd starts without prompting
    && ssh-keygen -A

# ---------------------------------------------------------------------------
# IREE compiler + runtime (Python packages — self-contained, no LLVM conflict)
# ---------------------------------------------------------------------------
RUN pip3 install --break-system-packages \
        iree-compiler \
        iree-runtime \
        cudaq

# ---------------------------------------------------------------------------
# Expose SSH port
# ---------------------------------------------------------------------------
EXPOSE 22

# Default command — starts the SSH daemon in the foreground.
# The devcontainer runtime overrides this; sshd is started via
# postStartCommand in devcontainer.json instead.
CMD ["/usr/sbin/sshd", "-D"]
