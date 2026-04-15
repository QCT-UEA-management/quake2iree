#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "build/tool/q2i-opt" ]]; then
  echo "error: build/tool/q2i-opt was not found or is not executable."
  echo "Run ./build.sh before running tests."
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s test -v
