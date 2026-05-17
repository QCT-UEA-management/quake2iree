#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build" -G Ninja
cmake --build "$PROJECT_DIR/build" -j4
