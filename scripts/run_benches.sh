#!/usr/bin/env bash
#
# Run the per-kernel benchmarks side-by-side with the awkward C++ kernels
# and emit BENCHMARKS.md.
#
# Usage:
#   scripts/run_benches.sh                        # uses ~/Projects/awkward.2.9.x/awkward/awkward-cpp
#   AWKWARD_CPP_PATH=/path/to/awkward-cpp scripts/run_benches.sh
#   scripts/run_benches.sh --rust-only            # skip the C++ side entirely
#
# Requires: rust toolchain, a C++17 compiler (clang on macOS works), python3
# (any version ≥ 3.8). The awkward-cpp checkout is only needed if you want
# the side-by-side comparison.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1-}" == "--rust-only" ]]; then
    FEATURES=""
    echo "[bench] running Rust kernels only — comparison column will be empty"
else
    FEATURES="--features bench-cxx"
    : "${AWKWARD_CPP_PATH:=$HOME/Projects/awkward.2.9.x/awkward/awkward-cpp}"
    if [[ ! -d "$AWKWARD_CPP_PATH/src/cpu-kernels" ]]; then
        echo "error: AWKWARD_CPP_PATH=$AWKWARD_CPP_PATH does not look right." >&2
        echo "       expected to find src/cpu-kernels/ inside it." >&2
        echo "       Set AWKWARD_CPP_PATH or pass --rust-only." >&2
        exit 1
    fi
    export AWKWARD_CPP_PATH
    echo "[bench] using awkward-cpp at $AWKWARD_CPP_PATH"
fi

# Run the criterion harness. --no-default-features avoids dragging in pyo3
# (the bench code doesn't need it and skipping it speeds up the build).
echo "[bench] cargo bench --no-default-features $FEATURES --bench kernels"
# shellcheck disable=SC2086
cargo bench --no-default-features $FEATURES --bench kernels

# Convert criterion's per-bench JSON into a single Markdown report.
echo "[bench] generating BENCHMARKS.md"
python3 scripts/bench_to_markdown.py

echo "[bench] done — see BENCHMARKS.md"
