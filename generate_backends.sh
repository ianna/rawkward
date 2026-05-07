#!/usr/bin/env bash

set -e

LICENSE="// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

"

TODO="// TODO: implement
"

# Kernel names (shared across CPU/SIMD/CUDA/HIP)
KERNELS=(
  "reduce/sum"
  "reduce/prod"
  "reduce/min"
  "reduce/max"
  "reduce/argmax"
  "reduce/argmin"
  "reduce/count"
  "reduce/countnonzero"
  "reduce/sorting_ranges"
  "reduce/unique_offsets"
  "reduce/unique_ranges"

  "index/carry"
  "index/localindex"
  "index/getitem_nextcarry"
  "index/flatten"
  "index/ranges"

  "list/broadcast"
  "list/offsets_to_parents"
  "list/parents_to_offsets"
  "list/merge"
  "list/compact_offsets"

  "mask/byte_mask"
  "mask/bit_mask"
  "mask/indexed_mask"

  "util/error"
  "util/ptr"
  "util/unicode"
)

BACKENDS=("cpu" "simd" "cuda" "hip")

echo "Creating backend kernel modules…"

for backend in "${BACKENDS[@]}"; do
  for kernel in "${KERNELS[@]}"; do
    dir="src/kernels/${backend}/$(dirname "$kernel")"
    mkdir -p "$dir"

    case "$backend" in
      cpu|simd)
        file="src/kernels/${backend}/${kernel}.rs"
        ;;
      cuda)
        file="src/kernels/${backend}/${kernel}.cu"
        ;;
      hip)
        file="src/kernels/${backend}/${kernel}.hip.cpp"
        ;;
    esac

    if [[ ! -f "$file" ]]; then
      echo "Creating $file"
      printf "%s%s" "$LICENSE" "$TODO" > "$file"
    else
      echo "Skipping existing file $file"
    fi
  done
done

echo "Done."

