// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

// ── Segmented count kernel ────────────────────────────────────────────────────
//
// out[seg] = offsets[seg+1] - offsets[seg].
// Single kernel (output always long long regardless of data type).

extern "C" __global__ void segmented_count(
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;
    out[seg] = offsets[seg + 1] - offsets[seg];
}
