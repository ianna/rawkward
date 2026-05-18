// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Bitonic argsort for lists of 65–256 elements — BATCHED edition.
//
// One call covers ALL lists; each block iterates over list indices
// via the blockIdx.x + gridDim.x stride loop, skipping any list
// whose length is outside (SMALL_N, MEDIUM_N].  Lists ≤ 64 are
// handled by argsort_small; lists > 256 by argsort_large.
//
//   grid  = (min(nlists, 65536), 1, 1)
//   block = (256, 1, 1)                 — one thread per MEDIUM_N slot
//   args  = (values, offsets, indices, total_size, nlists)
//
// LDS footprint: 256×4 (float) + 256×4 (int) = 2 KB per block.
//
// Output: for each list l with SMALL_N < len(l) ≤ MEDIUM_N,
//   indices[offsets[l] .. offsets[l+1]) holds the global positions into
//   `values` that produce an ascending sort of the list.

#include <hip/hip_runtime.h>
#include <float.h>

#define SMALL_N  64
#define MEDIUM_N 256

extern "C" __global__ void argsort_medium_hip(
    const float* __restrict__      values,
    const long long* __restrict__  offsets,
    long long* __restrict__        indices,
    long long                      total_size,
    long long                      nlists)
{
    __shared__ float s_val[MEDIUM_N];
    __shared__ int   s_idx[MEDIUM_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)nlists;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        // Skip lists handled by the small or large kernels.
        if (len <= SMALL_N || len > MEDIUM_N) continue;

        const int tid = (int)threadIdx.x;   // 0 .. 255

        // Load values; pad out-of-range slots with +inf so they sort last.
        s_val[tid] = (tid < len) ? values[start + tid] : FLT_MAX;
        s_idx[tid] = tid;
        __syncthreads();

        // Bitonic sort over MEDIUM_N = 256 elements.
        // Correctness: same argument as argsort_small — ixj > tid guard
        // prevents double-acting on any pair; __syncthreads() between
        // steps prevents data races.
        for (int k = 2; k <= MEDIUM_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const bool asc = ((tid & k) == 0);
                    if (( asc && s_val[tid] > s_val[ixj]) ||
                        (!asc && s_val[tid] < s_val[ixj])) {
                        float tv   = s_val[tid]; s_val[tid] = s_val[ixj]; s_val[ixj] = tv;
                        int   ti   = s_idx[tid]; s_idx[tid] = s_idx[ixj]; s_idx[ixj] = ti;
                    }
                }
                __syncthreads();
            }
        }

        // Write global indices for valid (non-padded) elements only.
        if (tid < len) {
            indices[start + tid] = (long long)(start + s_idx[tid]);
        }
        // Sync before the next list overwrites shared memory.
        __syncthreads();
    }
}
