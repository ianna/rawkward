// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Bitonic argsort for lists of 1–64 elements — BATCHED edition.
//
// One call covers ALL lists; each block iterates over list indices
// via the blockIdx.x + gridDim.x stride loop, skipping any list
// whose length is outside [1, SMALL_N].  Zero-length lists produce
// no output; lists > SMALL_N are handled by medium/large kernels.
//
//   grid  = (min(nlists, 65536), 1, 1)
//   block = (64, 1, 1)                  — one thread per SMALL_N slot
//   args  = (values, offsets, indices, total_size, nlists)
//
// LDS footprint: 64×4 (float) + 64×4 (int) = 512 B per block.
//
// Output: for each list l with 1 ≤ len(l) ≤ SMALL_N,
//   indices[offsets[l] .. offsets[l+1]) holds the global positions into
//   `values` that produce an ascending sort of the list.

#include <hip/hip_runtime.h>
#include <float.h>

#define SMALL_N 64

extern "C" __global__ void argsort_small_hip(
    const float* __restrict__      values,
    const long long* __restrict__  offsets,
    long long* __restrict__        indices,
    long long                      total_size,
    long long                      nlists)
{
    __shared__ float s_val[SMALL_N];
    __shared__ int   s_idx[SMALL_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)nlists;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        // Skip empty lists and lists belonging to the medium/large tiers.
        if (len == 0 || len > SMALL_N) continue;

        const int tid = (int)threadIdx.x;   // 0 .. 63

        // Load values; pad out-of-range slots with +inf so they sort last.
        s_val[tid] = (tid < len) ? values[start + tid] : FLT_MAX;
        s_idx[tid] = tid;
        __syncthreads();

        // Standard bitonic sort (ascending, power-of-2 size = SMALL_N).
        // Each thread owns element [tid]; ixj > tid guard ensures only one
        // thread acts on each pair.
        for (int k = 2; k <= SMALL_N; k <<= 1) {
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
