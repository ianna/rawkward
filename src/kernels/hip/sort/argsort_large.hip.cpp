// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Bitonic argsort for lists of > 256 elements (up to MAX_LARGE_N = 4096).
//
// Launch: one call covers ALL lists; the kernel skips lists ≤ 256 elements
// since those are handled by the small/medium kernels.
//   grid  = (nblocks, 1, 1)   — caller chooses, typically 256
//   block = (256, 1, 1)
//   args  = (values, offsets, indices, total_size, nlists)
//
// Each block iterates over list indices blockIdx.x, blockIdx.x + gridDim.x, …
// so the grid size can be smaller than nlists.
//
// LDS footprint: 4096×4 (float) + 4096×4 (int) = 32 KB per block.
// This fits within the 64 KB LDS available on CDNA2 (gfx90a).
//
// Lists longer than MAX_LARGE_N are silently skipped.
// TODO: replace with rocPRIM segmented_radix_sort for lists > MAX_LARGE_N.

#include <hip/hip_runtime.h>
#include <float.h>

#define MAX_LARGE_N 4096

extern "C" __global__ void argsort_large_hip(
    const float* __restrict__      values,
    const long long* __restrict__  offsets,
    long long* __restrict__        indices,
    long long                      total_size,
    long long                      nlists)
{
    __shared__ float s_val[MAX_LARGE_N];
    __shared__ int   s_idx[MAX_LARGE_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)nlists;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        // Lists ≤ 256 are handled by the small/medium kernels.
        if (len <= 256) continue;

        // Round length up to the next power of two; skip if too large.
        int n = 1;
        while (n < len) n <<= 1;
        if (n > MAX_LARGE_N) continue;  // TODO: rocPRIM fallback

        // Load: each thread loads len/blockDim.x elements (strided).
        // Padding slots receive +inf so they sort to the end.
        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
            s_val[i] = (i < len) ? values[start + i] : FLT_MAX;
            s_idx[i] = i;
        }
        __syncthreads();

        // Bitonic sort over n elements with blockDim.x threads.
        //
        // Correctness argument: for any step j, each pair (i, i^j) with
        // i < i^j is owned by exactly one thread (the one whose stride
        // iteration covers i). The ixj > i guard prevents both threads
        // in a pair from acting; the __syncthreads() after each j step
        // serialises stages so there are no data races.
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool asc = ((i & k) == 0);
                        if (( asc && s_val[i] > s_val[ixj]) ||
                            (!asc && s_val[i] < s_val[ixj])) {
                            float tv   = s_val[i]; s_val[i] = s_val[ixj]; s_val[ixj] = tv;
                            int   ti   = s_idx[i]; s_idx[i] = s_idx[ixj]; s_idx[ixj] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Write back only the valid (non-padded) results.
        for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) {
            indices[start + i] = (long long)(start + s_idx[i]);
        }
        // Sync before the next list overwrites shared memory.
        __syncthreads();
    }
}
