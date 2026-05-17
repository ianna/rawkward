// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Bitonic argsort for lists of ≤ 64 elements.
//
// Launch: one call per list.
//   grid  = (1, 1, 1)
//   block = (64, 1, 1)
//
// Kernel signature (extern "C" suppresses C++ name mangling so
// hipModuleGetFunction("argsort_small_hip") finds it by exact name):
//
//   argsort_small_hip(values, offsets, indices, list_id)
//
// Output: indices[offsets[list_id] .. offsets[list_id+1]) holds the
// global positions into `values` that produce an ascending sort of
// the list, matching the CPU argsort reference.

#include <hip/hip_runtime.h>
#include <float.h>

#define SMALL_N 64

extern "C" __global__ void argsort_small_hip(
    const float* __restrict__      values,
    const long long* __restrict__  offsets,
    long long* __restrict__        indices,
    long long                      list_id)
{
    const int tid   = (int)threadIdx.x;   // 0 .. 63
    const int start = (int)offsets[list_id];
    const int end   = (int)offsets[list_id + 1];
    const int len   = end - start;

    __shared__ float s_val[SMALL_N];
    __shared__ int   s_idx[SMALL_N];

    // Load values; pad out-of-range threads with +inf so they sort to the end.
    s_val[tid] = (tid < len) ? values[start + tid] : FLT_MAX;
    s_idx[tid] = tid;
    __syncthreads();

    // Standard bitonic sort (ascending, power-of-2 size = 64).
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

    // Write global indices for the valid (non-padded) elements only.
    if (tid < len) {
        indices[start + tid] = (long long)(start + s_idx[tid]);
    }
}
