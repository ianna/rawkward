// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Bitonic argsort for lists of 65–256 elements, using LDS (shared memory).
//
// Launch: one call per list.
//   grid  = (1, 1, 1)
//   block = (256, 1, 1)
//
// LDS footprint: 256×4 (float) + 256×4 (int) = 2 KB per block.

#include <hip/hip_runtime.h>
#include <float.h>

#define MEDIUM_N 256

extern "C" __global__ void argsort_medium_hip(
    const float* __restrict__      values,
    const long long* __restrict__  offsets,
    long long* __restrict__        indices,
    long long                      list_id)
{
    const int tid   = (int)threadIdx.x;   // 0 .. 255
    const int start = (int)offsets[list_id];
    const int end   = (int)offsets[list_id + 1];
    const int len   = end - start;

    __shared__ float s_val[MEDIUM_N];
    __shared__ int   s_idx[MEDIUM_N];

    s_val[tid] = (tid < len) ? values[start + tid] : FLT_MAX;
    s_idx[tid] = tid;
    __syncthreads();

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

    if (tid < len) {
        indices[start + tid] = (long long)(start + s_idx[tid]);
    }
}
