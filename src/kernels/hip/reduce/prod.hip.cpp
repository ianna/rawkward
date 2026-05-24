// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

// ── Segmented product kernels ─────────────────────────────────────────────────
//
// out[seg] = product of elements in segment.  Empty segments write T(1).

template <typename T>
__device__ inline void segmented_prod_body(
    const T*         data,
    const long long* offsets,
    T*               out,
    long long        n_segments,
    long long        seg
) {
    if (seg >= n_segments) return;
    long long start = offsets[seg];
    long long end   = offsets[seg + 1];
    T acc = T(1);
    for (long long i = start; i < end; ++i) acc *= data[i];
    out[seg] = acc;
}

extern "C" __global__ void segmented_prod_f32(
    const float*     data,
    const long long* offsets,
    float*           out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_prod_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_prod_f64(
    const double*    data,
    const long long* offsets,
    double*          out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_prod_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_prod_i32(
    const int*       data,
    const long long* offsets,
    int*             out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_prod_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_prod_i64(
    const long long* data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_prod_body(data, offsets, out, n_segments, seg);
}
