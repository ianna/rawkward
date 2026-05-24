// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include <limits>

// ── Segmented max kernels ─────────────────────────────────────────────────────
//
// out[seg] = max element in segment.  Empty segments return numeric_limits::lowest().

template <typename T>
__device__ inline void segmented_max_body(
    const T*         data,
    const long long* offsets,
    T*               out,
    long long        n_segments,
    long long        seg
) {
    if (seg >= n_segments) return;
    long long start = offsets[seg];
    long long end   = offsets[seg + 1];
    if (end <= start) { out[seg] = std::numeric_limits<T>::lowest(); return; }
    T best = data[start];
    for (long long i = start + 1; i < end; ++i) {
        T v = data[i];
        if (v > best) best = v;
    }
    out[seg] = best;
}

extern "C" __global__ void segmented_max_f32(
    const float*     data,
    const long long* offsets,
    float*           out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_max_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_max_f64(
    const double*    data,
    const long long* offsets,
    double*          out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_max_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_max_i32(
    const int*       data,
    const long long* offsets,
    int*             out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_max_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_max_i64(
    const long long* data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_max_body(data, offsets, out, n_segments, seg);
}
