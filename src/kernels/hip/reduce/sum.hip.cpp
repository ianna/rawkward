// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

// ── Segmented reduce-sum kernels ─────────────────────────────────────────────
//
// One thread per segment.  `blockIdx.x * blockDim.x + threadIdx.x` is the
// segment index; the thread sums data[offsets[seg]..offsets[seg+1]] into
// out[seg].  Out-of-range threads return immediately.
//
// Launched via the GpuBackend abstraction (hipModuleLaunchKernel) with
// blocks = ceil(n_segments / 256), threads = 256.
//
// Named with `extern "C"` to suppress C++ name mangling; kernel names are
// looked up at runtime via `hipModuleGetFunction`.
//
// Note: double (f64) IS supported on AMD GCN and RDNA compute shaders,
// unlike Metal which forbids double entirely.

template <typename T>
__device__ inline void segmented_sum_body(
    const T*          data,
    const long long*  offsets,
    T*                out,
    long long         n_segments,
    long long         seg
) {
    if (seg >= n_segments) return;
    long long start = offsets[seg];
    long long end   = offsets[seg + 1];
    T acc = T(0);
    for (long long i = start; i < end; ++i) {
        acc += data[i];
    }
    out[seg] = acc;
}

extern "C" __global__ void segmented_sum_f32(
    const float*     data,
    const long long* offsets,
    float*           out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_sum_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_sum_f64(
    const double*    data,
    const long long* offsets,
    double*          out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_sum_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_sum_i32(
    const int*       data,
    const long long* offsets,
    int*             out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_sum_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_sum_i64(
    const long long* data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_sum_body(data, offsets, out, n_segments, seg);
}
