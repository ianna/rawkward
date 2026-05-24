// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

// ── Segmented argmax kernels ──────────────────────────────────────────────────
//
// One thread per segment.  Returns the absolute index (into `data`) of the
// maximum element in each segment.  Empty segments write -1.
//
// Named with `extern "C"` to suppress C++ name mangling; looked up at runtime
// via `hipModuleGetFunction`.  Launched via the GpuBackend abstraction.

template <typename T>
__device__ inline void segmented_argmax_body(
    const T*         data,
    const long long* offsets,
    long long*       out,
    long long        n_segments,
    long long        seg
) {
    if (seg >= n_segments) return;
    long long start = offsets[seg];
    long long end   = offsets[seg + 1];
    if (end <= start) { out[seg] = -1; return; }
    T best_val = data[start];
    long long best_idx = start;
    for (long long i = start + 1; i < end; ++i) {
        T v = data[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    out[seg] = best_idx;
}

extern "C" __global__ void segmented_argmax_f32(
    const float*     data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_argmax_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_argmax_f64(
    const double*    data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_argmax_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_argmax_i32(
    const int*       data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_argmax_body(data, offsets, out, n_segments, seg);
}

extern "C" __global__ void segmented_argmax_i64(
    const long long* data,
    const long long* offsets,
    long long*       out,
    long long        n_segments
) {
    long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    segmented_argmax_body(data, offsets, out, n_segments, seg);
}
