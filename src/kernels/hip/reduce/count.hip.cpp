// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// ----------------------------------------
// Segmented count kernel
// ----------------------------------------
// out[seg] = offsets[seg+1] - offsets[seg]  (number of elements in segment).
// The data pointer is accepted for API uniformity but is never read.
__global__ void segmented_count_kernel(
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    out[seg] = offsets[seg + 1] - offsets[seg];
}

// ----------------------------------------
// Launcher
// ----------------------------------------
static void launch_segmented_count(
    const long long* offsets,
    long long* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks  = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_count_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
// dtype_code is ignored — the count is always long long regardless of data type.
extern "C" void awkward_hip_segmented_count(
    const void* /*data*/,
    const long long* offsets,
    long long* out,
    long long n_segments,
    int /*dtype_code*/,
    hipStream_t stream
) {
    launch_segmented_count(offsets, out, n_segments, stream);
}
