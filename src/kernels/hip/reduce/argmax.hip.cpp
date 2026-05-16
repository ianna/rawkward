// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include <limits>

// ----------------------------------------
// Utility: HIP error checking
// ----------------------------------------
#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// ----------------------------------------
// Templated segmented argmax kernel
// ----------------------------------------
template <typename T>
__global__ void segmented_argmax_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    long long start = offsets[seg];
    long long end   = offsets[seg + 1];

    // Empty segment → -1
    if (end <= start) {
        out[seg] = -1;
        return;
    }

    T best_val = data[start];
    long long best_idx = start;

    // Linear scan
    for (long long i = start + 1; i < end; i++) {
        T v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    out[seg] = best_idx;
}

// ----------------------------------------
// Launcher (templated)
// ----------------------------------------
template <typename T>
void launch_segmented_argmax(
    const T* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_argmax_kernel<T>,
        dim3(blocks), dim3(threads), 0, stream,
        data, offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
extern "C" void awkward_hip_segmented_argmax(
    const void* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_argmax<float>(
                    static_cast<const float*>(data),
                    offsets, out, n_segments, stream); break;
        case 1: launch_segmented_argmax<double>(
                    static_cast<const double*>(data),
                    offsets, out, n_segments, stream); break;
        case 2: launch_segmented_argmax<int>(
                    static_cast<const int*>(data),
                    offsets, out, n_segments, stream); break;
        case 3: launch_segmented_argmax<long long>(
                    static_cast<const long long*>(data),
                    offsets, out, n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}

