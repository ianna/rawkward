// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// ----------------------------------------
// Templated segmented count-nonzero kernel
// ----------------------------------------
// out[seg] = number of elements in the segment that compare != T(0).
template <typename T>
__global__ void segmented_countnonzero_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    long long start = offsets[seg];
    long long end   = offsets[seg + 1];

    long long count = 0;
    for (long long i = start; i < end; i++) {
        if (data[i] != T(0)) {
            count++;
        }
    }

    out[seg] = count;
}

// ----------------------------------------
// Launcher (templated)
// ----------------------------------------
template <typename T>
void launch_segmented_countnonzero(
    const T* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks  = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_countnonzero_kernel<T>,
        dim3(blocks), dim3(threads), 0, stream,
        data, offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
extern "C" void awkward_hip_segmented_countnonzero(
    const void* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_countnonzero<float>(
                    static_cast<const float*>(data),
                    offsets, out, n_segments, stream); break;
        case 1: launch_segmented_countnonzero<double>(
                    static_cast<const double*>(data),
                    offsets, out, n_segments, stream); break;
        case 2: launch_segmented_countnonzero<int>(
                    static_cast<const int*>(data),
                    offsets, out, n_segments, stream); break;
        case 3: launch_segmented_countnonzero<long long>(
                    static_cast<const long long*>(data),
                    offsets, out, n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
