// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// ----------------------------------------
// Templated segmented product kernel
// ----------------------------------------
// Each thread owns one segment; it multiplies all elements in the segment
// and writes the product to out[seg].  Empty segments return T(1)
// (the multiplicative identity).
template <typename T>
__global__ void segmented_prod_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    long long start = offsets[seg];
    long long end   = offsets[seg + 1];

    if (end <= start) {
        out[seg] = T(1);
        return;
    }

    T acc = T(1);
    for (long long i = start; i < end; i++) {
        acc *= data[i];
    }

    out[seg] = acc;
}

// ----------------------------------------
// Launcher (templated)
// ----------------------------------------
template <typename T>
void launch_segmented_prod(
    const T* data,
    const long long* offsets,
    T* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks  = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_prod_kernel<T>,
        dim3(blocks), dim3(threads), 0, stream,
        data, offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
extern "C" void awkward_hip_segmented_prod(
    const void* data,
    const long long* offsets,
    void* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_prod<float>(
                    static_cast<const float*>(data),
                    offsets,
                    static_cast<float*>(out),
                    n_segments, stream); break;
        case 1: launch_segmented_prod<double>(
                    static_cast<const double*>(data),
                    offsets,
                    static_cast<double*>(out),
                    n_segments, stream); break;
        case 2: launch_segmented_prod<int>(
                    static_cast<const int*>(data),
                    offsets,
                    static_cast<int*>(out),
                    n_segments, stream); break;
        case 3: launch_segmented_prod<long long>(
                    static_cast<const long long*>(data),
                    offsets,
                    static_cast<long long*>(out),
                    n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
