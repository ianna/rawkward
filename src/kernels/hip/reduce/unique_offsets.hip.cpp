// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Segmented unique-count kernel ("unique offsets").
//
// Precondition: data within each segment is already sorted in ascending order
// (e.g. as produced by awkward_hip_segmented_sort).
//
// For each segment, counts the number of distinct values (i.e. the number of
// runs of consecutive equal elements) and writes that count to out[seg]:
//
//   segment: [1, 1, 2, 3, 3]  →  out[seg] = 3
//   segment: []               →  out[seg] = 0
//   segment: [7]              →  out[seg] = 1
//
// The caller can prefix-sum out[] to obtain a proper offsets array for a
// compact unique-values buffer.
//
// One thread per segment; output has one long long per segment.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// ----------------------------------------
// Templated kernel
// ----------------------------------------
template <typename T>
__global__ void segmented_unique_offsets_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    long long start = offsets[seg];
    long long end   = offsets[seg + 1];

    if (start >= end) {
        out[seg] = 0;
        return;
    }

    long long count = 1;
    for (long long i = start + 1; i < end; i++) {
        if (data[i] != data[i - 1]) {
            count++;
        }
    }

    out[seg] = count;
}

// ----------------------------------------
// Launcher (templated)
// ----------------------------------------
template <typename T>
void launch_segmented_unique_offsets(
    const T* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks  = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_unique_offsets_kernel<T>,
        dim3(blocks), dim3(threads), 0, stream,
        data, offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
extern "C" void awkward_hip_segmented_unique_offsets(
    const void* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_unique_offsets<float>(
                    static_cast<const float*>(data),
                    offsets, out, n_segments, stream); break;
        case 1: launch_segmented_unique_offsets<double>(
                    static_cast<const double*>(data),
                    offsets, out, n_segments, stream); break;
        case 2: launch_segmented_unique_offsets<int>(
                    static_cast<const int*>(data),
                    offsets, out, n_segments, stream); break;
        case 3: launch_segmented_unique_offsets<long long>(
                    static_cast<const long long*>(data),
                    offsets, out, n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
