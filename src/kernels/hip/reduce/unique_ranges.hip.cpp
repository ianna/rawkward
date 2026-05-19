// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Segmented run-index kernel ("unique ranges").
//
// Precondition: data within each segment is already sorted in ascending order
// (e.g. as produced by awkward_hip_segmented_sort).
//
// For every position i in the flat data array, this kernel writes the 0-based
// index of the run (sequence of consecutive equal values) that i belongs to
// within its segment:
//
//   sorted segment: [1, 1, 2, 3, 3]
//   out:            [0, 0, 1, 2, 2]
//
// Combined with unique_offsets (which counts the runs per segment) this
// allows callers to scatter each position to its unique-value slot:
//   unique_slot = unique_offsets_prefix_sum[seg] + unique_ranges[i]
//
// One thread per segment; output has the same total size as the input data.

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
__global__ void segmented_unique_ranges_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    long long n_segments
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    long long start = offsets[seg];
    long long end   = offsets[seg + 1];

    if (start >= end) return;

    long long run_idx = 0;
    out[start] = 0;

    for (long long i = start + 1; i < end; i++) {
        if (data[i] != data[i - 1]) {
            run_idx++;
        }
        out[i] = run_idx;
    }
}

// ----------------------------------------
// Launcher (templated)
// ----------------------------------------
template <typename T>
void launch_segmented_unique_ranges(
    const T* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    hipStream_t stream
) {
    int threads = 256;
    int blocks  = (n_segments + threads - 1) / threads;

    hipLaunchKernelGGL(
        segmented_unique_ranges_kernel<T>,
        dim3(blocks), dim3(threads), 0, stream,
        data, offsets, out, n_segments
    );

    HIP_CHECK(hipGetLastError());
}

// ----------------------------------------
// C API entry point (for Rust / Python FFI)
// ----------------------------------------
extern "C" void awkward_hip_segmented_unique_ranges(
    const void* data,
    const long long* offsets,
    long long* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_unique_ranges<float>(
                    static_cast<const float*>(data),
                    offsets, out, n_segments, stream); break;
        case 1: launch_segmented_unique_ranges<double>(
                    static_cast<const double*>(data),
                    offsets, out, n_segments, stream); break;
        case 2: launch_segmented_unique_ranges<int>(
                    static_cast<const int*>(data),
                    offsets, out, n_segments, stream); break;
        case 3: launch_segmented_unique_ranges<long long>(
                    static_cast<const long long*>(data),
                    offsets, out, n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
