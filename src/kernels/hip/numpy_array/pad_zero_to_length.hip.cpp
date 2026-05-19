// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Pad sublists to a fixed length, zero-filling any gaps.
//
// For each sublist k defined by fromoffsets[k..k+1]:
//   - Copy fromptr[fromoffsets[k] .. fromoffsets[k+1]] to toptr[k*target ..]
//   - Zero-fill toptr[k*target + count .. k*target + target]
//
// One thread per sublist.  toptr must have n_lists * target elements.
//
// Corresponds to CPU: numpy_array_pad_zero_to_length_uint8_int64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

__global__ void pad_zero_to_length_kernel(
    const unsigned char* __restrict__ fromptr,
    const long long* __restrict__     fromoffsets,
    long long                          target,
    unsigned char* __restrict__       toptr,
    long long                          n_lists
) {
    long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_lists) return;

    long long start = fromoffsets[k];
    long long end   = fromoffsets[k + 1];
    long long count = end - start;
    long long dest  = k * target;

    // Copy content.
    for (long long i = 0; i < count; i++) {
        toptr[dest + i] = fromptr[start + i];
    }
    // Zero-fill the remainder.
    for (long long i = count; i < target; i++) {
        toptr[dest + i] = 0;
    }
}

extern "C" void awkward_hip_pad_zero_to_length(
    const void*      fromptr,
    const long long* fromoffsets,
    long long        target,
    void*            toptr,
    long long        n_lists,
    hipStream_t      stream
) {
    if (n_lists == 0) return;

    int threads = 256;
    int blocks  = (n_lists + threads - 1) / threads;

    hipLaunchKernelGGL(
        pad_zero_to_length_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const unsigned char*>(fromptr),
        fromoffsets,
        target,
        static_cast<unsigned char*>(toptr),
        n_lists
    );

    HIP_CHECK(hipGetLastError());
}
