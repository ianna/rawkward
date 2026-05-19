// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Adjust argmin/argmax results from global flat indices to within-list offsets.
//
// For each output group i:
//   if toptr[i] >= 0:
//     parent = parents[toptr[i]]
//     toptr[i] -= starts[parent]
//
// This converts a global flat index (as returned by argmin/argmax) into a
// within-list local position.  Empty-group sentinels (-1) are left unchanged.
//
// One thread per element of toptr.
//
// Corresponds to CPU: numpy_array_reduce_adjust_starts_64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

__global__ void reduce_adjust_starts_kernel(
    long long* __restrict__       toptr,
    const long long* __restrict__ parents,
    const long long* __restrict__ starts,
    long long                      length
) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;

    long long v = toptr[i];
    if (v >= 0) {
        long long parent = parents[v];
        toptr[i] = v - starts[parent];
    }
}

extern "C" void awkward_hip_reduce_adjust_starts(
    long long*       toptr,
    const long long* parents,
    const long long* starts,
    long long        length,
    hipStream_t      stream
) {
    if (length == 0) return;

    int threads = 256;
    int blocks  = (length + threads - 1) / threads;

    hipLaunchKernelGGL(
        reduce_adjust_starts_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        toptr, parents, starts, length
    );

    HIP_CHECK(hipGetLastError());
}
