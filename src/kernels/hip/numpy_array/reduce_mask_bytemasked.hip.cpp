// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Build a ByteMaskedArray mask from parent indices.
//
// Two-phase launch:
//   Phase 1: initialise toptr[0..ngroups] = 1  (masked / null)
//   Phase 2: for each entry in parents[], set toptr[parents[j]] = 0  (valid)
//
// Groups absent from parents remain 1 (null); groups present become 0 (valid).
// Multiple parents mapping to the same group all write 0 — benign race.
//
// Corresponds to CPU: numpy_array_reduce_mask_byte_masked_array_64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// Phase 1: fill with 1.
__global__ void reduce_mask_fill_kernel(
    signed char* __restrict__ toptr,
    long long                  ngroups
) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ngroups) {
        toptr[i] = 1;
    }
}

// Phase 2: scatter 0 for each present parent.
// Multiple threads may write 0 to the same location; since all writes are
// identical (0) the result is deterministic without atomics.
__global__ void reduce_mask_scatter_kernel(
    signed char* __restrict__       toptr,
    const long long* __restrict__   parents,
    long long                        nparents
) {
    long long j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nparents) {
        toptr[parents[j]] = 0;
    }
}

extern "C" void awkward_hip_reduce_mask_bytemasked(
    signed char*     toptr,
    const long long* parents,
    long long        ngroups,
    long long        nparents,
    hipStream_t      stream
) {
    int threads = 256;

    // Phase 1
    int blocks1 = (ngroups + threads - 1) / threads;
    hipLaunchKernelGGL(
        reduce_mask_fill_kernel,
        dim3(blocks1), dim3(threads), 0, stream,
        toptr, ngroups
    );
    HIP_CHECK(hipGetLastError());

    // Phase 2
    if (nparents > 0) {
        int blocks2 = (nparents + threads - 1) / threads;
        hipLaunchKernelGGL(
            reduce_mask_scatter_kernel,
            dim3(blocks2), dim3(threads), 0, stream,
            toptr, parents, nparents
        );
        HIP_CHECK(hipGetLastError());
    }
}
