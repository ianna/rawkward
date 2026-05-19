// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Build the carry index for the local-reduction preparation step.
//
// For each i in 0..parentslength:
//   if j < nextlen && parents[i] == nextparents[j]:
//     tocarry[i] = j, j++
//   else:
//     tocarry[i] = -1
//
// Single-threaded (grid=1, block=1) — j advances only when a match is found,
// creating a strict data dependency that forbids parallel execution.
//
// Corresponds to CPU: indexed_local_preparenext
//                     (indexed_array_local_preparenext_64).

#include <hip/hip_runtime.h>

__global__ void indexed_local_preparenext_kernel(
    long long*         tocarry,
    const long long*   parents,
    long long          parentslength,
    const long long*   nextparents,
    long long          nextlen
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long j = 0;
    for (long long i = 0; i < parentslength; i++) {
        if (j < nextlen && parents[i] == nextparents[j]) {
            tocarry[i] = j;
            j++;
        } else {
            tocarry[i] = -1;
        }
    }
}

extern "C" void awkward_hip_indexed_array_local_preparenext(
    long long*         tocarry,
    const long long*   parents,
    long long          parentslength,
    const long long*   nextparents,
    long long          nextlen,
    hipStream_t        stream
) {
    if (parentslength <= 0) return;
    hipLaunchKernelGGL(
        indexed_local_preparenext_kernel,
        dim3(1), dim3(1), 0, stream,
        tocarry, parents, parentslength, nextparents, nextlen
    );
}
