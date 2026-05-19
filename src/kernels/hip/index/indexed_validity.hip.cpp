// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Validate the index array of an IndexedArray or IndexedOptionArray.
//
// isoption=0: every entry must be >= 0.
// isoption=1: -1 is allowed; all non-negative entries must be < lencontent.
//
// Writes to *out_error (device int, caller must zero-init):
//   0  – all entries valid
//   1  – negative index in a non-option array
//   2  – index >= lencontent
//
// Uses atomicCAS so only the first violation is recorded.
// One GPU thread per element.
//
// Corresponds to CPU: indexed_validity (indexed_array_validity).

#include <hip/hip_runtime.h>

__global__ void indexed_array_validity_kernel(
    const long long* index,
    long long        length,
    long long        lencontent,
    int              isoption,
    int*             out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;

    long long v = index[i];

    if (!isoption && v < 0) {
        atomicCAS(out_error, 0, 1);
        return;
    }
    if (v >= lencontent) {
        atomicCAS(out_error, 0, 2);
    }
}

extern "C" void awkward_hip_indexed_array_validity(
    const long long* index,
    long long        length,
    long long        lencontent,
    int              isoption,
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_array_validity_kernel,
        grid, block, 0, stream,
        index, length, lencontent, isoption, out_error
    );
}
