// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Replace -1 (None) entries in a device index array with unique sequential
// values that extend beyond the existing non-null entries.
//
// Algorithm (two-pass, single-threaded for correct ordering):
//   Pass 1: count n_non_null = number of entries != -1.
//   Pass 2: scan left-to-right; each -1 becomes n_non_null, n_non_null+1, ...
//
// Single-threaded kernel (grid=1, block=1) — pass 2 requires deterministic
// left-to-right assignment.
//
// Corresponds to CPU: index_nones_as_index (nones_as_index).

#include <hip/hip_runtime.h>

__global__ void index_nones_as_index_kernel(
    long long* toindex,
    long long  length
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Pass 1: count non-null entries.
    long long n_non_null = 0;
    for (long long i = 0; i < length; i++) {
        if (toindex[i] != -1) n_non_null++;
    }

    // Pass 2: replace each -1 with the next sequential value.
    long long next = n_non_null;
    for (long long i = 0; i < length; i++) {
        if (toindex[i] == -1) {
            toindex[i] = next++;
        }
    }
}

extern "C" void awkward_hip_index_nones_as_index(
    long long*  toindex,    // device array, modified in place
    long long   length,
    hipStream_t stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        index_nones_as_index_kernel,
        dim3(1), dim3(1), 0, stream,
        toindex, length
    );
}
