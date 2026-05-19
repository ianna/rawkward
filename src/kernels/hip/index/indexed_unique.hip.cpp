// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Build output index and offsets for the unique-values reduction step.
//
// For each group i defined by fromoffsets[i..i+1]:
//   For each element j in the group: toindex[k] = ll, k++, ll++.
//   After the group, if fromnulls[k] == 1:
//     toindex[k] = -1, k++, shift++.
//   tooffsets[i+1] = fromoffsets[i+1] + shift.
//
// tooffsets[0] = fromoffsets[0] (initialised before the loop).
// Single-threaded (sequential k, ll, shift counters required).
//
// Corresponds to CPU: indexed_unique_next_index_and_offsets
//                     (indexed_array_unique_next_index_and_offsets_64).

#include <hip/hip_runtime.h>

__global__ void indexed_unique_next_index_and_offsets_kernel(
    long long*         toindex,
    long long*         tooffsets,
    const long long*   fromoffsets,
    const long long*   fromnulls,
    long long          startslength
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    long long k     = 0;
    long long ll    = 0;
    long long shift = 0;

    toindex[0]   = ll;
    tooffsets[0] = fromoffsets[0];

    for (long long i = 0; i < startslength; i++) {
        long long seg_start = fromoffsets[i];
        long long seg_stop  = fromoffsets[i + 1];
        for (long long j = seg_start; j < seg_stop; j++) {
            toindex[k] = ll;
            k++;
            ll++;
        }
        if (fromnulls[k] == 1) {
            toindex[k] = -1;
            k++;
            shift++;
        }
        tooffsets[i + 1] = fromoffsets[i + 1] + shift;
    }
}

extern "C" void awkward_hip_indexed_array_unique_next_index_and_offsets(
    long long*         toindex,
    long long*         tooffsets,
    const long long*   fromoffsets,
    const long long*   fromnulls,
    long long          startslength,
    hipStream_t        stream
) {
    if (startslength <= 0) return;
    hipLaunchKernelGGL(
        indexed_unique_next_index_and_offsets_kernel,
        dim3(1), dim3(1), 0, stream,
        toindex, tooffsets, fromoffsets, fromnulls, startslength
    );
}
