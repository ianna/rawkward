// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Right-pad each list to a target length, building an index + new starts/stops.
//
// awkward_hip_list_array_rpad_axis1:
//   For each list i:
//     tostarts[i] = offset
//     for j in 0..rangeval: toindex[offset+j] = fromstarts[i] + j   (copy)
//     for j in rangeval..target: toindex[offset+j] = -1              (pad)
//     offset += max(target, rangeval)
//     tostops[i] = offset
//
// Single-threaded kernel (sequential offset accumulation).
// All arrays are i64.  dtype_code encodes starts/stops type (0=i32,1=u32,2=i64)
// for output tostarts/tostops (toindex is always i64; we write i64 starts/stops).
//
// Corresponds to CPU: list_array_rpad_axis1.

#include <hip/hip_runtime.h>

__global__ void list_array_rpad_axis1_kernel(
    long long*       toindex,
    const long long* fromstarts,
    const long long* fromstops,
    long long*       tostarts,
    long long*       tostops,
    long long        target,
    long long        length
) {
    long long offset = 0;
    for (long long i = 0; i < length; ++i) {
        tostarts[i] = offset;
        long long start    = fromstarts[i];
        long long rangeval = fromstops[i] - start;
        for (long long j = 0; j < rangeval; ++j) {
            toindex[offset + j] = start + j;
        }
        for (long long j = rangeval; j < target; ++j) {
            toindex[offset + j] = -1LL;
        }
        offset += (target > rangeval) ? target : rangeval;
        tostops[i] = offset;
    }
}

extern "C" void awkward_hip_list_array_rpad_axis1(
    long long*       toindex,
    const long long* fromstarts,
    const long long* fromstops,
    long long*       tostarts,
    long long*       tostops,
    long long        target,
    long long        length,
    hipStream_t      stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        list_array_rpad_axis1_kernel,
        dim3(1), dim3(1), 0, stream,
        toindex, fromstarts, fromstops, tostarts, tostops, target, length
    );
}
