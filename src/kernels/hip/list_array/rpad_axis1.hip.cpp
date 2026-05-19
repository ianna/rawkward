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
// dtype_code: 0=i32, 1=u32, 2=i64  — applies to fromstarts/fromstops/tostarts/tostops.
// toindex is always i64.
//
// Corresponds to CPU: list_array_rpad_axis1 (generic over C).

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_rpad_axis1_kernel(
    long long* toindex,
    const C*   fromstarts,
    const C*   fromstops,
    C*         tostarts,
    C*         tostops,
    long long  target,
    long long  length
) {
    long long offset = 0;
    for (long long i = 0; i < length; ++i) {
        tostarts[i] = (C)offset;
        long long start    = (long long)fromstarts[i];
        long long rangeval = (long long)fromstops[i] - start;
        for (long long j = 0; j < rangeval; ++j) {
            toindex[offset + j] = start + j;
        }
        for (long long j = rangeval; j < target; ++j) {
            toindex[offset + j] = -1LL;
        }
        offset += (target > rangeval) ? target : rangeval;
        tostops[i] = (C)offset;
    }
}

template <typename C>
static void launch_rpad_axis1(
    long long*  toindex,
    const void* fromstarts,
    const void* fromstops,
    void*       tostarts,
    void*       tostops,
    long long   target,
    long long   length,
    hipStream_t stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        list_array_rpad_axis1_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        toindex,
        (const C*)fromstarts, (const C*)fromstops,
        (C*)tostarts, (C*)tostops,
        target, length
    );
}

extern "C" void awkward_hip_list_array_rpad_axis1(
    long long*  toindex,
    const void* fromstarts,
    const void* fromstops,
    void*       tostarts,
    void*       tostops,
    long long   target,
    long long   length,
    int         dtype_code,  // 0=i32, 1=u32, 2=i64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_rpad_axis1<int>         (toindex, fromstarts, fromstops, tostarts, tostops, target, length, stream); break;
        case 1:  launch_rpad_axis1<unsigned int>(toindex, fromstarts, fromstops, tostarts, tostops, target, length, stream); break;
        default: launch_rpad_axis1<long long>   (toindex, fromstarts, fromstops, tostarts, tostops, target, length, stream); break;
    }
}
