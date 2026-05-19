// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Find the minimum list length (stops[i] - starts[i]) across all lists.
//
// awkward_hip_list_array_min_range:
//   Writes min(stops[i] - starts[i]) into *out_min (device pointer).
//   Caller must initialise *out_min to INT64_MAX before launch.
//
// Uses a single-threaded kernel to avoid the signed-64-bit atomicMin issue.
// dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_min_range.

#include <hip/hip_runtime.h>
#include <climits>

template <typename C>
__global__ void list_array_min_range_kernel(
    const C*   fromstarts,
    const C*   fromstops,
    long long  length,
    long long* out_min
) {
    // Single-threaded: grid=(1,1,1), block=(1,1,1).
    long long mn = *out_min;
    for (long long i = 0; i < length; ++i) {
        long long v = (long long)fromstops[i] - (long long)fromstarts[i];
        if (v < mn) mn = v;
    }
    *out_min = mn;
}

template <typename C>
static void launch_list_array_min_range(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long*  out_min,
    hipStream_t stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        list_array_min_range_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        (const C*)fromstarts, (const C*)fromstops, length, out_min
    );
}

extern "C" void awkward_hip_list_array_min_range(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    int         dtype_code,  // 0=i32, 1=u32, 2=i64
    long long*  out_min,     // device pointer, caller sets to INT64_MAX
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_min_range<int>         (fromstarts, fromstops, length, out_min, stream); break;
        case 1:  launch_list_array_min_range<unsigned int>(fromstarts, fromstops, length, out_min, stream); break;
        default: launch_list_array_min_range<long long>   (fromstarts, fromstops, length, out_min, stream); break;
    }
}
