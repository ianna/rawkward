// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Convert starts/stops to a compact offsets array.
//
// awkward_hip_list_array_compact_offsets:
//   tooffsets[0] = 0
//   tooffsets[i+1] = tooffsets[i] + (stops[i] - starts[i])
//   If stops[i] < starts[i] → *out_error = 1 (stops before starts).
//
// Single-threaded kernel (sequential prefix-sum dependencies).
// dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_compact_offsets.

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_compact_offsets_kernel(
    long long*  tooffsets,
    const C*    fromstarts,
    const C*    fromstops,
    long long   length,
    int*        out_error
) {
    tooffsets[0] = 0LL;
    long long acc = 0;
    for (long long i = 0; i < length; ++i) {
        long long start = (long long)fromstarts[i];
        long long stop  = (long long)fromstops [i];
        if (stop < start) {
            *out_error = 1;
            return;
        }
        acc += stop - start;
        tooffsets[i + 1] = acc;
    }
}

template <typename C>
static void launch_list_array_compact_offsets(
    long long*  tooffsets,
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    int*        out_error,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        list_array_compact_offsets_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tooffsets, (const C*)fromstarts, (const C*)fromstops, length, out_error
    );
}

extern "C" void awkward_hip_list_array_compact_offsets(
    long long*  tooffsets,
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    int         dtype_code,   // 0=i32, 1=u32, 2=i64
    int*        out_error,    // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_compact_offsets<int>         (tooffsets, fromstarts, fromstops, length, out_error, stream); break;
        case 1:  launch_list_array_compact_offsets<unsigned int>(tooffsets, fromstarts, fromstops, length, out_error, stream); break;
        default: launch_list_array_compact_offsets<long long>   (tooffsets, fromstarts, fromstops, length, out_error, stream); break;
    }
}
