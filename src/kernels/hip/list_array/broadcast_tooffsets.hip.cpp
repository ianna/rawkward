// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Broadcast a ListArray to a target offsets array, producing a carry index.
//
// awkward_hip_list_array_broadcast_tooffsets:
//   For each list i:
//     * Verify stop - start == fromoffsets[i+1] - fromoffsets[i].
//     * If non-empty, verify stop <= lencontent.
//     * Fill tocarry[k++] = start, start+1, ..., stop-1.
//
//   *out_error codes (device int, caller zero-inits):
//     1 – stop > lencontent
//     2 – fromoffsets not monotonically increasing
//     3 – length mismatch (cannot broadcast)
//
// Single-threaded kernel (sequential k).  dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_broadcast_tooffsets.

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_broadcast_tooffsets_kernel(
    long long*       tocarry,
    const long long* fromoffsets,
    const C*         fromstarts,
    const C*         fromstops,
    long long        nlists,
    long long        lencontent,
    int*             out_error
) {
    long long k = 0;
    for (long long i = 0; i < nlists; ++i) {
        long long start = (long long)fromstarts[i];
        long long stop  = (long long)fromstops [i];
        if (start != stop && stop > lencontent) {
            *out_error = 1;
            return;
        }
        long long count = fromoffsets[i + 1] - fromoffsets[i];
        if (count < 0) {
            *out_error = 2;
            return;
        }
        if (stop - start != count) {
            *out_error = 3;
            return;
        }
        for (long long j = start; j < stop; ++j) {
            tocarry[k++] = j;
        }
    }
}

template <typename C>
static void launch_list_array_broadcast_tooffsets(
    long long*       tocarry,
    const long long* fromoffsets,
    const void*      fromstarts,
    const void*      fromstops,
    long long        nlists,
    long long        lencontent,
    int*             out_error,
    hipStream_t      stream
) {
    hipLaunchKernelGGL(
        list_array_broadcast_tooffsets_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tocarry, fromoffsets,
        (const C*)fromstarts, (const C*)fromstops,
        nlists, lencontent, out_error
    );
}

extern "C" void awkward_hip_list_array_broadcast_tooffsets(
    long long*       tocarry,
    const long long* fromoffsets,
    const void*      fromstarts,
    const void*      fromstops,
    long long        nlists,
    long long        lencontent,
    int              dtype_code,   // 0=i32, 1=u32, 2=i64
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_broadcast_tooffsets<int>         (tocarry, fromoffsets, fromstarts, fromstops, nlists, lencontent, out_error, stream); break;
        case 1:  launch_list_array_broadcast_tooffsets<unsigned int>(tocarry, fromoffsets, fromstarts, fromstops, nlists, lencontent, out_error, stream); break;
        default: launch_list_array_broadcast_tooffsets<long long>   (tocarry, fromoffsets, fromstarts, fromstops, nlists, lencontent, out_error, stream); break;
    }
}
