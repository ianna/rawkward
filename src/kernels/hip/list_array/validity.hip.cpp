// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Validate the starts/stops arrays of a ListArray.
//
// For each list i where start != stop:
//   1. start > stop  → out_error = 1
//   2. start < 0     → out_error = 2
//   3. stop > lencontent → out_error = 3
//
// Empty lists (start == stop) are always valid.
// Uses atomicCAS so only the first violation is recorded.
// One GPU thread per list.  dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_validity.

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_validity_kernel(
    const C*  fromstarts,
    const C*  fromstops,
    long long length,
    long long lencontent,
    int*      out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long start = (long long)fromstarts[i];
    long long stop  = (long long)fromstops [i];
    if (start == stop) return;
    if (start > stop) {
        atomicCAS(out_error, 0, 1);
        return;
    }
    if (start < 0) {
        atomicCAS(out_error, 0, 2);
        return;
    }
    if (stop > lencontent) {
        atomicCAS(out_error, 0, 3);
    }
}

template <typename C>
static void launch_list_array_validity(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   lencontent,
    int*        out_error,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_validity_kernel<C>,
        grid, block, 0, stream,
        (const C*)fromstarts, (const C*)fromstops, length, lencontent, out_error
    );
}

extern "C" void awkward_hip_list_array_validity(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   lencontent,
    int         dtype_code,   // 0=i32, 1=u32, 2=i64
    int*        out_error,    // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_validity<int>         (fromstarts, fromstops, length, lencontent, out_error, stream); break;
        case 1:  launch_list_array_validity<unsigned int>(fromstarts, fromstops, length, lencontent, out_error, stream); break;
        default: launch_list_array_validity<long long>   (fromstarts, fromstops, length, lencontent, out_error, stream); break;
    }
}
