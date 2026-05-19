// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Carry kernel: single-integer index into each list.
//
// awkward_hip_list_array_getitem_next_at:
//   For each list i, resolve the integer index `at` (negative wrap-around)
//   and write the absolute content position into tocarry[i].
//
//   *out_error codes (device int, caller zero-inits):
//     1 – index out of range
//
// One GPU thread per list.  dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_getitem_next_at.

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_getitem_next_at_kernel(
    long long* tocarry,
    const C*   fromstarts,
    const C*   fromstops,
    long long  at,
    long long  length,
    int*       out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long start  = (long long)fromstarts[i];
    long long stop   = (long long)fromstops [i];
    long long len    = stop - start;
    long long regular_at = at;
    if (regular_at < 0) regular_at += len;
    if (regular_at < 0 || regular_at >= len) {
        atomicCAS(out_error, 0, 1);
        return;
    }
    tocarry[i] = start + regular_at;
}

template <typename C>
static void launch_list_array_getitem_next_at(
    long long*  tocarry,
    const void* fromstarts,
    const void* fromstops,
    long long   at,
    long long   length,
    int*        out_error,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_at_kernel<C>,
        grid, block, 0, stream,
        tocarry, (const C*)fromstarts, (const C*)fromstops, at, length, out_error
    );
}

extern "C" void awkward_hip_list_array_getitem_next_at(
    long long*  tocarry,
    const void* fromstarts,
    const void* fromstops,
    long long   at,
    long long   length,
    int         dtype_code,   // 0=i32, 1=u32, 2=i64
    int*        out_error,    // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_getitem_next_at<int>         (tocarry, fromstarts, fromstops, at, length, out_error, stream); break;
        case 1:  launch_list_array_getitem_next_at<unsigned int>(tocarry, fromstarts, fromstops, at, length, out_error, stream); break;
        default: launch_list_array_getitem_next_at<long long>   (tocarry, fromstarts, fromstops, at, length, out_error, stream); break;
    }
}
