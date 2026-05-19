// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Compute the total output length for rpad-and-clip on axis=1.
//
// awkward_hip_list_array_rpad_and_clip_length_axis1:
//   Accumulates sum of max(target, stops[i] - starts[i]) into *out_total.
//   Caller must zero-init *out_total before launch.
//
// One GPU thread per list; atomicAdd to accumulate.
// dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_rpad_and_clip_length_axis1.

#include <hip/hip_runtime.h>

template <typename C>
__global__ void list_array_rpad_and_clip_length_axis1_kernel(
    const C*   fromstarts,
    const C*   fromstops,
    long long  length,
    long long  target,
    long long* out_total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long range = (long long)fromstops[i] - (long long)fromstarts[i];
    long long contrib = target > range ? target : range;
    atomicAdd((unsigned long long*)out_total, (unsigned long long)contrib);
}

template <typename C>
static void launch_list_array_rpad_and_clip_length_axis1(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   target,
    long long*  out_total,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_rpad_and_clip_length_axis1_kernel<C>,
        grid, block, 0, stream,
        (const C*)fromstarts, (const C*)fromstops, length, target, out_total
    );
}

extern "C" void awkward_hip_list_array_rpad_and_clip_length_axis1(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   target,
    int         dtype_code,   // 0=i32, 1=u32, 2=i64
    long long*  out_total,    // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_rpad_and_clip_length_axis1<int>         (fromstarts, fromstops, length, target, out_total, stream); break;
        case 1:  launch_list_array_rpad_and_clip_length_axis1<unsigned int>(fromstarts, fromstops, length, target, out_total, stream); break;
        default: launch_list_array_rpad_and_clip_length_axis1<long long>   (fromstarts, fromstops, length, target, out_total, stream); break;
    }
}
