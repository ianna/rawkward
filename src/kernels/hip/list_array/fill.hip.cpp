// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Copy starts/stops into a destination slice with a base offset added.
//
// awkward_hip_list_array_fill:
//   tostarts[tostartsoffset + i] = fromstarts[i] + base
//   tostops [tostopsoffset  + i] = fromstops [i] + base
//
// dtype_code: 0=i32, 1=u32, 2=i64 for fromstarts/fromstops.
// tostarts/tostops are always i64.  One GPU thread per element.
//
// Corresponds to CPU: list_array_fill.

#include <hip/hip_runtime.h>

template <typename FROM>
__global__ void list_array_fill_kernel(
    long long*  tostarts,
    long long   tostartsoffset,
    long long*  tostops,
    long long   tostopsoffset,
    const FROM* fromstarts,
    const FROM* fromstops,
    long long   base,
    long long   length
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    tostarts[tostartsoffset + i] = (long long)fromstarts[i] + base;
    tostops [tostopsoffset  + i] = (long long)fromstops [i] + base;
}

template <typename FROM>
static void launch_list_array_fill(
    long long*  tostarts,
    long long   tostartsoffset,
    long long*  tostops,
    long long   tostopsoffset,
    const void* fromstarts,
    const void* fromstops,
    long long   base,
    long long   length,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_fill_kernel<FROM>,
        grid, block, 0, stream,
        tostarts, tostartsoffset, tostops, tostopsoffset,
        (const FROM*)fromstarts, (const FROM*)fromstops, base, length
    );
}

extern "C" void awkward_hip_list_array_fill(
    long long*  tostarts,
    long long   tostartsoffset,
    long long*  tostops,
    long long   tostopsoffset,
    const void* fromstarts,
    const void* fromstops,
    long long   base,
    long long   length,
    int         dtype_code,    // 0=i32, 1=u32, 2=i64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_fill<int>         (tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstops, base, length, stream); break;
        case 1:  launch_list_array_fill<unsigned int>(tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstops, base, length, stream); break;
        default: launch_list_array_fill<long long>   (tostarts, tostartsoffset, tostops, tostopsoffset, fromstarts, fromstops, base, length, stream); break;
    }
}
