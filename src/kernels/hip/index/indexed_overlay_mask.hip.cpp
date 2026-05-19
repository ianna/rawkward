// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Overlay a byte mask on an index array.
//
// For each i:
//   mask[i] != 0  →  toindex[i] = -1
//   mask[i] == 0  →  toindex[i] = (long long)fromindex[i]
//
// `toindex` is always i64.  `fromindex` dtype is selected by dtype_code:
//   0 = i32, 1 = u32, 2 = i64
//
// One GPU thread per element.
//
// Corresponds to CPU: indexed_overlay_mask.

#include <hip/hip_runtime.h>

template <typename FROM>
__global__ void indexed_overlay_mask_kernel(
    long long*           toindex,
    const signed char*   mask,
    const FROM*          fromindex,
    long long            length
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    toindex[i] = (mask[i] != 0) ? -1LL : (long long)fromindex[i];
}

template <typename FROM>
static void launch_indexed_overlay_mask(
    long long*         toindex,
    const signed char* mask,
    const void*        fromindex,
    long long          length,
    hipStream_t        stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_overlay_mask_kernel<FROM>,
        grid, block, 0, stream,
        toindex, mask, (const FROM*)fromindex, length
    );
}

extern "C" void awkward_hip_indexed_array_overlay_mask(
    long long*         toindex,
    const signed char* mask,
    const void*        fromindex,
    long long          length,
    int                dtype_code,    // 0=i32, 1=u32, 2=i64
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_indexed_overlay_mask<int>          (toindex, mask, fromindex, length, stream); break;
        case 1:  launch_indexed_overlay_mask<unsigned int> (toindex, mask, fromindex, length, stream); break;
        default: launch_indexed_overlay_mask<long long>    (toindex, mask, fromindex, length, stream); break;
    }
}
