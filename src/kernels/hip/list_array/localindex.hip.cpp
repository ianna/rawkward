// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Fill a flat index with within-list positions.
//
// awkward_hip_list_array_localindex:
//   For each list i spanning offsets[i]..offsets[i+1]:
//     toindex[j] = j - offsets[i]   for j in offsets[i]..offsets[i+1]
//
// One GPU thread per list. offsets has dtype_code type (0=i32, 1=u32, 2=i64).
// toindex is always i64.
//
// Corresponds to CPU: list_array_localindex.

#include <hip/hip_runtime.h>

template <typename OFF>
__global__ void list_array_localindex_kernel(
    long long*  toindex,
    const OFF*  offsets,
    long long   length    // number of lists = offsetslength - 1
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long start = (long long)offsets[i];
    long long stop  = (long long)offsets[i + 1];
    for (long long j = start; j < stop; ++j) {
        toindex[j] = j - start;
    }
}

template <typename OFF>
static void launch_list_array_localindex(
    long long*  toindex,
    const void* offsets,
    long long   length,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_localindex_kernel<OFF>,
        grid, block, 0, stream,
        toindex, (const OFF*)offsets, length
    );
}

extern "C" void awkward_hip_list_array_localindex(
    long long*  toindex,
    const void* offsets,
    long long   length,      // number of lists (= offsetslength - 1)
    int         dtype_code,  // 0=i32, 1=u32, 2=i64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_list_array_localindex<int>         (toindex, offsets, length, stream); break;
        case 1:  launch_list_array_localindex<unsigned int>(toindex, offsets, length, stream); break;
        default: launch_list_array_localindex<long long>   (toindex, offsets, length, stream); break;
    }
}
