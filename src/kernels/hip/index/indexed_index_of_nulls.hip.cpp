// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Collect within-list positions of null entries.
//
// For each i where fromindex[i] < 0:
//   toindex[j++] = i - starts[parents[i]]
//
// Single-threaded (sequential j counter required for deterministic ordering).
// *out_count receives the number of nulls written.
//
// dtype_code for fromindex: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: indexed_index_of_nulls (indexed_array_index_of_nulls).

#include <hip/hip_runtime.h>

template <typename FROM>
__global__ void indexed_index_of_nulls_kernel(
    long long*         toindex,
    const FROM*        fromindex,
    const long long*   parents,
    const long long*   starts,
    long long          length,
    long long*         out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long j = 0;
    for (long long i = 0; i < length; i++) {
        if ((long long)fromindex[i] < 0) {
            long long parent = parents[i];
            toindex[j++] = i - starts[parent];
        }
    }
    *out_count = j;
}

template <typename FROM>
static void launch_index_of_nulls(
    long long*         toindex,
    const void*        fromindex,
    const long long*   parents,
    const long long*   starts,
    long long          length,
    long long*         out_count,
    hipStream_t        stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        indexed_index_of_nulls_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        toindex, (const FROM*)fromindex, parents, starts, length, out_count
    );
}

extern "C" void awkward_hip_indexed_array_index_of_nulls(
    long long*         toindex,
    const void*        fromindex,
    const long long*   parents,
    const long long*   starts,
    long long          length,
    long long*         out_count,    // device pointer, receives null count
    int                dtype_code,
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_index_of_nulls<int>         (toindex, fromindex, parents, starts, length, out_count, stream); break;
        case 1:  launch_index_of_nulls<unsigned int>(toindex, fromindex, parents, starts, length, out_count, stream); break;
        default: launch_index_of_nulls<long long>   (toindex, fromindex, parents, starts, length, out_count, stream); break;
    }
}
