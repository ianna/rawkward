// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two fill operations for IndexedArray output indices.
//
// ── awkward_hip_indexed_array_fill ────────────────────────────────────────────
// For each i in [0, length):
//   fromindex[i] < 0  →  toindex[toindexoffset + i] = -1
//   otherwise         →  toindex[toindexoffset + i] = fromindex[i] + base
//
// dtype_code selects fromindex element type: 0=i32, 1=u32, 2=i64.
// toindex is always i64.  One GPU thread per element.
//
// ── awkward_hip_indexed_array_fill_count ──────────────────────────────────────
// For each i in [0, length):
//   toindex[toindexoffset + i] = base + i
//
// No dtype dispatch needed (always i64).  One GPU thread per element.
//
// Corresponds to CPU: indexed_fill, indexed_fill_count.

#include <hip/hip_runtime.h>

// ── indexed_fill ──────────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_fill_kernel(
    long long*  toindex,
    long long   toindexoffset,
    const FROM* fromindex,
    long long   length,
    long long   base
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long v = (long long)fromindex[i];
    toindex[toindexoffset + i] = (v < 0) ? -1LL : (v + base);
}

template <typename FROM>
static void launch_indexed_fill(
    long long*  toindex,
    long long   toindexoffset,
    const void* fromindex,
    long long   length,
    long long   base,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_fill_kernel<FROM>,
        grid, block, 0, stream,
        toindex, toindexoffset, (const FROM*)fromindex, length, base
    );
}

extern "C" void awkward_hip_indexed_array_fill(
    long long*  toindex,
    long long   toindexoffset,
    const void* fromindex,
    long long   length,
    long long   base,
    int         dtype_code,    // 0=i32, 1=u32, 2=i64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_indexed_fill<int>         (toindex, toindexoffset, fromindex, length, base, stream); break;
        case 1:  launch_indexed_fill<unsigned int>(toindex, toindexoffset, fromindex, length, base, stream); break;
        default: launch_indexed_fill<long long>   (toindex, toindexoffset, fromindex, length, base, stream); break;
    }
}

// ── indexed_fill_count ────────────────────────────────────────────────────────

__global__ void indexed_fill_count_kernel(
    long long* toindex,
    long long  toindexoffset,
    long long  length,
    long long  base
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    toindex[toindexoffset + i] = base + i;
}

extern "C" void awkward_hip_indexed_array_fill_count(
    long long*  toindex,
    long long   toindexoffset,
    long long   length,
    long long   base,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_fill_count_kernel,
        grid, block, 0, stream,
        toindex, toindexoffset, length, base
    );
}
