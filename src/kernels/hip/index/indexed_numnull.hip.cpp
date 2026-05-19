// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Three null-counting kernels for IndexedArray:
//
// ── awkward_hip_indexed_array_numnull ─────────────────────────────────────────
//   Count negative entries; atomicAdd into *out_count (device i64, caller zero-inits).
//   One GPU thread per element.
//
// ── awkward_hip_indexed_array_numnull_parents ─────────────────────────────────
//   Per-element null flag (1 or 0) written to numnull[]; total count in *out_total
//   (device i64, caller zero-inits).  One GPU thread per element.
//
// ── awkward_hip_indexed_array_numnull_unique ──────────────────────────────────
//   Fill toindex[0..lenindex-1] = 0,1,...,lenindex-1; toindex[lenindex] = -1.
//   One GPU thread per element.
//
// dtype_code for numnull and numnull_parents: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: indexed_numnull, indexed_numnull_parents,
//                     indexed_numnull_unique.

#include <hip/hip_runtime.h>

// ── numnull ───────────────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_numnull_kernel(
    const FROM* fromindex,
    long long   length,
    long long*  out_count
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    if ((long long)fromindex[i] < 0)
        atomicAdd((unsigned long long*)out_count, 1ULL);
}

template <typename FROM>
static void launch_indexed_numnull(
    const void* fromindex,
    long long   length,
    long long*  out_count,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_numnull_kernel<FROM>,
        grid, block, 0, stream,
        (const FROM*)fromindex, length, out_count
    );
}

extern "C" void awkward_hip_indexed_array_numnull(
    const void* fromindex,
    long long   length,
    long long*  out_count,    // device pointer, caller zero-inits
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_indexed_numnull<int>         (fromindex, length, out_count, stream); break;
        case 1:  launch_indexed_numnull<unsigned int>(fromindex, length, out_count, stream); break;
        default: launch_indexed_numnull<long long>   (fromindex, length, out_count, stream); break;
    }
}

// ── numnull_parents ───────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_numnull_parents_kernel(
    long long*  numnull,
    const FROM* fromindex,
    long long   length,
    long long*  out_total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    if ((long long)fromindex[i] < 0) {
        numnull[i] = 1;
        atomicAdd((unsigned long long*)out_total, 1ULL);
    } else {
        numnull[i] = 0;
    }
}

template <typename FROM>
static void launch_numnull_parents(
    long long*  numnull,
    const void* fromindex,
    long long   length,
    long long*  out_total,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_numnull_parents_kernel<FROM>,
        grid, block, 0, stream,
        numnull, (const FROM*)fromindex, length, out_total
    );
}

extern "C" void awkward_hip_indexed_array_numnull_parents(
    long long*  numnull,      // device array, length elements
    const void* fromindex,
    long long   length,
    long long*  out_total,    // device pointer, caller zero-inits
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_numnull_parents<int>         (numnull, fromindex, length, out_total, stream); break;
        case 1:  launch_numnull_parents<unsigned int>(numnull, fromindex, length, out_total, stream); break;
        default: launch_numnull_parents<long long>   (numnull, fromindex, length, out_total, stream); break;
    }
}

// ── numnull_unique ────────────────────────────────────────────────────────────
// Fill toindex[0..lenindex] = 0,1,...,lenindex-1 and toindex[lenindex] = -1.
// Launch lenindex+1 threads.

__global__ void indexed_numnull_unique_kernel(
    long long* toindex,
    long long  lenindex
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lenindex)
        toindex[i] = i;
    else if (i == lenindex)
        toindex[i] = -1;
}

extern "C" void awkward_hip_indexed_array_numnull_unique(
    long long*  toindex,    // device array, lenindex+1 elements
    long long   lenindex,
    hipStream_t stream
) {
    if (lenindex < 0) return;
    long long total = lenindex + 1;
    dim3 block(256);
    dim3 grid((unsigned int)((total + 255) / 256));
    hipLaunchKernelGGL(
        indexed_numnull_unique_kernel,
        grid, block, 0, stream,
        toindex, lenindex
    );
}
