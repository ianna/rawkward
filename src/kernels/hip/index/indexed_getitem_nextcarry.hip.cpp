// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two carry-building kernels for indexed getitem.
//
// ── awkward_hip_indexed_array_getitem_nextcarry ────────────────────────────────
//   Copy all fromindex entries to tocarry (tocarry[i] = fromindex[i]).
//   Error if any entry is < 0 or >= lencontent.
//   One GPU thread per element (order preserved; no skipping).
//
// ── awkward_hip_indexed_array_getitem_nextcarry_outindex ──────────────────────
//   For each i:
//     j >= lencontent  →  error
//     j < 0            →  toindex[i] = -1
//     j >= 0           →  tocarry[k] = j, toindex[i] = k, k++
//   Single-threaded (grid=1, block=1) — sequential k counter required.
//
// dtype_code for fromindex: 0=i32, 1=u32, 2=i64.
// out_error: device int, caller zero-inits; 1 = error.
// out_count: device i64, receives number of entries written to tocarry.
//
// Corresponds to CPU: indexed_getitem_nextcarry, indexed_getitem_nextcarry_outindex.

#include <hip/hip_runtime.h>

// ── getitem_nextcarry ─────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_getitem_nextcarry_kernel(
    long long*   tocarry,
    const FROM*  fromindex,
    long long    length,
    long long    lencontent,
    int*         out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long j = (long long)fromindex[i];
    if (j < 0 || j >= lencontent) {
        atomicCAS(out_error, 0, 1);
    } else {
        tocarry[i] = j;
    }
}

template <typename FROM>
static void launch_getitem_nextcarry(
    long long*  tocarry,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    int*        out_error,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        indexed_getitem_nextcarry_kernel<FROM>,
        grid, block, 0, stream,
        tocarry, (const FROM*)fromindex, length, lencontent, out_error
    );
}

extern "C" void awkward_hip_indexed_array_getitem_nextcarry(
    long long*  tocarry,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    int*        out_error,    // device pointer, caller zero-inits
    int         dtype_code,   // 0=i32, 1=u32, 2=i64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_getitem_nextcarry<int>         (tocarry, fromindex, length, lencontent, out_error, stream); break;
        case 1:  launch_getitem_nextcarry<unsigned int>(tocarry, fromindex, length, lencontent, out_error, stream); break;
        default: launch_getitem_nextcarry<long long>   (tocarry, fromindex, length, lencontent, out_error, stream); break;
    }
}

// ── getitem_nextcarry_outindex ────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_getitem_nextcarry_outindex_kernel(
    long long*   tocarry,
    long long*   toindex,
    const FROM*  fromindex,
    long long    length,
    long long    lencontent,
    long long*   out_count,
    int*         out_error
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long k = 0;
    for (long long i = 0; i < length; i++) {
        long long j = (long long)fromindex[i];
        if (j >= lencontent) {
            atomicCAS(out_error, 0, 1);
            *out_count = k;
            return;
        } else if (j < 0) {
            toindex[i] = -1;
        } else {
            tocarry[k] = j;
            toindex[i] = k;
            k++;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_getitem_nextcarry_outindex(
    long long*  tocarry,
    long long*  toindex,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    long long*  out_count,
    int*        out_error,
    hipStream_t stream
) {
    if (length <= 0) { return; }
    hipLaunchKernelGGL(
        indexed_getitem_nextcarry_outindex_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        tocarry, toindex, (const FROM*)fromindex,
        length, lencontent, out_count, out_error
    );
}

extern "C" void awkward_hip_indexed_array_getitem_nextcarry_outindex(
    long long*  tocarry,
    long long*  toindex,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    long long*  out_count,    // device pointer, receives count written
    int*        out_error,    // device pointer, caller zero-inits
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_getitem_nextcarry_outindex<int>         (tocarry, toindex, fromindex, length, lencontent, out_count, out_error, stream); break;
        case 1:  launch_getitem_nextcarry_outindex<unsigned int>(tocarry, toindex, fromindex, length, lencontent, out_count, out_error, stream); break;
        default: launch_getitem_nextcarry_outindex<long long>   (tocarry, toindex, fromindex, length, lencontent, out_count, out_error, stream); break;
    }
}
