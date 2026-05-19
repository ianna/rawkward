// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two kernels for array-index getitem on a ListArray.
//
// 1. awkward_hip_list_array_getitem_next_array
//    For each list i and array-index j:
//      tocarry   [i*lenarray+j] = fromstarts[i] + regular_at
//      toadvanced[i*lenarray+j] = j
//
//    *out_error codes (device int, caller zero-inits):
//      1 – stops[i] < starts[i]
//      2 – stops[i] > lencontent
//      3 – fromarray[j] out of range for list i
//
// 2. awkward_hip_list_array_getitem_next_array_advanced
//    For each list i, pick one element via fromarray[fromadvanced[i]]:
//      tocarry   [i] = fromstarts[i] + regular_at
//      toadvanced[i] = i
//
//    Same *out_error codes.
//
// dtype_code: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: list_array_getitem_next_array*.

#include <hip/hip_runtime.h>

// ── 1. getitem_next_array ─────────────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_next_array_kernel(
    long long*       tocarry,
    long long*       toadvanced,
    const C*         fromstarts,
    const C*         fromstops,
    const long long* fromarray,
    long long        lenarray,
    long long        lenstarts,
    long long        lencontent,
    int*             out_error
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lenstarts * lenarray;
    if (tid >= total) return;

    long long i = tid / lenarray;
    long long j = tid % lenarray;

    long long start = (long long)fromstarts[i];
    long long stop  = (long long)fromstops [i];
    if (stop < start)                           { atomicCAS(out_error, 0, 1); return; }
    if (start != stop && stop > lencontent)     { atomicCAS(out_error, 0, 2); return; }
    long long length = stop - start;
    long long regular_at = fromarray[j];
    if (regular_at < 0) regular_at += length;
    if (regular_at < 0 || regular_at >= length) { atomicCAS(out_error, 0, 3); return; }
    tocarry   [i * lenarray + j] = start + regular_at;
    toadvanced[i * lenarray + j] = j;
}

template <typename C>
static void launch_getitem_next_array(
    long long* tocarry, long long* toadvanced,
    const void* fromstarts, const void* fromstops,
    const long long* fromarray,
    long long lenarray, long long lenstarts, long long lencontent,
    int* out_error, hipStream_t stream
) {
    long long total = lenstarts * lenarray;
    if (total <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((total + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_array_kernel<C>,
        grid, block, 0, stream,
        tocarry, toadvanced,
        (const C*)fromstarts, (const C*)fromstops,
        fromarray, lenarray, lenstarts, lencontent, out_error
    );
}

extern "C" void awkward_hip_list_array_getitem_next_array(
    long long*       tocarry,
    long long*       toadvanced,
    const void*      fromstarts,
    const void*      fromstops,
    const long long* fromarray,
    long long        lenarray,
    long long        lenstarts,
    long long        lencontent,
    int              dtype_code,
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_getitem_next_array<int>         (tocarry, toadvanced, fromstarts, fromstops, fromarray, lenarray, lenstarts, lencontent, out_error, stream); break;
        case 1:  launch_getitem_next_array<unsigned int>(tocarry, toadvanced, fromstarts, fromstops, fromarray, lenarray, lenstarts, lencontent, out_error, stream); break;
        default: launch_getitem_next_array<long long>   (tocarry, toadvanced, fromstarts, fromstops, fromarray, lenarray, lenstarts, lencontent, out_error, stream); break;
    }
}

// ── 2. getitem_next_array_advanced ────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_next_array_advanced_kernel(
    long long*       tocarry,
    long long*       toadvanced,
    const C*         fromstarts,
    const C*         fromstops,
    const long long* fromarray,
    const long long* fromadvanced,
    long long        lenstarts,
    long long        lencontent,
    int*             out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lenstarts) return;

    long long start = (long long)fromstarts[i];
    long long stop  = (long long)fromstops [i];
    if (stop < start)                           { atomicCAS(out_error, 0, 1); return; }
    if (start != stop && stop > lencontent)     { atomicCAS(out_error, 0, 2); return; }
    long long length = stop - start;
    long long raw = fromarray[fromadvanced[i]];
    long long regular_at = raw;
    if (regular_at < 0) regular_at += length;
    if (regular_at < 0 || regular_at >= length) { atomicCAS(out_error, 0, 3); return; }
    tocarry   [i] = start + regular_at;
    toadvanced[i] = i;
}

template <typename C>
static void launch_getitem_next_array_advanced(
    long long* tocarry, long long* toadvanced,
    const void* fromstarts, const void* fromstops,
    const long long* fromarray, const long long* fromadvanced,
    long long lenstarts, long long lencontent,
    int* out_error, hipStream_t stream
) {
    if (lenstarts <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((lenstarts + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_array_advanced_kernel<C>,
        grid, block, 0, stream,
        tocarry, toadvanced,
        (const C*)fromstarts, (const C*)fromstops,
        fromarray, fromadvanced, lenstarts, lencontent, out_error
    );
}

extern "C" void awkward_hip_list_array_getitem_next_array_advanced(
    long long*       tocarry,
    long long*       toadvanced,
    const void*      fromstarts,
    const void*      fromstops,
    const long long* fromarray,
    const long long* fromadvanced,
    long long        lenstarts,
    long long        lencontent,
    int              dtype_code,
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_getitem_next_array_advanced<int>         (tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, lenstarts, lencontent, out_error, stream); break;
        case 1:  launch_getitem_next_array_advanced<unsigned int>(tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, lenstarts, lencontent, out_error, stream); break;
        default: launch_getitem_next_array_advanced<long long>   (tocarry, toadvanced, fromstarts, fromstops, fromarray, fromadvanced, lenstarts, lencontent, out_error, stream); break;
    }
}
