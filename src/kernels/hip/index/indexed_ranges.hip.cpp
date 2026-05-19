// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two range-based carry kernels for IndexedArray.
//
// ── awkward_hip_indexed_array_ranges_carry_next ───────────────────────────────
//   For each range i defined by [fromstarts[i], fromstops[i]), scan index[]
//   and append every non-negative value to tocarry.
//   *out_count receives the total entries written.
//   Single-threaded (sequential carry counter required).
//
// ── awkward_hip_indexed_array_ranges_next ─────────────────────────────────────
//   For each range i, count non-negative index values → prefix sum into
//   tostarts[i] and tostops[i].  *out_total receives the grand total.
//   Single-threaded (prefix-sum dependency).
//
// dtype_code for index: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: indexed_ranges_carry_next, indexed_ranges_next.

#include <hip/hip_runtime.h>

// ── ranges_carry_next ─────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_ranges_carry_next_kernel(
    long long*         tocarry,
    const FROM*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long k = 0;
    for (long long i = 0; i < nranges; i++) {
        long long start = fromstarts[i];
        long long stop  = fromstops[i];
        for (long long j = start; j < stop; j++) {
            long long v = (long long)index[j];
            if (v >= 0) tocarry[k++] = v;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_ranges_carry_next(
    long long*         tocarry,
    const void*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         out_count,
    hipStream_t        stream
) {
    if (nranges <= 0) return;
    hipLaunchKernelGGL(
        indexed_ranges_carry_next_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        tocarry, (const FROM*)index, fromstarts, fromstops, nranges, out_count
    );
}

extern "C" void awkward_hip_indexed_array_ranges_carry_next(
    long long*         tocarry,
    const void*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         out_count,    // device pointer, receives total written
    int                dtype_code,
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_ranges_carry_next<int>         (tocarry, index, fromstarts, fromstops, nranges, out_count, stream); break;
        case 1:  launch_ranges_carry_next<unsigned int>(tocarry, index, fromstarts, fromstops, nranges, out_count, stream); break;
        default: launch_ranges_carry_next<long long>   (tocarry, index, fromstarts, fromstops, nranges, out_count, stream); break;
    }
}

// ── ranges_next ───────────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_ranges_next_kernel(
    const FROM*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         tostarts,
    long long*         tostops,
    long long*         out_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long k = 0;
    for (long long i = 0; i < nranges; i++) {
        long long start = fromstarts[i];
        long long stop  = fromstops[i];
        tostarts[i] = k;
        for (long long j = start; j < stop; j++) {
            if ((long long)index[j] >= 0) k++;
        }
        tostops[i] = k;
    }
    *out_total = k;
}

template <typename FROM>
static void launch_ranges_next(
    const void*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         tostarts,
    long long*         tostops,
    long long*         out_total,
    hipStream_t        stream
) {
    if (nranges <= 0) return;
    hipLaunchKernelGGL(
        indexed_ranges_next_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        (const FROM*)index, fromstarts, fromstops, nranges,
        tostarts, tostops, out_total
    );
}

extern "C" void awkward_hip_indexed_array_ranges_next(
    const void*        index,
    const long long*   fromstarts,
    const long long*   fromstops,
    long long          nranges,
    long long*         tostarts,
    long long*         tostops,
    long long*         out_total,    // device pointer, receives grand total
    int                dtype_code,
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_ranges_next<int>         (index, fromstarts, fromstops, nranges, tostarts, tostops, out_total, stream); break;
        case 1:  launch_ranges_next<unsigned int>(index, fromstarts, fromstops, nranges, tostarts, tostops, out_total, stream); break;
        default: launch_ranges_next<long long>   (index, fromstarts, fromstops, nranges, tostarts, tostops, out_total, stream); break;
    }
}
