// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Four kernels for range-slice getitem on a ListArray.
//
// All use INT64_MIN as the sentinel for "unspecified" (Python None) start/stop.
//
// 1. awkward_hip_list_array_getitem_next_range_carrylength
//    Sum the number of elements selected by [start:stop:step] across all lists.
//    One GPU thread per list; atomicAdd to accumulate.
//
// 2. awkward_hip_list_array_getitem_next_range_counts
//    Sum fromoffsets[i+1]-fromoffsets[i] for i in 0..lenstarts.
//    One GPU thread per entry; atomicAdd.
//
// 3. awkward_hip_list_array_getitem_next_range
//    Build tooffsets and tocarry for the range slice.
//    Single-threaded (sequential k counter).
//
// 4. awkward_hip_list_array_getitem_next_range_spreadadvanced
//    For each list i fill fromoffsets[i]..fromoffsets[i+1] with fromadvanced[i].
//    One GPU thread per list.
//
// dtype_code: 0=i32, 1=u32, 2=i64 (for fromstarts/fromstops/fromoffsets).
//
// Corresponds to CPU: list_array_getitem_next_range*.

#include <hip/hip_runtime.h>
#include <climits>

// ── regularize_rangeslice (device helper) ─────────────────────────────────────
// start/stop: INT64_MIN means "unspecified" (Python None).

__device__ static void regularize_rangeslice(
    long long  start_in, long long  stop_in,
    int        posstep,
    long long  length,
    long long* out_rs, long long* out_re
) {
    if (posstep) {
        long long s, e;
        if (start_in == LLONG_MIN) {
            s = 0;
        } else if (start_in < 0) {
            s = start_in + length;
            if (s < 0) s = 0;
        } else {
            s = start_in < length ? start_in : length;
        }
        if (s < 0) s = 0; else if (s > length) s = length;

        if (stop_in == LLONG_MIN) {
            e = length;
        } else if (stop_in < 0) {
            e = stop_in + length;
            if (e < 0) e = 0;
        } else {
            e = stop_in < length ? stop_in : length;
        }
        if (e < 0) e = 0; else if (e > length) e = length;
        if (e < s) e = s;

        *out_rs = s;
        *out_re = e;
    } else {
        long long s, e;
        if (start_in == LLONG_MIN) {
            s = length - 1;
        } else if (start_in < 0) {
            s = start_in + length;
            if (s < -1) s = -1;
        } else {
            s = start_in < length - 1 ? start_in : length - 1;
        }
        if (s < -1) s = -1; else if (s > length - 1) s = length - 1;

        if (stop_in == LLONG_MIN) {
            e = -1;
        } else if (stop_in < 0) {
            e = stop_in + length;
            if (e < -1) e = -1;
        } else {
            e = stop_in < length - 1 ? stop_in : length - 1;
        }
        if (e < -1) e = -1; else if (e > length - 1) e = length - 1;
        if (e > s) e = s;

        *out_rs = s;
        *out_re = e;
    }
}

// ── 1. carrylength ────────────────────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_next_range_carrylength_kernel(
    const C*   fromstarts,
    const C*   fromstops,
    long long  length,
    long long  start,
    long long  stop,
    long long  step,
    long long* out_total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long listlen = (long long)fromstops[i] - (long long)fromstarts[i];
    int posstep = (step > 0) ? 1 : 0;
    long long rs, re;
    regularize_rangeslice(start, stop, posstep, listlen, &rs, &re);
    long long count;
    if (posstep) {
        count = (re > rs) ? (re - rs + step - 1) / step : 0;
    } else {
        long long neg = -step;
        count = (rs > re) ? (rs - re + neg - 1) / neg : 0;
    }
    atomicAdd((unsigned long long*)out_total, (unsigned long long)count);
}

template <typename C>
static void launch_carrylength(
    const void* fromstarts, const void* fromstops,
    long long length, long long start, long long stop, long long step,
    long long* out_total, hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_range_carrylength_kernel<C>,
        grid, block, 0, stream,
        (const C*)fromstarts, (const C*)fromstops, length, start, stop, step, out_total
    );
}

extern "C" void awkward_hip_list_array_getitem_next_range_carrylength(
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   start,      // INT64_MIN = None
    long long   stop,       // INT64_MIN = None
    long long   step,
    int         dtype_code,
    long long*  out_total,  // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_carrylength<int>         (fromstarts, fromstops, length, start, stop, step, out_total, stream); break;
        case 1:  launch_carrylength<unsigned int>(fromstarts, fromstops, length, start, stop, step, out_total, stream); break;
        default: launch_carrylength<long long>   (fromstarts, fromstops, length, start, stop, step, out_total, stream); break;
    }
}

// ── 2. counts ─────────────────────────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_next_range_counts_kernel(
    const C*   fromoffsets,
    long long  lenstarts,
    long long* out_total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lenstarts) return;
    long long delta = (long long)fromoffsets[i + 1] - (long long)fromoffsets[i];
    atomicAdd((unsigned long long*)out_total, (unsigned long long)delta);
}

template <typename C>
static void launch_counts(
    const void* fromoffsets, long long lenstarts,
    long long* out_total, hipStream_t stream
) {
    if (lenstarts <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((lenstarts + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_range_counts_kernel<C>,
        grid, block, 0, stream,
        (const C*)fromoffsets, lenstarts, out_total
    );
}

extern "C" void awkward_hip_list_array_getitem_next_range_counts(
    const void* fromoffsets,
    long long   lenstarts,
    int         dtype_code,
    long long*  out_total,  // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_counts<int>         (fromoffsets, lenstarts, out_total, stream); break;
        case 1:  launch_counts<unsigned int>(fromoffsets, lenstarts, out_total, stream); break;
        default: launch_counts<long long>   (fromoffsets, lenstarts, out_total, stream); break;
    }
}

// ── 3. range (build tooffsets + tocarry) ──────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_next_range_kernel(
    long long*  tooffsets,
    long long*  tocarry,
    const C*    fromstarts,
    const C*    fromstops,
    long long   length,
    long long   start,
    long long   stop,
    long long   step
) {
    tooffsets[0] = 0LL;
    int posstep = (step > 0) ? 1 : 0;
    long long k = 0;
    for (long long i = 0; i < length; ++i) {
        long long fstart = (long long)fromstarts[i];
        long long listlen = (long long)fromstops[i] - fstart;
        long long rs, re;
        regularize_rangeslice(start, stop, posstep, listlen, &rs, &re);
        if (posstep) {
            for (long long j = rs; j < re; j += step) {
                tocarry[k++] = fstart + j;
            }
        } else {
            for (long long j = rs; j > re; j += step) {
                tocarry[k++] = fstart + j;
            }
        }
        tooffsets[i + 1] = k;
    }
}

template <typename C>
static void launch_range(
    long long* tooffsets, long long* tocarry,
    const void* fromstarts, const void* fromstops,
    long long length, long long start, long long stop, long long step,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        list_array_getitem_next_range_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tooffsets, tocarry,
        (const C*)fromstarts, (const C*)fromstops,
        length, start, stop, step
    );
}

extern "C" void awkward_hip_list_array_getitem_next_range(
    long long*  tooffsets,
    long long*  tocarry,
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   start,      // INT64_MIN = None
    long long   stop,       // INT64_MIN = None
    long long   step,
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_range<int>         (tooffsets, tocarry, fromstarts, fromstops, length, start, stop, step, stream); break;
        case 1:  launch_range<unsigned int>(tooffsets, tocarry, fromstarts, fromstops, length, start, stop, step, stream); break;
        default: launch_range<long long>   (tooffsets, tocarry, fromstarts, fromstops, length, start, stop, step, stream); break;
    }
}

// ── 4. spreadadvanced ─────────────────────────────────────────────────────────

__global__ void list_array_getitem_next_range_spreadadvanced_kernel(
    long long*       toadvanced,
    const long long* fromadvanced,
    const long long* fromoffsets,
    long long        lenstarts
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lenstarts) return;
    long long start = fromoffsets[i];
    long long stop  = fromoffsets[i + 1];
    long long val   = fromadvanced[i];
    for (long long j = start; j < stop; ++j) {
        toadvanced[j] = val;
    }
}

extern "C" void awkward_hip_list_array_getitem_next_range_spreadadvanced(
    long long*       toadvanced,
    const long long* fromadvanced,
    const long long* fromoffsets,
    long long        lenstarts,
    hipStream_t      stream
) {
    if (lenstarts <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((lenstarts + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_next_range_spreadadvanced_kernel,
        grid, block, 0, stream,
        toadvanced, fromadvanced, fromoffsets, lenstarts
    );
}
