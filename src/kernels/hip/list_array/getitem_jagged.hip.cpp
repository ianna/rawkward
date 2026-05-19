// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Five kernels for jagged-slice getitem on a ListArray.
//
// 1. awkward_hip_list_array_getitem_jagged_apply (single-threaded)
//    Apply a jagged integer-index slice to each outer list.
//    Writes tooffsets + tocarry.
//    *out_error: 1=slice stop<start, 2=slice OOB, 3=list stop<start,
//                4=list OOB, 5=index OOB.
//
// 2. awkward_hip_list_array_getitem_jagged_numvalid (parallel)
//    Count non-negative entries in missing across all slice windows.
//    Writes *out_count.
//    *out_error: 1=stop<start, 2=stop>missinglength.
//
// 3. awkward_hip_list_array_getitem_jagged_carrylen (parallel)
//    Sum slicestops[i]-slicestarts[i] into *out_count.
//
// 4. awkward_hip_list_array_getitem_jagged_shrink (single-threaded)
//    Filter missing>=0 into tocarry; build tosmalloffsets + tolargeoffsets.
//    Writes *out_count = number of valid entries.
//
// 5. awkward_hip_list_array_getitem_jagged_descend (single-threaded)
//    Verify slice lengths == list lengths, build cumulative tooffsets.
//    *out_error: 1=length mismatch.
//
// dtype_code: 0=i32, 1=u32, 2=i64 (for fromstarts/fromstops in apply/descend).
// All other arrays are i64.
//
// Corresponds to CPU: list_array_getitem_jagged_*.

#include <hip/hip_runtime.h>

// ── 1. apply (single-threaded) ────────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_jagged_apply_kernel(
    long long*       tooffsets,
    long long*       tocarry,
    const long long* slicestarts,
    const long long* slicestops,
    const long long* sliceindex,
    long long        sliceinnerlen,
    const C*         fromstarts,
    const C*         fromstops,
    long long        contentlen,
    long long        sliceouterlen,
    int*             out_error
) {
    long long k = 0;
    for (long long i = 0; i < sliceouterlen; ++i) {
        long long ss = slicestarts[i];
        long long se = slicestops [i];
        tooffsets[i] = k;
        if (ss != se) {
            if (se < ss)           { *out_error = 1; return; }
            if (se > sliceinnerlen){ *out_error = 2; return; }
            long long start  = (long long)fromstarts[i];
            long long stop   = (long long)fromstops [i];
            if (stop < start)      { *out_error = 3; return; }
            if (start != stop && stop > contentlen) { *out_error = 4; return; }
            long long count = stop - start;
            for (long long j = ss; j < se; ++j) {
                long long idx = sliceindex[j];
                if (idx < -count || idx >= count) { *out_error = 5; return; }
                if (idx < 0) idx += count;
                tocarry[k++] = start + idx;
            }
        }
    }
    tooffsets[sliceouterlen] = k;
}

template <typename C>
static void launch_jagged_apply(
    long long* tooffsets, long long* tocarry,
    const long long* slicestarts, const long long* slicestops,
    const long long* sliceindex, long long sliceinnerlen,
    const void* fromstarts, const void* fromstops,
    long long contentlen, long long sliceouterlen,
    int* out_error, hipStream_t stream
) {
    hipLaunchKernelGGL(
        list_array_getitem_jagged_apply_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tooffsets, tocarry,
        slicestarts, slicestops, sliceindex, sliceinnerlen,
        (const C*)fromstarts, (const C*)fromstops,
        contentlen, sliceouterlen, out_error
    );
}

extern "C" void awkward_hip_list_array_getitem_jagged_apply(
    long long*       tooffsets,
    long long*       tocarry,
    const long long* slicestarts,
    const long long* slicestops,
    const long long* sliceindex,
    long long        sliceinnerlen,
    const void*      fromstarts,
    const void*      fromstops,
    long long        contentlen,
    long long        sliceouterlen,
    int              dtype_code,
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_jagged_apply<int>         (tooffsets, tocarry, slicestarts, slicestops, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, sliceouterlen, out_error, stream); break;
        case 1:  launch_jagged_apply<unsigned int>(tooffsets, tocarry, slicestarts, slicestops, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, sliceouterlen, out_error, stream); break;
        default: launch_jagged_apply<long long>   (tooffsets, tocarry, slicestarts, slicestops, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, sliceouterlen, out_error, stream); break;
    }
}

// ── 2. numvalid (parallel) ────────────────────────────────────────────────────

__global__ void list_array_getitem_jagged_numvalid_kernel(
    const long long* slicestarts,
    const long long* slicestops,
    const long long* missing,
    long long        missinglength,
    long long        length,
    long long*       out_count,
    int*             out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long ss = slicestarts[i];
    long long se = slicestops [i];
    if (ss == se) return;
    if (se < ss)              { atomicCAS(out_error, 0, 1); return; }
    if (se > missinglength)   { atomicCAS(out_error, 0, 2); return; }
    long long local_count = 0;
    for (long long j = ss; j < se; ++j) {
        if (missing[j] >= 0) ++local_count;
    }
    atomicAdd((unsigned long long*)out_count, (unsigned long long)local_count);
}

extern "C" void awkward_hip_list_array_getitem_jagged_numvalid(
    const long long* slicestarts,
    const long long* slicestops,
    const long long* missing,
    long long        missinglength,
    long long        length,
    long long*       out_count,  // device pointer, caller zero-inits
    int*             out_error,  // device pointer, caller zero-inits
    hipStream_t      stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_jagged_numvalid_kernel,
        grid, block, 0, stream,
        slicestarts, slicestops, missing, missinglength, length, out_count, out_error
    );
}

// ── 3. carrylen (parallel) ────────────────────────────────────────────────────

__global__ void list_array_getitem_jagged_carrylen_kernel(
    const long long* slicestarts,
    const long long* slicestops,
    long long        length,
    long long*       out_count
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    long long delta = slicestops[i] - slicestarts[i];
    atomicAdd((unsigned long long*)out_count, (unsigned long long)delta);
}

extern "C" void awkward_hip_list_array_getitem_jagged_carrylen(
    const long long* slicestarts,
    const long long* slicestops,
    long long        length,
    long long*       out_count,  // device pointer, caller zero-inits
    hipStream_t      stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_getitem_jagged_carrylen_kernel,
        grid, block, 0, stream,
        slicestarts, slicestops, length, out_count
    );
}

// ── 4. shrink (single-threaded) ───────────────────────────────────────────────

__global__ void list_array_getitem_jagged_shrink_kernel(
    long long*       tocarry,
    long long*       tosmalloffsets,
    long long*       tolargeoffsets,
    const long long* slicestarts,
    const long long* slicestops,
    const long long* missing,
    long long        length,
    long long*       out_count
) {
    long long k = 0;
    if (length == 0) {
        tosmalloffsets[0] = 0;
        tolargeoffsets[0] = 0;
        *out_count = 0;
        return;
    }
    tosmalloffsets[0] = slicestarts[0];
    tolargeoffsets[0] = slicestarts[0];
    for (long long i = 0; i < length; ++i) {
        long long ss = slicestarts[i];
        long long se = slicestops [i];
        if (ss != se) {
            long long smallcount = 0;
            for (long long j = ss; j < se; ++j) {
                if (missing[j] >= 0) {
                    tocarry[k++] = j;
                    ++smallcount;
                }
            }
            tosmalloffsets[i + 1] = tosmalloffsets[i] + smallcount;
        } else {
            tosmalloffsets[i + 1] = tosmalloffsets[i];
        }
        tolargeoffsets[i + 1] = tolargeoffsets[i] + (se - ss);
    }
    *out_count = k;
}

extern "C" void awkward_hip_list_array_getitem_jagged_shrink(
    long long*       tocarry,
    long long*       tosmalloffsets,
    long long*       tolargeoffsets,
    const long long* slicestarts,
    const long long* slicestops,
    const long long* missing,
    long long        length,
    long long*       out_count,  // device pointer
    hipStream_t      stream
) {
    hipLaunchKernelGGL(
        list_array_getitem_jagged_shrink_kernel,
        dim3(1), dim3(1), 0, stream,
        tocarry, tosmalloffsets, tolargeoffsets,
        slicestarts, slicestops, missing, length, out_count
    );
}

// ── 5. descend (single-threaded) ─────────────────────────────────────────────

template <typename C>
__global__ void list_array_getitem_jagged_descend_kernel(
    long long*       tooffsets,
    const long long* slicestarts,
    const long long* slicestops,
    const C*         fromstarts,
    const C*         fromstops,
    long long        sliceouterlen,
    int*             out_error
) {
    if (sliceouterlen == 0) {
        tooffsets[0] = 0;
        return;
    }
    tooffsets[0] = slicestarts[0];
    for (long long i = 0; i < sliceouterlen; ++i) {
        long long slicecount = slicestops[i] - slicestarts[i];
        long long count = (long long)fromstops[i] - (long long)fromstarts[i];
        if (slicecount != count) { *out_error = 1; return; }
        tooffsets[i + 1] = tooffsets[i] + count;
    }
}

template <typename C>
static void launch_jagged_descend(
    long long* tooffsets,
    const long long* slicestarts, const long long* slicestops,
    const void* fromstarts, const void* fromstops,
    long long sliceouterlen, int* out_error, hipStream_t stream
) {
    hipLaunchKernelGGL(
        list_array_getitem_jagged_descend_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tooffsets, slicestarts, slicestops,
        (const C*)fromstarts, (const C*)fromstops,
        sliceouterlen, out_error
    );
}

extern "C" void awkward_hip_list_array_getitem_jagged_descend(
    long long*       tooffsets,
    const long long* slicestarts,
    const long long* slicestops,
    const void*      fromstarts,
    const void*      fromstops,
    long long        sliceouterlen,
    int              dtype_code,
    int*             out_error,    // device pointer, caller zero-inits
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_jagged_descend<int>         (tooffsets, slicestarts, slicestops, fromstarts, fromstops, sliceouterlen, out_error, stream); break;
        case 1:  launch_jagged_descend<unsigned int>(tooffsets, slicestarts, slicestops, fromstarts, fromstops, sliceouterlen, out_error, stream); break;
        default: launch_jagged_descend<long long>   (tooffsets, slicestarts, slicestops, fromstarts, fromstops, sliceouterlen, out_error, stream); break;
    }
}
