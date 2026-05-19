// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two kernels for flattening an IndexedOptionArray.
//
// ── awkward_hip_indexed_array_flatten_nextcarry ───────────────────────────────
//   Collect non-negative fromindex entries into tocarry, skipping nulls (-1).
//   Error if any non-negative entry >= lencontent.
//   Single-threaded (sequential k counter required).
//   *out_count receives the number of entries written.
//
// ── awkward_hip_indexed_array_flatten_none2empty ──────────────────────────────
//   Build outoffsets for flattening: None → empty list (zero length),
//   valid entry idx → list length = offsets[idx+1] - offsets[idx].
//   outoffsets[0] = offsets[0]; outoffsets[i+1] = outoffsets[i] + count.
//   Error if idx+1 >= offsetslength for a non-null entry.
//   Single-threaded (prefix-sum dependency).
//
// dtype_code for fromindex/outindex: 0=i32, 1=u32, 2=i64.
// out_error: device int, caller zero-inits; 1 = error.
//
// Corresponds to CPU: indexed_flatten_nextcarry, indexed_flatten_none2empty.

#include <hip/hip_runtime.h>

// ── flatten_nextcarry ─────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_flatten_nextcarry_kernel(
    long long*   tocarry,
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
        } else if (j >= 0) {
            tocarry[k++] = j;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_flatten_nextcarry(
    long long*  tocarry,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    long long*  out_count,
    int*        out_error,
    hipStream_t stream
) {
    if (length <= 0) { if (out_count) { /* caller handles */ } return; }
    hipLaunchKernelGGL(
        indexed_flatten_nextcarry_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        tocarry, (const FROM*)fromindex, length, lencontent, out_count, out_error
    );
}

extern "C" void awkward_hip_indexed_array_flatten_nextcarry(
    long long*  tocarry,
    const void* fromindex,
    long long   length,
    long long   lencontent,
    long long*  out_count,    // device pointer, receives number written
    int*        out_error,    // device pointer, caller zero-inits
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_flatten_nextcarry<int>         (tocarry, fromindex, length, lencontent, out_count, out_error, stream); break;
        case 1:  launch_flatten_nextcarry<unsigned int>(tocarry, fromindex, length, lencontent, out_count, out_error, stream); break;
        default: launch_flatten_nextcarry<long long>   (tocarry, fromindex, length, lencontent, out_count, out_error, stream); break;
    }
}

// ── flatten_none2empty ────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_flatten_none2empty_kernel(
    long long*       outoffsets,
    const FROM*      outindex,
    const long long* offsets,
    long long        outindexlength,
    long long        offsetslength,
    int*             out_error
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    outoffsets[0] = offsets[0];
    for (long long i = 0; i < outindexlength; i++) {
        long long idx = (long long)outindex[i];
        if (idx < 0) {
            outoffsets[i + 1] = outoffsets[i];
        } else if (idx + 1 >= offsetslength) {
            atomicCAS(out_error, 0, 1);
            return;
        } else {
            long long count = offsets[idx + 1] - offsets[idx];
            outoffsets[i + 1] = outoffsets[i] + count;
        }
    }
}

template <typename FROM>
static void launch_flatten_none2empty(
    long long*       outoffsets,
    const void*      outindex,
    const long long* offsets,
    long long        outindexlength,
    long long        offsetslength,
    int*             out_error,
    hipStream_t      stream
) {
    if (outindexlength <= 0) return;
    hipLaunchKernelGGL(
        indexed_flatten_none2empty_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        outoffsets, (const FROM*)outindex, offsets,
        outindexlength, offsetslength, out_error
    );
}

extern "C" void awkward_hip_indexed_array_flatten_none2empty(
    long long*       outoffsets,
    const void*      outindex,
    const long long* offsets,
    long long        outindexlength,
    long long        offsetslength,
    int*             out_error,    // device pointer, caller zero-inits
    int              dtype_code,
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0:  launch_flatten_none2empty<int>         (outoffsets, outindex, offsets, outindexlength, offsetslength, out_error, stream); break;
        case 1:  launch_flatten_none2empty<unsigned int>(outoffsets, outindex, offsets, outindexlength, offsetslength, out_error, stream); break;
        default: launch_flatten_none2empty<long long>   (outoffsets, outindex, offsets, outindexlength, offsetslength, out_error, stream); break;
    }
}
