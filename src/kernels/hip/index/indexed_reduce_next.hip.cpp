// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Four reduction-step kernels for IndexedArray.
//
// ── awkward_hip_indexed_array_reduce_next ─────────────────────────────────────
//   For each i: index[i]>=0 → nextcarry[k]=index[i], nextparents[k]=parents[i],
//               outindex[i]=k, k++.  index[i]<0 → outindex[i]=-1.
//   *out_count receives k.  Single-threaded (sequential k).
//
// ── awkward_hip_indexed_array_reduce_next_fix_offsets ─────────────────────────
//   Copy starts[0..startslength] to outoffsets[0..startslength],
//   then write outindexlength to outoffsets[startslength].
//   Parallel (one thread per entry; startslength+1 threads total).
//
// ── awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts ─────────────────
//   For each i: non-null → nextshifts[k]=nullsum, k++; null → nullsum++.
//   *out_count receives k.  Single-threaded.
//
// ── awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts ──────
//   Like above but non-null → nextshifts[k] = shifts[i] + nullsum.
//   Single-threaded.
//
// dtype_code for index: 0=i32, 1=u32, 2=i64.
//
// Corresponds to CPU: indexed_reduce_next, indexed_reduce_next_fix_offsets,
//   indexed_reduce_next_nonlocal_nextshifts,
//   indexed_reduce_next_nonlocal_nextshifts_fromshifts.

#include <hip/hip_runtime.h>

// ── reduce_next ───────────────────────────────────────────────────────────────

template <typename FROM>
__global__ void indexed_reduce_next_kernel(
    long long*         nextcarry,
    long long*         nextparents,
    long long*         outindex,
    const FROM*        index,
    const long long*   parents,
    long long          length,
    long long*         out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long k = 0;
    for (long long i = 0; i < length; i++) {
        long long j = (long long)index[i];
        if (j >= 0) {
            nextcarry[k]   = j;
            nextparents[k] = parents[i];
            outindex[i]    = k;
            k++;
        } else {
            outindex[i] = -1;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_reduce_next(
    long long*         nextcarry,
    long long*         nextparents,
    long long*         outindex,
    const void*        index,
    const long long*   parents,
    long long          length,
    long long*         out_count,
    hipStream_t        stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        indexed_reduce_next_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        nextcarry, nextparents, outindex,
        (const FROM*)index, parents, length, out_count
    );
}

extern "C" void awkward_hip_indexed_array_reduce_next(
    long long*         nextcarry,
    long long*         nextparents,
    long long*         outindex,
    const void*        index,
    const long long*   parents,
    long long          length,
    long long*         out_count,    // device pointer, receives k
    int                dtype_code,
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_reduce_next<int>         (nextcarry, nextparents, outindex, index, parents, length, out_count, stream); break;
        case 1:  launch_reduce_next<unsigned int>(nextcarry, nextparents, outindex, index, parents, length, out_count, stream); break;
        default: launch_reduce_next<long long>   (nextcarry, nextparents, outindex, index, parents, length, out_count, stream); break;
    }
}

// ── reduce_next_fix_offsets ───────────────────────────────────────────────────
// Parallel: one thread per slot (startslength + 1 total).

__global__ void indexed_reduce_next_fix_offsets_kernel(
    long long*         outoffsets,
    const long long*   starts,
    long long          startslength,
    long long          outindexlength
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < startslength) {
        outoffsets[i] = starts[i];
    } else if (i == startslength) {
        outoffsets[i] = outindexlength;
    }
}

extern "C" void awkward_hip_indexed_array_reduce_next_fix_offsets(
    long long*         outoffsets,
    const long long*   starts,
    long long          startslength,
    long long          outindexlength,
    hipStream_t        stream
) {
    if (startslength < 0) return;
    long long total = startslength + 1;
    dim3 block(256);
    dim3 grid((unsigned int)((total + 255) / 256));
    hipLaunchKernelGGL(
        indexed_reduce_next_fix_offsets_kernel,
        grid, block, 0, stream,
        outoffsets, starts, startslength, outindexlength
    );
}

// ── reduce_next_nonlocal_nextshifts ──────────────────────────────────────────

template <typename FROM>
__global__ void indexed_reduce_next_nonlocal_nextshifts_kernel(
    long long*   nextshifts,
    const FROM*  index,
    long long    length,
    long long*   out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long nullsum = 0, k = 0;
    for (long long i = 0; i < length; i++) {
        if ((long long)index[i] >= 0) {
            nextshifts[k++] = nullsum;
        } else {
            nullsum++;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_reduce_next_nonlocal_nextshifts(
    long long*  nextshifts,
    const void* index,
    long long   length,
    long long*  out_count,
    hipStream_t stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        indexed_reduce_next_nonlocal_nextshifts_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        nextshifts, (const FROM*)index, length, out_count
    );
}

extern "C" void awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts(
    long long*  nextshifts,
    const void* index,
    long long   length,
    long long*  out_count,    // device pointer, receives count of non-null
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_reduce_next_nonlocal_nextshifts<int>         (nextshifts, index, length, out_count, stream); break;
        case 1:  launch_reduce_next_nonlocal_nextshifts<unsigned int>(nextshifts, index, length, out_count, stream); break;
        default: launch_reduce_next_nonlocal_nextshifts<long long>   (nextshifts, index, length, out_count, stream); break;
    }
}

// ── reduce_next_nonlocal_nextshifts_fromshifts ────────────────────────────────

template <typename FROM>
__global__ void indexed_reduce_next_nonlocal_nextshifts_fromshifts_kernel(
    long long*         nextshifts,
    const FROM*        index,
    const long long*   shifts,
    long long          length,
    long long*         out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long nullsum = 0, k = 0;
    for (long long i = 0; i < length; i++) {
        if ((long long)index[i] >= 0) {
            nextshifts[k++] = shifts[i] + nullsum;
        } else {
            nullsum++;
        }
    }
    *out_count = k;
}

template <typename FROM>
static void launch_reduce_next_nonlocal_nextshifts_fromshifts(
    long long*         nextshifts,
    const void*        index,
    const long long*   shifts,
    long long          length,
    long long*         out_count,
    hipStream_t        stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        indexed_reduce_next_nonlocal_nextshifts_fromshifts_kernel<FROM>,
        dim3(1), dim3(1), 0, stream,
        nextshifts, (const FROM*)index, shifts, length, out_count
    );
}

extern "C" void awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts(
    long long*         nextshifts,
    const void*        index,
    const long long*   shifts,
    long long          length,
    long long*         out_count,
    int                dtype_code,
    hipStream_t        stream
) {
    switch (dtype_code) {
        case 0:  launch_reduce_next_nonlocal_nextshifts_fromshifts<int>         (nextshifts, index, shifts, length, out_count, stream); break;
        case 1:  launch_reduce_next_nonlocal_nextshifts_fromshifts<unsigned int>(nextshifts, index, shifts, length, out_count, stream); break;
        default: launch_reduce_next_nonlocal_nextshifts_fromshifts<long long>   (nextshifts, index, shifts, length, out_count, stream); break;
    }
}
