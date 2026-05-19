// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Three rpad-and-clip index-building kernels.
//
// ── awkward_hip_indexedoption_rpad_and_clip_mask_axis1 ────────────────────────
//   For each i:
//     frommask[i] != 0  →  toindex[i] = -1
//     frommask[i] == 0  →  toindex[i] = count, count++
//   Single-threaded (sequential count required).
//
// ── awkward_hip_index_rpad_and_clip_axis0 ────────────────────────────────────
//   Fill toindex[0..shorter-1] = 0..shorter-1, toindex[shorter..target-1] = -1.
//   shorter = min(target, length).  One GPU thread per element.
//
// ── awkward_hip_index_rpad_and_clip_axis1 ────────────────────────────────────
//   tostarts[i] = i * target, tostops[i] = (i+1) * target.
//   One GPU thread per element (0..length-1).
//
// Corresponds to CPU: indexedoption_rpad_and_clip_mask_axis1,
//                     index_rpad_and_clip_axis0, index_rpad_and_clip_axis1.

#include <hip/hip_runtime.h>

// ── indexedoption_rpad_and_clip_mask_axis1 ────────────────────────────────────

__global__ void indexedoption_rpad_and_clip_mask_axis1_kernel(
    long long*         toindex,
    const signed char* frommask,
    long long          length
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long count = 0;
    for (long long i = 0; i < length; i++) {
        if (frommask[i] != 0) {
            toindex[i] = -1;
        } else {
            toindex[i] = count++;
        }
    }
}

extern "C" void awkward_hip_indexedoption_rpad_and_clip_mask_axis1(
    long long*         toindex,
    const signed char* frommask,
    long long          length,
    hipStream_t        stream
) {
    if (length <= 0) return;
    hipLaunchKernelGGL(
        indexedoption_rpad_and_clip_mask_axis1_kernel,
        dim3(1), dim3(1), 0, stream,
        toindex, frommask, length
    );
}

// ── index_rpad_and_clip_axis0 ─────────────────────────────────────────────────

__global__ void index_rpad_and_clip_axis0_kernel(
    long long* toindex,
    long long  target,
    long long  shorter    // min(target, length)
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= target) return;
    toindex[i] = (i < shorter) ? i : -1LL;
}

extern "C" void awkward_hip_index_rpad_and_clip_axis0(
    long long*  toindex,
    long long   target,
    long long   length,
    hipStream_t stream
) {
    if (target <= 0) return;
    long long shorter = (target < length) ? target : length;
    dim3 block(256);
    dim3 grid((unsigned int)((target + 255) / 256));
    hipLaunchKernelGGL(
        index_rpad_and_clip_axis0_kernel,
        grid, block, 0, stream,
        toindex, target, shorter
    );
}

// ── index_rpad_and_clip_axis1 ─────────────────────────────────────────────────

__global__ void index_rpad_and_clip_axis1_kernel(
    long long* tostarts,
    long long* tostops,
    long long  target,
    long long  length
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    tostarts[i] = i * target;
    tostops[i]  = (i + 1) * target;
}

extern "C" void awkward_hip_index_rpad_and_clip_axis1(
    long long*  tostarts,
    long long*  tostops,
    long long   target,
    long long   length,
    hipStream_t stream
) {
    if (length <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        index_rpad_and_clip_axis1_kernel,
        grid, block, 0, stream,
        tostarts, tostops, target, length
    );
}
