// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two-pass rearrangement of a shifted index array.
//
// fromoffsets is a cumulative array (same layout as the reduce/ offsets):
//   - fromoffsets[i+1] - fromoffsets[i] = number of elements in segment i
//   - fromoffsets[i] is also the offset value added to every element in
//     that segment's portion of toptr
//
// toptr elements for segment i occupy the contiguous block:
//   toptr[fromoffsets[i] - fromoffsets[0]  ..  fromoffsets[i+1] - fromoffsets[0]]
//
// Pass 1 (one thread per segment):
//   For each segment i, add fromoffsets[i] to every toptr element in that
//   segment's block.
//
// Pass 2 (one thread per data element):
//   toptr[j] += fromshifts[toptr[j]] - fromstarts[fromparents[j]]
//
// Corresponds to CPU: numpy_array_rearrange_shifted_toint64_fromint64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// Pass 1: one thread per segment.
__global__ void rearrange_shifted_pass1_kernel(
    long long* __restrict__       toptr,
    const long long* __restrict__ fromoffsets,
    long long                      n_lists,
    long long                      base           // = fromoffsets[0]
) {
    long long seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_lists) return;

    long long off        = fromoffsets[seg];           // value to add
    long long seg_start  = off - base;                 // start index in toptr
    long long seg_end    = fromoffsets[seg + 1] - base;// end index in toptr

    for (long long i = seg_start; i < seg_end; i++) {
        toptr[i] += off;
    }
}

// Pass 2: one thread per data element.
__global__ void rearrange_shifted_pass2_kernel(
    long long* __restrict__       toptr,
    const long long* __restrict__ fromshifts,
    const long long* __restrict__ fromparents,
    const long long* __restrict__ fromstarts,
    long long                      length
) {
    long long j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= length) return;

    long long idx    = toptr[j];
    long long parent = fromparents[j];
    toptr[j] = idx + fromshifts[idx] - fromstarts[parent];
}

extern "C" void awkward_hip_rearrange_shifted(
    long long*       toptr,
    const long long* fromshifts,
    const long long* fromoffsets,
    const long long* fromparents,
    const long long* fromstarts,
    long long        offsetslength,   // number of offsets = n_lists + 1
    long long        length,          // total elements in toptr / fromparents
    hipStream_t      stream
) {
    if (length == 0) return;

    long long n_lists = offsetslength - 1;
    int threads = 256;

    // We need fromoffsets[0] on the host to compute base.
    // fromoffsets lives on the device; copy the first element.
    long long base = 0;
    hipMemcpyAsync(&base, fromoffsets, sizeof(long long),
                   hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    // Pass 1
    int blocks1 = (n_lists + threads - 1) / threads;
    hipLaunchKernelGGL(
        rearrange_shifted_pass1_kernel,
        dim3(blocks1), dim3(threads), 0, stream,
        toptr, fromoffsets, n_lists, base
    );
    HIP_CHECK(hipGetLastError());

    // Pass 2
    int blocks2 = (length + threads - 1) / threads;
    hipLaunchKernelGGL(
        rearrange_shifted_pass2_kernel,
        dim3(blocks2), dim3(threads), 0, stream,
        toptr, fromshifts, fromparents, fromstarts, length
    );
    HIP_CHECK(hipGetLastError());
}
