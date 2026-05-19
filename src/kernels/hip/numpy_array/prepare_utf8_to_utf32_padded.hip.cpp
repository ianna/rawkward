// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Compute the maximum UTF-8 codepoint count across all sublists.
//
// Each thread handles one sublist: it walks the UTF-8 bytes in that sublist,
// counts codepoints, then updates the global maximum with an atomicMax.
// The caller must zero-initialise *out before the kernel launch.
//
// Invalid leading bytes (w == 0) are gracefully skipped by advancing 1 byte,
// matching the CPU reference behaviour.
//
// Corresponds to CPU: numpy_array_prepare_utf8_to_utf32_padded_int64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// UTF-8 leading-byte classification — mirrors cpu/unicode.rs.
__device__ __forceinline__ int utf8_codepoint_size(unsigned char b) {
    if ((b & 0x80u) == 0x00u) return 1;  // 0xxxxxxx
    if ((b & 0xE0u) == 0xC0u) return 2;  // 110xxxxx
    if ((b & 0xF0u) == 0xE0u) return 3;  // 1110xxxx
    if ((b & 0xF8u) == 0xF0u) return 4;  // 11110xxx
    return 0;                             // continuation / invalid
}

__global__ void prepare_utf8_to_utf32_padded_kernel(
    const unsigned char* __restrict__ fromptr,
    const long long* __restrict__     fromoffsets,
    long long                          offsetslength,
    long long* __restrict__           out           // device pointer to scalar
) {
    long long k = blockIdx.x * blockDim.x + threadIdx.x;
    long long n_lists = offsetslength - 1;
    if (k >= n_lists) return;

    long long i   = fromoffsets[k];
    long long end = fromoffsets[k + 1];
    long long count = 0;

    while (i < end) {
        int w = utf8_codepoint_size(fromptr[i]);
        if (w == 0) w = 1;   // skip invalid bytes gracefully
        i += w;
        count++;
    }

    // Update the global maximum atomically.
    atomicMax(out, count);
}

extern "C" void awkward_hip_prepare_utf8_to_utf32_padded(
    const unsigned char* fromptr,
    const long long*     fromoffsets,
    long long            offsetslength,
    long long*           out,           // device pointer; caller must zero-init
    hipStream_t          stream
) {
    long long n_lists = offsetslength - 1;
    if (n_lists <= 0) return;

    int threads = 256;
    int blocks  = (n_lists + threads - 1) / threads;

    hipLaunchKernelGGL(
        prepare_utf8_to_utf32_padded_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        fromptr, fromoffsets, offsetslength, out
    );

    HIP_CHECK(hipGetLastError());
}
