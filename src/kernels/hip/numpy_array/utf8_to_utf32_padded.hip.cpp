// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Decode every UTF-8 sublist into padded UTF-32 codepoints.
//
// For each sublist k in fromoffsets[0..offsetslength-1], decode up to
// maxcodepoints codepoints into toptr[k*maxcodepoints .. (k+1)*maxcodepoints],
// zero-padding any remaining slots.
//
// toptr must have (offsetslength - 1) * maxcodepoints uint32 elements.
//
// Error handling: if an invalid leading byte is encountered, the kernel writes
// the 1-based sublist index (k+1) into *out_error and stops processing that
// sublist (subsequent sublists continue normally).  On success *out_error = 0.
// The caller must zero-initialise *out_error before the launch.
//
// One thread per sublist.
//
// Corresponds to CPU: numpy_array_utf8_to_utf32_padded_int64.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// UTF-8 payload masks.
#define ONE_BYTE_PAYLOAD    0x7Fu
#define TWO_BYTE_PAYLOAD    0x1Fu
#define THREE_BYTE_PAYLOAD  0x0Fu
#define FOUR_BYTE_PAYLOAD   0x07u
#define CONT_PAYLOAD        0x3Fu

__device__ __forceinline__ int utf8_cp_size(unsigned char b) {
    if ((b & 0x80u) == 0x00u) return 1;
    if ((b & 0xE0u) == 0xC0u) return 2;
    if ((b & 0xF0u) == 0xE0u) return 3;
    if ((b & 0xF8u) == 0xF0u) return 4;
    return 0;
}

__global__ void utf8_to_utf32_padded_kernel(
    const unsigned char* __restrict__ fromptr,
    const long long* __restrict__     fromoffsets,
    long long                          offsetslength,
    long long                          maxcodepoints,
    unsigned int* __restrict__        toptr,
    int* __restrict__                 out_error   // device scalar; 0 = ok
) {
    long long k      = blockIdx.x * blockDim.x + threadIdx.x;
    long long n_lists = offsetslength - 1;
    if (k >= n_lists) return;

    long long byte_pos   = fromoffsets[k];
    long long byte_end   = fromoffsets[k + 1];
    long long out_base   = k * maxcodepoints;
    long long n_sublist  = 0;

    while (byte_pos < byte_end) {
        unsigned char b0 = fromptr[byte_pos];
        int w = utf8_cp_size(b0);

        if (w == 0) {
            // Invalid leading byte — record error and stop this sublist.
            atomicMax(out_error, (int)(k + 1));
            break;
        }

        unsigned int cp;
        switch (w) {
            case 1:
                cp = (unsigned int)b0 & ONE_BYTE_PAYLOAD;
                break;
            case 2:
                cp = (((unsigned int)b0 & TWO_BYTE_PAYLOAD) << 6)
                   | ((unsigned int)fromptr[byte_pos + 1] & CONT_PAYLOAD);
                break;
            case 3:
                cp = (((unsigned int)b0 & THREE_BYTE_PAYLOAD) << 12)
                   | (((unsigned int)fromptr[byte_pos + 1] & CONT_PAYLOAD) << 6)
                   | ((unsigned int)fromptr[byte_pos + 2] & CONT_PAYLOAD);
                break;
            default: // 4
                cp = (((unsigned int)b0 & FOUR_BYTE_PAYLOAD) << 18)
                   | (((unsigned int)fromptr[byte_pos + 1] & CONT_PAYLOAD) << 12)
                   | (((unsigned int)fromptr[byte_pos + 2] & CONT_PAYLOAD) << 6)
                   | ((unsigned int)fromptr[byte_pos + 3] & CONT_PAYLOAD);
                break;
        }

        toptr[out_base + n_sublist] = cp;
        n_sublist++;
        byte_pos += w;
    }

    // Zero-pad remaining slots.
    for (long long i = n_sublist; i < maxcodepoints; i++) {
        toptr[out_base + i] = 0u;
    }
}

extern "C" void awkward_hip_utf8_to_utf32_padded(
    const unsigned char* fromptr,
    const long long*     fromoffsets,
    long long            offsetslength,
    long long            maxcodepoints,
    unsigned int*        toptr,
    int*                 out_error,   // device pointer; caller must zero-init
    hipStream_t          stream
) {
    long long n_lists = offsetslength - 1;
    if (n_lists <= 0) return;

    int threads = 256;
    int blocks  = (n_lists + threads - 1) / threads;

    hipLaunchKernelGGL(
        utf8_to_utf32_padded_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        fromptr, fromoffsets, offsetslength, maxcodepoints, toptr, out_error
    );

    HIP_CHECK(hipGetLastError());
}
