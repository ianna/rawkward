// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Deduplicate consecutive equal strings in a flat byte array.
//
// Scans toptr (strings defined by offsets[0..offsetslength]) and keeps only
// strings that differ from the immediately preceding one.  Kept strings are
// compacted in-place from the front of toptr.  outoffsets receives the new
// boundary positions of the kept strings.  *out_count receives the number of
// entries written to outoffsets (= unique string count + 1).
//
// This is a sequential scan with data-dependent writes; it runs as a single-
// threaded kernel (grid=1, block=1).
//
// Corresponds to CPU: numpy_array_unique_strings_uint8.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

__global__ void unique_strings_uint8_kernel(
    unsigned char* __restrict__   toptr,
    const long long* __restrict__ offsets,
    long long                      offsetslength,
    long long* __restrict__       outoffsets,
    long long* __restrict__       out_count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    long long slen   = 0;   // length of the previous kept string
    long long index  = 0;   // write head in toptr
    long long counter = 0;
    long long start  = 0;   // start of the previous kept string in toptr

    outoffsets[counter] = offsets[0];
    counter++;

    long long n_strings = offsetslength > 0 ? offsetslength - 1 : 0;

    for (long long i = 0; i < n_strings; i++) {
        long long cur_len = offsets[i + 1] - offsets[i];
        int differ = (cur_len != slen) ? 1 : 0;

        if (!differ) {
            for (long long k = 0; k < cur_len; k++) {
                long long j = offsets[i] + k;
                if (toptr[start + k] != toptr[j]) {
                    differ = 1;
                    break;
                }
            }
        }

        if (differ) {
            // Copy this string to the write head.
            for (long long j = offsets[i]; j < offsets[i + 1]; j++) {
                toptr[index] = toptr[j];
                index++;
            }
            start = offsets[i];          // remember where this string was
            outoffsets[counter] = index;
            counter++;
        }
        slen = cur_len;
    }

    *out_count = counter;
}

extern "C" void awkward_hip_unique_strings_uint8(
    unsigned char*   toptr,
    const long long* offsets,
    long long        offsetslength,
    long long*       outoffsets,
    long long*       out_count,   // device pointer; receives counter value
    hipStream_t      stream
) {
    hipLaunchKernelGGL(
        unique_strings_uint8_kernel,
        dim3(1), dim3(1), 0, stream,
        toptr, offsets, offsetslength, outoffsets, out_count
    );
    HIP_CHECK(hipGetLastError());
}
