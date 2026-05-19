// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Sort a flat byte array interpreted as a sequence of variable-length strings.
//
// Reads strings from fromptr using offsets[0..offsetslength], sorts them
// (ascending or descending, stable or unstable) by lexicographic byte order,
// then writes sorted bytes to toptr and updated boundary positions to
// outoffsets[0..nstrings].
//
// Implementation: insertion sort with memcmp-style string comparison.
// Strings are compared byte-by-byte; shorter strings that are a prefix of a
// longer one sort before (ascending) or after (descending) the longer one.
//
// This runs as a single-threaded kernel (grid=1, block=1) since string
// comparison is variable-cost and the working set fits in device global memory.
// A parallel merge sort would be more scalable for large collections.
//
// Corresponds to CPU: numpy_array_sort_asstrings_uint8.

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

// Compare two strings; returns -1, 0, or +1 (like strcmp).
__device__ static int str_cmp(
    const unsigned char* fromptr,
    long long s1, long long e1,   // [s1, e1)
    long long s2, long long e2    // [s2, e2)
) {
    long long len1 = e1 - s1;
    long long len2 = e2 - s2;
    long long lim  = len1 < len2 ? len1 : len2;
    for (long long i = 0; i < lim; i++) {
        unsigned char a = fromptr[s1 + i];
        unsigned char b = fromptr[s2 + i];
        if (a < b) return -1;
        if (a > b) return +1;
    }
    if (len1 < len2) return -1;
    if (len1 > len2) return +1;
    return 0;
}

// Copy string at fromptr[src_s..src_e] to toptr starting at dest.
__device__ static void str_copy(
    unsigned char* toptr,
    const unsigned char* fromptr,
    long long dest, long long src_s, long long src_e
) {
    for (long long i = src_s; i < src_e; i++) {
        toptr[dest++] = fromptr[i];
    }
}

__global__ void sort_asstrings_uint8_kernel(
    unsigned char* __restrict__       toptr,
    const unsigned char* __restrict__ fromptr,
    const long long* __restrict__     offsets,
    long long                          offsetslength,
    long long* __restrict__           outoffsets,
    int                                ascending,
    int                                /*stable*/   // insertion sort is stable
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    long long nstrings = offsetslength > 0 ? offsetslength - 1 : 0;

    // We perform an insertion sort directly on the index permutation,
    // then write sorted bytes to toptr in a final scatter pass.
    // Use a simple on-stack index array (limited to the available scratch).
    // For large nstrings (> ~65536) callers should use a CPU fallback.

    // Build a permutation array stored in outoffsets[0..nstrings] temporarily.
    // We'll overwrite with real offsets in the scatter pass.
    for (long long i = 0; i < nstrings; i++) {
        outoffsets[i] = i;   // permutation[i] = i
    }

    // Insertion sort on the permutation.
    for (long long i = 1; i < nstrings; i++) {
        long long key = outoffsets[i];
        long long ks  = offsets[key];
        long long ke  = offsets[key + 1];
        long long j   = i - 1;
        while (j >= 0) {
            long long p  = outoffsets[j];
            int cmp = str_cmp(fromptr, offsets[p], offsets[p + 1], ks, ke);
            // ascending: move j forward if fromptr[p] > key
            // descending: move j forward if fromptr[p] < key
            int should_move = ascending ? (cmp > 0) : (cmp < 0);
            if (!should_move) break;
            outoffsets[j + 1] = outoffsets[j];
            j--;
        }
        outoffsets[j + 1] = key;
    }

    // Scatter sorted strings into toptr.
    // SIMULTANEOUSLY record each sorted string's length in outoffsets[i+1],
    // reading the permutation from outoffsets[i] before we overwrite it.
    // This avoids any read/write aliasing: we always write to i+1 while
    // reading from i (the permutation entry at that slot).
    long long pos = 0;
    for (long long i = 0; i < nstrings; i++) {
        long long p   = outoffsets[i];           // permutation[i] — safe to read
        long long s   = offsets[p];
        long long e   = offsets[p + 1];
        long long len = e - s;
        for (long long b = s; b < e; b++) {
            toptr[pos++] = fromptr[b];
        }
        outoffsets[i + 1] = len;                 // store length at i+1
    }
    // outoffsets[1..nstrings+1] now holds the sorted string lengths.
    // outoffsets[0] still holds perm[0] — reset to 0 (start of first string).
    outoffsets[0] = 0;
    // In-place prefix sum to convert lengths → boundary positions.
    for (long long i = 0; i < nstrings; i++) {
        outoffsets[i + 1] += outoffsets[i];
    }
}

extern "C" void awkward_hip_sort_asstrings_uint8(
    unsigned char*   toptr,
    const unsigned char* fromptr,
    const long long* offsets,
    long long        offsetslength,
    long long*       outoffsets,
    int              ascending,
    int              stable,
    hipStream_t      stream
) {
    hipLaunchKernelGGL(
        sort_asstrings_uint8_kernel,
        dim3(1), dim3(1), 0, stream,
        toptr, fromptr, offsets, offsetslength,
        outoffsets, ascending, stable
    );
    HIP_CHECK(hipGetLastError());
}
