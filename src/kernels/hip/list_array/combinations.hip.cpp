// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Two kernels for n-combinations over ListArray lists.
//
// 1. awkward_hip_list_array_combinations_length
//    For each list i, compute C(size, n) or C(size+n-1, n) (with replacement),
//    write cumulative sums into tooffsets, add to *out_total.
//    One GPU thread per list; atomicAdd for *out_total.
//
// 2. awkward_hip_list_array_combinations
//    Enumerate all n-combinations per list, write n flat tocarry columns
//    laid out as tocarry[col * total_combinations + k].
//    Single-threaded (inherently sequential enumeration).
//
// dtype_code: 0=i32, 1=u32, 2=i64 (for starts/stops).
//
// Corresponds to CPU: list_array_combinations_length, list_array_combinations.

#include <hip/hip_runtime.h>

// ── 1. combinations_length ────────────────────────────────────────────────────

template <typename C>
__global__ void list_array_combinations_length_kernel(
    long long*  tooffsets,
    const C*    fromstarts,
    const C*    fromstops,
    long long   length,
    long long   n,
    int         replacement,
    long long*  out_total
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i > length) return;   // thread 0..length (inclusive) to set tooffsets[0]

    if (i == 0) tooffsets[0] = 0LL;

    if (i >= length) return;

    long long size = (long long)fromstops[i] - (long long)fromstarts[i];
    if (replacement) size += n - 1;

    long long thisn = n;
    long long combinationslen;
    if (thisn > size) {
        combinationslen = 0;
    } else if (thisn == size) {
        combinationslen = 1;
    } else {
        if (thisn * 2 > size) thisn = size - thisn;
        long long c = size;
        for (long long j = 2; j <= thisn; ++j) {
            c *= size - j + 1;
            c /= j;
        }
        combinationslen = c;
    }

    tooffsets[i + 1] = combinationslen;  // store per-list count first
    atomicAdd((unsigned long long*)out_total, (unsigned long long)combinationslen);
}

// Convert per-list counts stored in tooffsets[1..length] to prefix sums.
__global__ void combinations_length_prefix_sum_kernel(
    long long* tooffsets,
    long long  length
) {
    // Single-threaded prefix sum pass.
    for (long long i = 0; i < length; ++i) {
        tooffsets[i + 1] += tooffsets[i];
    }
}

template <typename C>
static void launch_combinations_length(
    long long* tooffsets, const void* fromstarts, const void* fromstops,
    long long length, long long n, int replacement,
    long long* out_total, hipStream_t stream
) {
    if (length <= 0) return;
    // Step 1: fill per-list counts + accumulate total (parallel)
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_combinations_length_kernel<C>,
        grid, block, 0, stream,
        tooffsets, (const C*)fromstarts, (const C*)fromstops,
        length, n, replacement, out_total
    );
    // Step 2: convert counts to prefix sums (single-threaded)
    hipLaunchKernelGGL(
        combinations_length_prefix_sum_kernel,
        dim3(1), dim3(1), 0, stream,
        tooffsets, length
    );
}

extern "C" void awkward_hip_list_array_combinations_length(
    long long*  tooffsets,
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   n,
    int         replacement,
    int         dtype_code,
    long long*  out_total,   // device pointer, caller zero-inits
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_combinations_length<int>         (tooffsets, fromstarts, fromstops, length, n, replacement, out_total, stream); break;
        case 1:  launch_combinations_length<unsigned int>(tooffsets, fromstarts, fromstops, length, n, replacement, out_total, stream); break;
        default: launch_combinations_length<long long>   (tooffsets, fromstarts, fromstops, length, n, replacement, out_total, stream); break;
    }
}

// ── 2. combinations (enumerate, single-threaded) ──────────────────────────────
//
// tocarry layout: tocarry[col * total_combinations + k]
// Iterative simulation of the recursive combinations step.

template <typename C>
__global__ void list_array_combinations_kernel(
    long long*  tocarry,       // n_cols * total_combinations
    const C*    fromstarts,
    const C*    fromstops,
    long long   length,
    long long   n,
    int         replacement,
    long long   total_combinations   // pre-computed grand total
) {
    // Stack-based iterative combination enumeration.
    // Maximum n supported by fixed-size stack arrays.
    // We use a VLA-like approach via local arrays capped at a reasonable n.
    // In practice n is small (2..5); use fixed-size 64.
    const int MAXN = 64;
    long long fromindex[MAXN];
    long long toindex  [MAXN];   // write positions per column

    for (long long c = 0; c < n; ++c) toindex[c] = 0;

    for (long long i = 0; i < length; ++i) {
        long long start = (long long)fromstarts[i];
        long long stop  = (long long)fromstops [i];

        if (stop <= start) continue;

        // Iterative depth-first enumeration.
        // fromindex[d] = current value at depth d.
        for (long long c = 0; c < n; ++c) fromindex[c] = start;

        // depth = current recursion level
        long long depth = 0;
        while (depth >= 0) {
            if (depth == n) {
                // Emit combination
                for (long long c = 0; c < n; ++c) {
                    tocarry[c * total_combinations + toindex[c]] = fromindex[c];
                    ++toindex[c];
                }
                --depth;
                // Advance the value at parent depth
                ++fromindex[depth];
                continue;
            }
            long long val = fromindex[depth];
            if (val >= stop) {
                // Backtrack
                --depth;
                if (depth >= 0) ++fromindex[depth];
                continue;
            }
            // Descend
            ++depth;
            if (depth < n) {
                fromindex[depth] = replacement ? val : val + 1;
            }
        }
    }
}

template <typename C>
static void launch_combinations(
    long long* tocarry, const void* fromstarts, const void* fromstops,
    long long length, long long n, int replacement,
    long long total_combinations, hipStream_t stream
) {
    if (length <= 0 || total_combinations <= 0) return;
    hipLaunchKernelGGL(
        list_array_combinations_kernel<C>,
        dim3(1), dim3(1), 0, stream,
        tocarry, (const C*)fromstarts, (const C*)fromstops,
        length, n, replacement, total_combinations
    );
}

extern "C" void awkward_hip_list_array_combinations(
    long long*  tocarry,             // n * total_combinations flat array
    const void* fromstarts,
    const void* fromstops,
    long long   length,
    long long   n,
    int         replacement,
    long long   total_combinations,  // caller pre-computes via combinations_length
    int         dtype_code,
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0:  launch_combinations<int>         (tocarry, fromstarts, fromstops, length, n, replacement, total_combinations, stream); break;
        case 1:  launch_combinations<unsigned int>(tocarry, fromstarts, fromstops, length, n, replacement, total_combinations, stream); break;
        default: launch_combinations<long long>   (tocarry, fromstarts, fromstops, length, n, replacement, total_combinations, stream); break;
    }
}
