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
//    Single-threaded iterative DFS matching the CPU recursive combinations_step:
//      At depth j: propagate ALL deeper positions k>j from fromindex[j], then
//      either emit (if leaf) or descend. This matches:
//        CPU: for k in (j+1)..n { fromindex[k] = fromindex[j] + (k-j); }
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

    tooffsets[i + 1] = combinationslen;  // raw count; converted to prefix sum in pass 2
    atomicAdd((unsigned long long*)out_total, (unsigned long long)combinationslen);
}

// Single-threaded prefix-sum pass to convert per-list counts → cumulative offsets.
__global__ void combinations_length_prefix_sum_kernel(
    long long* tooffsets,
    long long  length
) {
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
    dim3 block(256);
    dim3 grid((unsigned int)((length + 255) / 256));
    hipLaunchKernelGGL(
        list_array_combinations_length_kernel<C>,
        grid, block, 0, stream,
        tooffsets, (const C*)fromstarts, (const C*)fromstops,
        length, n, replacement, out_total
    );
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
// Iterative DFS that correctly mirrors the CPU recursive combinations_step:
//   At depth j, after confirming fromindex[j] < stop:
//     Propagate ALL deeper positions: fromindex[k] = fromindex[j] + (k-j)  (no replacement)
//                                                  = fromindex[j]           (with replacement)
//     If leaf (j+1 == n): emit combination, then advance fromindex[j]++.
//     Otherwise: descend (j++).
//   Backtrack when fromindex[j] >= stop: j--, then fromindex[j]++ (parent advances).
//
// tocarry layout: tocarry[col * total_combinations + k]

template <typename C>
__global__ void list_array_combinations_kernel(
    long long*  tocarry,       // n * total_combinations
    const C*    fromstarts,
    const C*    fromstops,
    long long   length,
    long long   n,
    int         replacement,
    long long   total_combinations
) {
    const int MAXN = 64;
    long long fromindex[MAXN];
    long long toindex  [MAXN];   // write position per column

    for (long long c = 0; c < n; ++c) toindex[c] = 0;

    for (long long i = 0; i < length; ++i) {
        long long start = (long long)fromstarts[i];
        long long stop  = (long long)fromstops [i];

        if (stop <= start) continue;

        // Initialise depth 0.
        long long j = 0;
        fromindex[0] = start;

        while (j >= 0) {
            if (fromindex[j] >= stop) {
                // Exhausted at this depth — backtrack.
                --j;
                if (j >= 0) ++fromindex[j];
                continue;
            }

            // Propagate ALL deeper positions from the current depth j.
            // This matches: for k in (j+1)..n { fromindex[k] = fromindex[j] + (k-j); }
            for (long long k = j + 1; k < n; ++k) {
                fromindex[k] = replacement ? fromindex[j]
                                           : fromindex[j] + (k - j);
            }

            if (j + 1 == n) {
                // Leaf: emit the combination.
                for (long long c = 0; c < n; ++c) {
                    tocarry[c * total_combinations + toindex[c]] = fromindex[c];
                    ++toindex[c];
                }
                // Advance at the current (leaf) depth.
                ++fromindex[j];
            } else {
                // Descend.
                ++j;
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
