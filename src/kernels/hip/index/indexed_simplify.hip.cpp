// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Compose two index arrays into a single flattened index.
//
// For each i:
//   j = outerindex[i]
//   j < 0            →  toindex[i] = -1
//   j >= innerlength →  atomicCAS(out_error, 0, 1)  (out-of-range)
//   otherwise        →  toindex[i] = (long long)innerindex[j]
//
// `toindex` is always i64.
// `outer_dtype` selects outerindex type: 0=i32, 1=u32, 2=i64.
// `inner_dtype` selects innerindex type: 0=i32, 1=u32, 2=i64.
// `out_error` is a device int (caller zero-inits); 1 = out-of-range error.
//
// One GPU thread per element.
//
// Corresponds to CPU: indexed_simplify (indexed_array_simplify).

#include <hip/hip_runtime.h>

template <typename OUTER, typename INNER>
__global__ void indexed_simplify_kernel(
    long long*    toindex,
    const OUTER*  outerindex,
    const INNER*  innerindex,
    long long     outerlength,
    long long     innerlength,
    int*          out_error
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outerlength) return;
    long long j = (long long)outerindex[i];
    if (j < 0) {
        toindex[i] = -1;
    } else if (j >= innerlength) {
        atomicCAS(out_error, 0, 1);
    } else {
        toindex[i] = (long long)innerindex[j];
    }
}

// Dispatch table: outer × inner = 9 combinations.
// outer_dtype: 0=i32, 1=u32, 2=i64
// inner_dtype: 0=i32, 1=u32, 2=i64

template <typename OUTER, typename INNER>
static void launch_simplify(
    long long*  toindex,
    const void* outerindex,
    const void* innerindex,
    long long   outerlength,
    long long   innerlength,
    int*        out_error,
    hipStream_t stream
) {
    if (outerlength <= 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)((outerlength + 255) / 256));
    hipLaunchKernelGGL(
        (indexed_simplify_kernel<OUTER, INNER>),
        grid, block, 0, stream,
        toindex,
        (const OUTER*)outerindex,
        (const INNER*)innerindex,
        outerlength, innerlength,
        out_error
    );
}

extern "C" void awkward_hip_indexed_array_simplify(
    long long*  toindex,
    const void* outerindex,
    const void* innerindex,
    long long   outerlength,
    long long   innerlength,
    int         outer_dtype,   // 0=i32, 1=u32, 2=i64
    int         inner_dtype,   // 0=i32, 1=u32, 2=i64
    int*        out_error,     // device pointer, caller zero-inits
    hipStream_t stream
) {
#define DISPATCH_INNER(OUTER) \
    switch (inner_dtype) { \
        case 0: launch_simplify<OUTER, int>         (toindex, outerindex, innerindex, outerlength, innerlength, out_error, stream); break; \
        case 1: launch_simplify<OUTER, unsigned int>(toindex, outerindex, innerindex, outerlength, innerlength, out_error, stream); break; \
        default: launch_simplify<OUTER, long long>  (toindex, outerindex, innerindex, outerlength, innerlength, out_error, stream); break; \
    }

    switch (outer_dtype) {
        case 0: DISPATCH_INNER(int)          break;
        case 1: DISPATCH_INNER(unsigned int) break;
        default: DISPATCH_INNER(long long)   break;
    }
#undef DISPATCH_INNER
}
