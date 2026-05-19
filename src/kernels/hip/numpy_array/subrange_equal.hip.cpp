// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Check whether all pairs of sub-ranges that have the same length are
// pairwise equal (O(n²) nested comparison, single-threaded kernel).
//
// Faithfully ports the CPU reference:
//   differ = true
//   for i in 0..length-1:
//     leftlen = fromstops[i] - fromstarts[i]
//     for ii in i+1..length-1:
//       rightlen = fromstops[ii] - fromstarts[ii]
//       if leftlen == rightlen:
//         differ = false
//         for j in 0..leftlen:
//           if tmpptr[fromstarts[i]+j] != tmpptr[fromstarts[ii]+j]:
//             differ = true; break
//   *out_equal = !differ ? 1 : 0
//
// Note: the outer loop runs to length-1 (exclusive) and the inner loop also
// runs to length-1, so the last sub-range is never used as a RIGHT operand.
// This matches the C++ and Rust reference implementations.
//
// Single-threaded (grid=1, block=1); this operation is inherently sequential.
// Dtype dispatch via template instantiations.
//
// Corresponds to CPU: numpy_array_subrange_equal (multiple typed variants).

#include <hip/hip_runtime.h>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

template <typename T>
__global__ void subrange_equal_kernel(
    const T* __restrict__         tmpptr,
    const long long* __restrict__ fromstarts,
    const long long* __restrict__ fromstops,
    long long                      length,
    int* __restrict__             out_equal   // 1 = equal, 0 = not equal
) {
    // Single thread only.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int differ = 1;  // 1 = differ (not proven equal yet)

    long long lim = length > 0 ? length - 1 : 0;

    for (long long i = 0; i < lim; i++) {
        long long leftlen = fromstops[i] - fromstarts[i];
        for (long long ii = i + 1; ii < lim; ii++) {
            long long rightlen = fromstops[ii] - fromstarts[ii];
            if (leftlen == rightlen) {
                differ = 0;
                for (long long j = 0; j < leftlen; j++) {
                    if (tmpptr[fromstarts[i] + j] != tmpptr[fromstarts[ii] + j]) {
                        differ = 1;
                        break;
                    }
                }
            }
        }
    }

    *out_equal = differ ? 0 : 1;
}

template <typename T>
static void launch_subrange_equal(
    const void*      tmpptr,
    const long long* fromstarts,
    const long long* fromstops,
    long long        length,
    int*             out_equal,
    hipStream_t      stream
) {
    hipLaunchKernelGGL(
        subrange_equal_kernel<T>,
        dim3(1), dim3(1), 0, stream,
        static_cast<const T*>(tmpptr), fromstarts, fromstops, length, out_equal
    );
    HIP_CHECK(hipGetLastError());
}

extern "C" void awkward_hip_subrange_equal(
    const void*      tmpptr,
    const long long* fromstarts,
    const long long* fromstops,
    long long        length,
    int*             out_equal,    // device pointer; 1=equal, 0=not equal
    int              dtype_code,   // 0=i8, 1=u8, 2=i16, 3=u16, 4=i32, 5=u32
                                   // 6=i64, 7=u64, 8=f32, 9=f64
    hipStream_t      stream
) {
    switch (dtype_code) {
        case 0: launch_subrange_equal<signed char>  (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 1: launch_subrange_equal<unsigned char>(tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 2: launch_subrange_equal<short>        (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 3: launch_subrange_equal<unsigned short>(tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 4: launch_subrange_equal<int>          (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 5: launch_subrange_equal<unsigned int> (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 6: launch_subrange_equal<long long>    (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 7: launch_subrange_equal<unsigned long long>(tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 8: launch_subrange_equal<float>        (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        case 9: launch_subrange_equal<double>       (tmpptr, fromstarts, fromstops, length, out_equal, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
