// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause
#include <hip/hip_runtime.h>

template <typename T>
__device__ inline void segmented_countnonzero_body(
    const T* data, const long long* offsets, long long* out,
    long long n_segments, long long seg)
{
    if (seg >= n_segments) return;
    long long start = offsets[seg], end = offsets[seg + 1];
    long long count = 0;
    for (long long i = start; i < end; ++i)
        if (data[i] != T(0)) ++count;
    out[seg] = count;
}

extern "C" __global__ void segmented_countnonzero_f32(
    const float* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_countnonzero_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_countnonzero_f64(
    const double* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_countnonzero_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_countnonzero_i32(
    const int* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_countnonzero_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_countnonzero_i64(
    const long long* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_countnonzero_body(data, offsets, out, n_segments, seg); }
